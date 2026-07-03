"""Student vLLM server with extra ``/trainer/*`` endpoints for tinther.

Wraps vLLM's standard OpenAI-compatible API server and mounts four extra
routes that drive the :class:`TintherWorkerExt` mixed into each vLLM worker:

* ``POST /trainer/init_weight_update_pg`` — workers join the NCCL side-channel
  with the trainer (rank 0 of that group).
* ``POST /trainer/update_weights`` — server iterates the request's
  ``weights`` list and fires one ``collective_rpc("recv_weight", ...)`` per
  entry. The trainer broadcasts the matching tensor on the same group while
  the request is in flight.
* ``POST /trainer/reset_prefix_cache`` — invalidate the engine's prefix
  cache after a weight update so subsequent prompts don't reuse KV blocks
  computed under the old weights. Mirrors TRL's ``/reset_prefix_cache/``
  in ``trl/scripts/vllm_serve.py``.
* ``POST /trainer/close_weight_update_pg`` — drop the side-channel.

Launch (the user runs this; tinther never spawns it)::

    python -m tinker_cookbook.tinther.vllm_student_server \\
        --model HuggingFaceTB/SmolLM2-135M \\
        --tensor-parallel-size 1 \\
        --port 8001 \\
        --worker-extension-cls \\
            tinker_cookbook.tinther.vllm_student_worker.TintherWorkerExt

Required environment (must match the trainer's process)::

    TINTHER_STUDENT_NCCL_MASTER_ADDR=127.0.0.1
    TINTHER_STUDENT_NCCL_MASTER_PORT=29500
    TINTHER_STUDENT_NCCL_WORLD_SIZE=2     # 1 (trainer) + tensor_parallel * pipeline
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger("tinther.student_server")

_WORKER_EXT_CLS = "tinker_cookbook.tinther.vllm_student_worker.TintherWorkerExt"


class _WeightMeta(BaseModel):
    name: str
    dtype: str
    shape: list[int]


class _UpdateWeightsRequest(BaseModel):
    weights: list[_WeightMeta]


def _make_trainer_router(engine_client: Any) -> APIRouter:
    router = APIRouter(prefix="/trainer", tags=["tinther"])

    @router.post("/init_weight_update_pg")
    async def init_weight_update_pg() -> dict[str, Any]:
        results = await engine_client.collective_rpc("init_weight_update_pg")
        return {"ok": True, "results": results}

    @router.post("/update_weights")
    async def update_weights(req: _UpdateWeightsRequest) -> dict[str, Any]:
        # Sequential dispatch: each call blocks until every worker has
        # entered (and exited) the NCCL broadcast for that one tensor.
        # The trainer broadcasts in the same order on its side.
        for w in req.weights:
            await engine_client.collective_rpc(
                "recv_weight", args=(w.name, w.dtype, list(w.shape))
            )
        return {"ok": True, "n": len(req.weights)}

    @router.post("/reset_prefix_cache")
    async def reset_prefix_cache() -> dict[str, Any]:
        # Mirrors TRL's POST /reset_prefix_cache/ — calls the engine-level
        # API directly so prompts after a weight update don't hit KV blocks
        # computed from the previous weights.
        await engine_client.reset_prefix_cache()
        return {"ok": True}

    @router.post("/close_weight_update_pg")
    async def close_weight_update_pg() -> dict[str, Any]:
        results = await engine_client.collective_rpc("close_weight_update_pg")
        return {"ok": True, "results": results}

    return router


def _patch_prometheus_fastapi_instrumentator() -> None:
    """Handle FastAPI's included-router wrapper in older prometheus middleware.

    The vLLM API app mounts a large APIRouter before adding
    prometheus-fastapi-instrumentator middleware. FastAPI 0.122 stores that
    router as a private ``_IncludedRouter`` route object without a ``path``
    attribute, while the installed instrumentator version assumes every
    matching route has one. Without this compatibility shim, every student HTTP
    request fails in middleware before it reaches vLLM.
    """
    try:
        from prometheus_fastapi_instrumentator import routing
        from starlette.routing import Match, Mount
    except ImportError:
        return

    if getattr(routing, "_tinther_included_router_patch", False):
        return

    def _get_route_name(
        scope: dict[str, Any],
        routes: list[Any],
        route_name: str | None = None,
    ) -> str | None:
        for route in routes:
            match, child_scope = route.matches(scope)
            if match == Match.FULL:
                child_scope = {**scope, **child_scope}
                path = getattr(route, "path", None)

                original_router = getattr(route, "original_router", None)
                if path is None and original_router is not None:
                    return _get_route_name(
                        child_scope,
                        list(getattr(original_router, "routes", [])),
                        route_name,
                    )
                if path is None:
                    return route_name

                route_name = path
                if isinstance(route, Mount) and route.routes:
                    child_route_name = _get_route_name(
                        child_scope, route.routes, route_name
                    )
                    if child_route_name is None:
                        route_name = None
                    else:
                        route_name += child_route_name
                return route_name
            if match == Match.PARTIAL and route_name is None:
                path = getattr(route, "path", None)
                if path is not None:
                    route_name = path
        return None

    routing._get_route_name = _get_route_name
    routing._tinther_included_router_patch = True


async def _run(args) -> None:
    # Imports deferred so the module is importable without vLLM installed.
    from vllm import envs
    from vllm.entrypoints.launcher import serve_http
    from vllm.entrypoints.openai import api_server

    listen_address, sock = api_server.setup_server(args)

    async with api_server.build_async_engine_client(args) as engine_client:
        # vLLM's OpenAI server helpers are internal and changed between the
        # versions used on H200 and B300 hosts. Reflect the installed version's
        # callable shapes so tinther can keep using the standard app startup.
        get_vllm_config = getattr(engine_client, "get_vllm_config", None)
        if get_vllm_config is not None:
            vllm_config = get_vllm_config()
            if inspect.isawaitable(vllm_config):
                vllm_config = await vllm_config
        else:
            vllm_config = engine_client.vllm_config
        register_tokenizer_info = getattr(
            api_server, "maybe_register_tokenizer_info_endpoint", None
        )
        if register_tokenizer_info is not None:
            register_tokenizer_info(args)

        supported_tasks = None
        get_supported_tasks = getattr(engine_client, "get_supported_tasks", None)
        if get_supported_tasks is not None:
            supported_tasks = get_supported_tasks()
            if inspect.isawaitable(supported_tasks):
                supported_tasks = await supported_tasks

        build_app_params = inspect.signature(api_server.build_app).parameters
        build_app_kwargs: dict[str, Any] = {}
        if "supported_tasks" in build_app_params:
            build_app_kwargs["supported_tasks"] = supported_tasks
        if "model_config" in build_app_params:
            build_app_kwargs["model_config"] = getattr(vllm_config, "model_config", None)
        _patch_prometheus_fastapi_instrumentator()
        app = api_server.build_app(args, **build_app_kwargs)

        init_params = inspect.signature(api_server.init_app_state).parameters
        if "vllm_config" in init_params:
            await api_server.init_app_state(
                engine_client, vllm_config, app.state, args
            )
        else:
            init_kwargs: dict[str, Any] = {}
            if "supported_tasks" in init_params:
                init_kwargs["supported_tasks"] = supported_tasks
            await api_server.init_app_state(engine_client, app.state, args, **init_kwargs)
        app.include_router(_make_trainer_router(engine_client))

        logger.info(
            "tinther student vLLM server listening on %s (worker_ext=%s)",
            listen_address,
            getattr(args, "worker_extension_cls", "<unset>"),
        )
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
            h11_max_header_count=args.h11_max_header_count,
        )

    try:
        await shutdown_task
    finally:
        sock.close()


def _build_arg_parser():
    from vllm.entrypoints.openai.cli_args import make_arg_parser

    # vLLM moved this parser helper out of vllm.utils in newer releases.
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError:
        from vllm.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(description="tinther student vLLM server")
    return make_arg_parser(parser)


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    requested = getattr(args, "worker_extension_cls", None)
    if not requested:
        args.worker_extension_cls = _WORKER_EXT_CLS
    elif _WORKER_EXT_CLS not in requested:
        logger.warning(
            "--worker-extension-cls=%r does not include %s; "
            "trainer weight push will fail unless your class also implements "
            "init_weight_update_pg / recv_weight / close_weight_update_pg.",
            requested,
            _WORKER_EXT_CLS,
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
