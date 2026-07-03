"""vLLM worker extension that receives weight updates from a tinther trainer.

Loaded by vLLM via ``--worker-extension-cls
tinker_cookbook.tinther.vllm_student_worker.TintherWorkerExt``. Each TP/PP
worker owns one instance and joins a side-channel communicator with the
trainer's rank-0 process. The trainer takes rank 0; vLLM workers occupy
ranks ``1..world_size-1`` (so ``world_size = 1 + tensor_parallel * pipeline``).

The communicator follows the pattern used by OpenRLHF / vLLM's RLHF examples:
``vllm.distributed.utils.StatelessProcessGroup`` for the rendezvous TCPStore +
``vllm.distributed.device_communicators.pynccl.PyNcclCommunicator`` for the
NCCL ops. This avoids touching torch.distributed's global process group state,
which both sides already use for their own collectives.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
from typing import Any

import torch

logger = logging.getLogger("tinther.vllm_student_worker")


_TORCH_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


def _resolve_dtype(name: str) -> torch.dtype:
    key = name.lower().replace("torch.", "")
    if key in _TORCH_DTYPES:
        return _TORCH_DTYPES[key]
    if hasattr(torch, key) and isinstance(getattr(torch, key), torch.dtype):
        return getattr(torch, key)
    raise ValueError(f"Unsupported tensor dtype for weight update: {name!r}")


def _stateless_pynccl(host: str, port: int, rank: int, world_size: int, device: torch.device):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    timeout_s = int(os.environ.get("TINTHER_STUDENT_NCCL_TIMEOUT_S", "1800"))
    pg = StatelessProcessGroup.create(
        host=host,
        port=port,
        rank=rank,
        world_size=world_size,
        store_timeout=timeout_s,
    )
    return PyNcclCommunicator(pg, device=device)


class TintherWorkerExt:
    """Mixed into vLLM's worker class via ``--worker-extension-cls``.

    The host worker class supplies ``self.rank`` (the worker's rank inside
    vLLM's own world) and ``self.model_runner.model`` (an ``nn.Module``).

    vLLM injects the methods on this class into the host worker class but
    does NOT call ``__init__`` (the host worker has its own constructor).
    Per-instance state must therefore be initialized lazily inside the RPC
    methods, never in a constructor.
    """

    # ---- helpers ------------------------------------------------------

    def _vllm_local_rank(self) -> int:
        rank = getattr(self, "rank", None)
        if rank is None:
            raise RuntimeError(
                "TintherWorkerExt requires the host worker to expose `self.rank`"
            )
        return int(rank)

    def _underlying_model(self) -> torch.nn.Module:
        runner = getattr(self, "model_runner", None)
        if runner is None or getattr(runner, "model", None) is None:
            raise RuntimeError(
                "TintherWorkerExt requires `self.model_runner.model` to be initialized"
            )
        return runner.model  # type: ignore[no-any-return]

    # ---- public RPCs --------------------------------------------------

    def init_weight_update_pg(
        self,
        master_addr: str | None = None,
        master_port: int | None = None,
        world_size: int | None = None,
    ) -> dict[str, Any]:
        """Create the side-channel PyNcclCommunicator shared with the trainer.

        Arguments default to ``TINTHER_STUDENT_NCCL_*`` env vars so the same
        configuration can come from either the launcher or an explicit RPC
        payload.
        """
        if getattr(self, "_tinther_comm", None) is not None:
            return {
                "ok": True,
                "already_initialized": True,
                "rank": getattr(self, "_tinther_rank", None),
                "world_size": getattr(self, "_tinther_world_size", None),
            }

        addr = master_addr or os.environ.get("TINTHER_STUDENT_NCCL_MASTER_ADDR")
        port_env = os.environ.get("TINTHER_STUDENT_NCCL_MASTER_PORT")
        port = int(master_port if master_port is not None else (port_env or 0))
        ws_env = os.environ.get("TINTHER_STUDENT_NCCL_WORLD_SIZE")
        ws = int(world_size if world_size is not None else (ws_env or 0))
        if not addr or port <= 0 or ws < 2:
            raise RuntimeError(
                "init_weight_update_pg requires TINTHER_STUDENT_NCCL_MASTER_ADDR / "
                "TINTHER_STUDENT_NCCL_MASTER_PORT / TINTHER_STUDENT_NCCL_WORLD_SIZE "
                f"(got addr={addr!r}, port={port}, world_size={ws})"
            )

        my_rank = 1 + self._vllm_local_rank()
        if my_rank >= ws:
            raise RuntimeError(
                f"vLLM worker rank {my_rank} exceeds NCCL world_size {ws}; "
                "set TINTHER_STUDENT_NCCL_WORLD_SIZE = 1 + (TP * PP)"
            )

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self._tinther_comm = _stateless_pynccl(addr, port, my_rank, ws, device)
        self._tinther_world_size = ws
        self._tinther_rank = my_rank
        logger.info(
            "tinther weight-update communicator ready: rank=%s/%s addr=%s:%s",
            my_rank,
            ws,
            addr,
            port,
        )
        return {"ok": True, "rank": my_rank, "world_size": ws}

    def recv_weight(self, name: str, dtype: str, shape: list[int]) -> dict[str, Any]:
        """Receive one parameter broadcast by the trainer and load it."""
        comm = getattr(self, "_tinther_comm", None)
        if comm is None:
            raise RuntimeError("recv_weight called before init_weight_update_pg")

        torch_dtype = _resolve_dtype(dtype)
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        buf = torch.empty(tuple(shape), dtype=torch_dtype, device=device)
        comm.broadcast(buf, src=0, stream=torch.cuda.current_stream())
        torch.cuda.current_stream().synchronize()

        model = self._underlying_model()
        loader = getattr(model, "load_weights", None)
        if loader is None:
            raise RuntimeError(
                f"Underlying model {type(model).__name__} has no load_weights()"
            )
        loader([(name, buf)])
        return {"ok": True}

    def close_weight_update_pg(self) -> dict[str, Any]:
        if getattr(self, "_tinther_comm", None) is None:
            return {"ok": True, "already_closed": True}
        self._tinther_comm = None
        self._tinther_world_size = None
        self._tinther_rank = None
        return {"ok": True}
