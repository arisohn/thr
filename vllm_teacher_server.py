"""Binary teacher-logprob vLLM server for tinther.

This is a small vLLM/FastAPI server for hot teacher-logprob paths:

    prompt token ids -> prompt top-k token ids/logprobs, preserving each
    actual prompt token logprob

It deliberately bypasses vLLM's OpenAI JSON response models. The endpoint
returns dense numpy arrays in a msgpack-header + raw-buffer format, avoiding the
large ``prompt_logprobs`` JSON payload created by ``/v1/completions``.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import time
import uuid
from collections.abc import Sequence
from typing import Any

import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs


logger = logging.getLogger("tinther.vllm_teacher_server")

app = FastAPI(title="tinther teacher vLLM")

ENGINE: AsyncLLMEngine | None = None
SERVED_MODEL_NAME = "Qwen/Qwen3-32B"


def _supported_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(cls).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def _pack_arrays(header: dict[str, Any], arrays: dict[str, np.ndarray]) -> bytes:
    array_descs: list[dict[str, Any]] = []
    buffers: list[bytes] = []
    for name, arr in arrays.items():
        contiguous = np.ascontiguousarray(arr)
        array_descs.append(
            {
                "name": name,
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
                "nbytes": int(contiguous.nbytes),
            }
        )
        buffers.append(contiguous.tobytes())
    header = dict(header)
    header["arrays"] = array_descs
    header_bytes = msgpack.packb(header, use_bin_type=True)
    return b"".join([len(header_bytes).to_bytes(4, "little"), header_bytes, *buffers])


def _unpack_arrays(raw: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if len(raw) < 4:
        raise ValueError("binary request is too short")
    header_len = int.from_bytes(raw[:4], "little")
    header_start = 4
    header_end = header_start + header_len
    if len(raw) < header_end:
        raise ValueError("binary request header is truncated")
    header = msgpack.unpackb(raw[header_start:header_end], raw=False)
    if not isinstance(header, dict):
        raise ValueError("binary request header must be a map")

    arrays: dict[str, np.ndarray] = {}
    offset = header_end
    for desc in header.get("arrays") or []:
        name = str(desc["name"])
        dtype = np.dtype(desc["dtype"])
        shape = tuple(int(x) for x in desc["shape"])
        nbytes = int(desc["nbytes"])
        next_offset = offset + nbytes
        if next_offset > len(raw):
            raise ValueError(f"binary request array {name!r} is truncated")
        arrays[name] = np.frombuffer(raw[offset:next_offset], dtype=dtype).reshape(shape)
        offset = next_offset
    return header, arrays


def _logprob_value(obj: object) -> float:
    if hasattr(obj, "logprob"):
        return float(getattr(obj, "logprob"))
    if isinstance(obj, dict):
        return float(obj.get("logprob", 0.0))
    return float(str(obj))


def _topk_arrays(
    prompt_logprobs: Sequence[dict[int, object] | None] | None,
    prompt_tokens: Sequence[int],
    prompt_len: int,
    topk: int,
) -> dict[str, np.ndarray]:
    if topk <= 0:
        return {
            "prompt_topk_token_ids": np.empty((prompt_len, 0), dtype=np.int32),
            "prompt_topk_logprobs": np.empty((prompt_len, 0), dtype=np.float32),
            "prompt_topk_counts": np.zeros((prompt_len,), dtype=np.int16),
        }

    # Allocate one extra slot: vLLM's prompt_logprobs payload may contain the
    # actual prompt token even when it is not among the top-k candidates.
    # compute_logprobs needs that exact token, while soft-target consumers can
    # still slice back to the requested top-k entries.
    width = topk + 1
    token_ids = np.full((prompt_len, width), -1, dtype=np.int32)
    logprobs = np.full((prompt_len, width), -np.inf, dtype=np.float32)
    counts = np.zeros((prompt_len,), dtype=np.int16)
    if prompt_logprobs is None:
        return {
            "prompt_topk_token_ids": token_ids,
            "prompt_topk_logprobs": logprobs,
            "prompt_topk_counts": counts,
        }

    for i, pos in enumerate(prompt_logprobs[:prompt_len]):
        if pos is None:
            continue
        items = [
            (int(tok_id), _logprob_value(lp_obj))
            for tok_id, lp_obj in pos.items()
        ]
        items.sort(key=lambda item: item[1], reverse=True)
        selected = items[:topk]
        if i > 0 and i < len(prompt_tokens):
            actual_tok = int(prompt_tokens[i])
            actual = next((item for item in items if item[0] == actual_tok), None)
            if actual is not None and all(tok_id != actual_tok for tok_id, _ in selected):
                # Keep top-k ordering intact, then append the actual token as a
                # sidecar entry for teacher-forced scoring.
                selected.append(actual)
        counts[i] = len(selected)
        for j, (tok_id, lp) in enumerate(selected):
            token_ids[i, j] = tok_id
            logprobs[i, j] = lp

    return {
        "prompt_topk_token_ids": token_ids,
        "prompt_topk_logprobs": logprobs,
        "prompt_topk_counts": counts,
    }


def _completion_arrays(outputs: Sequence[object]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    output_count = len(outputs)
    max_len = max((len(getattr(output, "token_ids", []) or []) for output in outputs), default=0)
    token_ids = np.full((output_count, max_len), -1, dtype=np.int32)
    logprobs = np.zeros((output_count, max_len), dtype=np.float32)
    counts = np.zeros((output_count,), dtype=np.int16)
    metadata: list[dict[str, Any]] = []

    for i, output in enumerate(outputs):
        out_tokens = [int(tok) for tok in (getattr(output, "token_ids", []) or [])]
        counts[i] = len(out_tokens)
        if out_tokens:
            token_ids[i, : len(out_tokens)] = np.asarray(out_tokens, dtype=np.int32)

        per_token_logprobs = getattr(output, "logprobs", None) or []
        for j, tok in enumerate(out_tokens):
            if j >= len(per_token_logprobs) or per_token_logprobs[j] is None:
                continue
            pos = per_token_logprobs[j]
            lp_obj = pos.get(tok) if isinstance(pos, dict) else None
            if lp_obj is None and isinstance(pos, dict) and pos:
                lp_obj = next(iter(pos.values()))
            if lp_obj is not None:
                logprobs[i, j] = _logprob_value(lp_obj)

        metadata.append(
            {
                "finish_reason": getattr(output, "finish_reason", None),
                "stop_reason": getattr(output, "stop_reason", None),
                "cumulative_logprob": getattr(output, "cumulative_logprob", None),
            }
        )

    return (
        {
            "completion_token_ids": token_ids,
            "completion_logprobs": logprobs,
            "completion_counts": counts,
        },
        metadata,
    )


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "created": 0,
                "owned_by": "tinther",
            }
        ],
    }


@app.post("/tinther/v1/completions_bin")
async def completions_bin(request: Request) -> Response:
    if ENGINE is None:
        raise HTTPException(503, "vLLM engine is not initialized")

    t0 = time.perf_counter()
    try:
        header, arrays = _unpack_arrays(await request.body())
        prompt_arr = arrays["prompt_token_ids"]
    except Exception as exc:
        raise HTTPException(400, f"invalid binary request: {exc}") from exc

    prompt_tokens = [int(tok) for tok in np.asarray(prompt_arr, dtype=np.int32).reshape(-1)]
    prompt_logprobs = int(header.get("prompt_logprobs") or 0)
    request_id = str(header.get("request_id") or uuid.uuid4())

    sampling_params = SamplingParams(
        n=int(header.get("n", 1)),
        max_tokens=int(header.get("max_tokens", 1)),
        temperature=float(header.get("temperature", 1.0)),
        logprobs=int(header.get("logprobs", 1)) if header.get("logprobs") is not None else None,
        prompt_logprobs=prompt_logprobs if prompt_logprobs > 0 else None,
        stop=header.get("stop"),
        stop_token_ids=header.get("stop_token_ids"),
        detokenize=False,
    )

    final_output = None
    async for output in ENGINE.generate(
        {"prompt_token_ids": prompt_tokens},
        sampling_params,
        request_id=request_id,
    ):
        final_output = output

    if final_output is None:
        raise HTTPException(500, "vLLM returned no output")

    prompt_arrays = _topk_arrays(
        final_output.prompt_logprobs,
        prompt_tokens=prompt_tokens,
        prompt_len=len(prompt_tokens),
        topk=prompt_logprobs,
    )
    completion_arrays, completion_metadata = _completion_arrays(final_output.outputs)
    arrays_out = {**prompt_arrays, **completion_arrays}

    t1 = time.perf_counter()
    response_header = {
        "version": 1,
        "request_id": request_id,
        "model": SERVED_MODEL_NAME,
        "prompt_len": len(prompt_tokens),
        "prompt_logprobs": prompt_logprobs,
        "completion_metadata": completion_metadata,
        "timing": {"engine_s": round(t1 - t0, 6)},
    }
    payload = _pack_arrays(response_header, arrays_out)
    logger.info(
        "completions_bin prompt_len=%d topk=%d bytes=%.1fMiB engine=%.3fs total=%.3fs",
        len(prompt_tokens),
        prompt_logprobs,
        len(payload) / 2**20,
        t1 - t0,
        time.perf_counter() - t0,
    )
    return Response(content=payload, media_type="application/octet-stream")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="Qwen/Qwen3-32B")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-logprobs", type=int, default=20)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--generation-config", default="auto")
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--uvicorn-log-level", default="info")
    parser.add_argument("--disable-uvicorn-access-log", action="store_true")
    return parser.parse_args()


def main() -> None:
    global ENGINE, SERVED_MODEL_NAME

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    SERVED_MODEL_NAME = args.served_model_name

    engine_kwargs = {
        "model": args.model,
        "served_model_name": args.served_model_name,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_logprobs": args.max_logprobs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "generation_config": args.generation_config,
        "async_scheduling": args.async_scheduling,
    }
    engine_kwargs = {k: v for k, v in engine_kwargs.items() if v is not None}
    engine_args = AsyncEngineArgs(**_supported_kwargs(AsyncEngineArgs, engine_kwargs))
    ENGINE = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        access_log=not args.disable_uvicorn_access_log,
    )


if __name__ == "__main__":
    main()
