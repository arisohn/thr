"""tinther — local drop-in for the `tinker` SDK.

Covers the API surface used (directly and transitively) by
``tinker_cookbook/distillation/train_on_policy.py`` and
``tinker_cookbook/distillation/train_off_policy.py``.

Training compute is delegated to a remote ``train.py`` server (see
``_HTTPTrainerBackend``). All sampling — student and teacher — goes through
external vLLM OpenAI-compatible HTTP servers, accessed via
``_HTTPSamplerBackend``. The student vLLM server (started separately by the
user via ``vllm_student_server.py``) receives its weights over a
side-channel NCCL communicator from the trainer; tinther itself never spawns
or holds an in-process sampler.

Required environment:
  * ``TINTHER_TRAIN_URL``       — train.py server.
  * ``TINTHER_STUDENT_URL``     — student vLLM server (single endpoint).
  * ``TINTHER_TEACHER_URL`` or ``TINTHER_TEACHER_URLS`` — teacher vLLM(s).
  * ``TINTHER_STUDENT_NCCL_*``  — used by train.py / vllm_student_worker.

Register as ``tinker`` before importing cookbook modules::

    import tinther
    tinther.install_as_tinker()
    from tinker_cookbook.distillation.train_on_policy import Config, main

BASED ON: d88b3c35905d0e4d3dbdc128a22008b18d1bec07
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import time
import types as _types_module
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import numpy as np
import torch

logger = logging.getLogger("tinther")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Exceptions and simple type aliases
# ---------------------------------------------------------------------------


class TinkerError(Exception):
    """Raised for tinther/tinker-level failures."""


StopReason = str
LossFnType = Literal["cross_entropy", "importance_sampling", "ppo"]


# ---------------------------------------------------------------------------
# Data types (mirror tinker's public surface)
# ---------------------------------------------------------------------------


@dataclass
class EncodedTextChunk:
    """A chunk of already-encoded text tokens."""

    tokens: list[int]

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass
class ImageChunk:
    """JPEG-encoded image chunk (stub; distillation scripts don't use images)."""

    data: bytes = b""
    width: int = 0
    height: int = 0

    @property
    def length(self) -> int:
        return 0


@dataclass
class ImageAssetPointerChunk:
    """Pointer to an uploaded image asset (stub)."""

    url: str = ""

    @property
    def length(self) -> int:
        return 0


ModelInputChunk = EncodedTextChunk  # runtime alias; type is broader at type-check time


@dataclass
class ModelInput:
    """Tokenized prompt built from one or more chunks."""

    chunks: list[ModelInputChunk] = field(default_factory=list)

    @classmethod
    def from_ints(cls, ints: list[int]) -> "ModelInput":
        return cls(chunks=[EncodedTextChunk(tokens=list(ints))])

    @classmethod
    def empty(cls) -> "ModelInput":
        return cls(chunks=[])

    def append_int(self, token_id: int) -> "ModelInput":
        new_chunks = [EncodedTextChunk(tokens=list(c.tokens)) for c in self.chunks]
        if new_chunks and isinstance(new_chunks[-1], EncodedTextChunk):
            new_chunks[-1].tokens.append(int(token_id))
        else:
            new_chunks.append(EncodedTextChunk(tokens=[int(token_id)]))
        return ModelInput(chunks=new_chunks)

    def to_ints(self) -> list[int]:
        out: list[int] = []
        for c in self.chunks:
            if isinstance(c, EncodedTextChunk):
                out.extend(c.tokens)
            else:
                raise TinkerError(f"Cannot flatten non-text chunk: {type(c).__name__}")
        return out

    @property
    def length(self) -> int:
        return sum(c.length for c in self.chunks)


@dataclass
class TensorData:
    """Numpy-backed tensor with metadata, round-trippable to torch."""

    data: np.ndarray
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        arr = np.asarray(self.data, dtype=np.dtype(self.dtype)).reshape(tuple(self.shape))
        self.data = arr
        self.dtype = str(arr.dtype)
        self.shape = tuple(arr.shape)

    @classmethod
    def from_torch(cls, t: torch.Tensor) -> "TensorData":
        t = t.detach().to("cpu").contiguous()
        # numpy does not support bf16; promote to float32 for round-tripping.
        if t.dtype == torch.bfloat16:
            t = t.to(torch.float32)
        arr = t.numpy()
        return cls(data=arr, dtype=str(arr.dtype), shape=tuple(arr.shape))

    def to_torch(self) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.data))

    def tolist(self) -> list[Any]:
        return np.asarray(self.data).tolist()


@dataclass
class Datum:
    """A single training example."""

    model_input: ModelInput
    loss_fn_inputs: dict[str, TensorData]


@dataclass
class SamplingParams:
    """Sampling knobs."""

    max_tokens: int
    temperature: float = 1.0
    stop: Any | None = None  # list[str] | list[int] | None


@dataclass
class AdamParams:
    """Adam optimizer knobs."""

    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8


@dataclass
class _Sequence:
    tokens: list[int]
    logprobs: list[float] | None
    stop_reason: StopReason


@dataclass
class SampleResponse:
    sequences: list[_Sequence]
    topk_prompt_logprobs: list[list[tuple[int, float]] | None] | None = None


@dataclass
class ForwardBackwardOutput:
    loss_fn_outputs: list[dict[str, TensorData]]
    metrics: dict[str, float] | None = None


@dataclass
class OptimStepResponse:
    metrics: dict[str, float] | None = None


@dataclass
class _SavePath:
    """Return value for save_state / save_weights_for_sampler futures."""

    path: str


class APIFuture(Generic[T]):
    """Handle to a TrainingClient operation running as a background task.

    ``forward_backward_async`` / ``optim_step_async`` / ``save_*_async`` schedule
    the actual work via ``asyncio.create_task`` and hand back this future
    immediately, so callers can pipeline submissions (e.g. submit fwd-bwd and
    optim-step back-to-back) before awaiting any results. Exceptions raised by
    the background task surface at ``result_async()``.
    """

    def __init__(self, task: "asyncio.Task[T]"):
        self._task = task

    async def result_async(self) -> T:
        return await self._task

    @property
    def result(self) -> T:
        if not self._task.done():
            raise TinkerError("APIFuture.result accessed before completion; await result_async().")
        return self._task.result()


def _cache_root() -> Path:
    root = Path(os.environ.get("TINTHER_CACHE_DIR", "/tmp/tinther"))
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Checkpoint store (tinther:// path <-> on-disk directory)
# ---------------------------------------------------------------------------


class _CheckpointStore:
    """Maps tinther://{state|sampler}/<uuid> strings to local directories."""

    @staticmethod
    def new_path(kind: Literal["state", "sampler"], name: str | None = None) -> tuple[str, Path]:
        uid = name or uuid.uuid4().hex
        d = _cache_root() / kind / uid
        d.mkdir(parents=True, exist_ok=True)
        return f"tinther://{kind}/{uid}", d

    @staticmethod
    def resolve(path: str) -> Path:
        if not path.startswith("tinther://"):
            # Treat as a direct filesystem path (useful for pre-existing HF snapshots).
            return Path(path)
        rest = path[len("tinther://") :]
        parts = rest.split("/", 1)
        if len(parts) != 2:
            raise TinkerError(f"Malformed tinther path: {path}")
        kind, uid = parts
        d = _cache_root() / kind / uid
        if not d.exists():
            raise TinkerError(f"Checkpoint path does not exist: {path}")
        return d

    @staticmethod
    def write_meta(path: str, meta: dict[str, Any]) -> None:
        d = _CheckpointStore.resolve(path)
        with (d / "meta.json").open("w") as f:
            json.dump(meta, f)

    @staticmethod
    def read_meta(path: str) -> dict[str, Any]:
        d = _CheckpointStore.resolve(path)
        p = d / "meta.json"
        if not p.exists():
            return {}
        try:
            with p.open("r") as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def delete(path: str) -> None:
        try:
            d = _CheckpointStore.resolve(path)
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass



# ---------------------------------------------------------------------------
# Shared HTTP plumbing: short-lived httpx clients
# ---------------------------------------------------------------------------


class _HTTPBackend:
    """Shared HTTP POST plumbing for sampler/trainer backends.

    Subclasses provide their own URL routing (round-robin or single endpoint)
    and concurrency controls. Each request creates and closes its own
    ``httpx.AsyncClient`` for a simple resource lifecycle.
    """

    def __init__(self, *, timeout_s: float, default_headers: dict[str, str], log_label: str) -> None:
        self._timeout_s = timeout_s
        self._default_headers = dict(default_headers)
        self._log_label = log_label

    async def _post(self, base_url: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
        import httpx

        url = f"{base_url}{path}"
        try:
            async with httpx.AsyncClient(headers=self._default_headers, timeout=httpx.Timeout(self._timeout_s)) as client:
                resp = await client.post(url, json=body)
        except httpx.HTTPError as e:
            raise TinkerError(f"{self._log_label} request to {url} failed: {e}") from e

        if resp.status_code != 200:
            raise TinkerError(f"{self._log_label} at {url} returned HTTP {resp.status_code}: {resp.text[:500]}")

        return resp.json()

    async def _post_bytes(self, base_url: str, path: str, payload: bytes, *, content_type: str = "application/octet-stream") -> bytes:
        import httpx

        url = f"{base_url}{path}"
        headers = dict(self._default_headers)
        headers["Content-Type"] = content_type
        try:
            async with httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(self._timeout_s)) as client:
                resp = await client.post(url, content=payload)
        except httpx.HTTPError as e:
            raise TinkerError(f"{self._log_label} request to {url} failed: {e}") from e

        if resp.status_code != 200:
            raise TinkerError(f"{self._log_label} at {url} returned HTTP {resp.status_code}: {resp.text[:500]}")

        return resp.content

# ---------------------------------------------------------------------------
# Sampler backend: external vLLM server via HTTP (OpenAI-compatible)
# ---------------------------------------------------------------------------


class _HTTPSamplerBackend(_HTTPBackend):
    """Talks to an out-of-process vLLM server via its OpenAI-compatible API.

    vLLM's ``POST /v1/completions`` accepts the standard OpenAI request shape
    plus two extensions we rely on:

    * ``prompt_logprobs: int`` — return top-K logprobs for each prompt position,
      including the teacher distribution needed for off-policy distillation.
    * ``return_token_ids: bool`` — populate ``choice.token_ids`` and
      ``choice.prompt_token_ids`` (raw IDs, no re-tokenization roundtrip).

    The response payload serializes ``list[dict[int, Logprob] | None]``;
    Pydantic writes the dict keys as strings (``"12345"``), and each ``Logprob``
    is ``{"logprob": float, "rank": int | None, "decoded_token": str | None}``.
    """

    def __init__(self, base_url: str, model_name: str, api_key: str | None = None, timeout_s: float = 600.0, concurrency: int = 64, base_urls: list[str] | None = None) -> None:
        normalized_urls = [url.rstrip("/") for url in (base_urls or [base_url]) if url]
        if not normalized_urls:
            raise TinkerError("Teacher HTTP backend requires at least one base URL")
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        super().__init__(
            timeout_s=timeout_s,
            default_headers=headers,
            log_label="Teacher",
        )
        self._base_urls = normalized_urls
        self.base_url = self._base_urls[0]
        self.model_name = model_name
        self._sem = asyncio.Semaphore(concurrency)
        self._url_index = 0
        self._use_binary = os.environ.get("TINTHER_TEACHER_BINARY", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._binary_path = os.environ.get(
            "TINTHER_TEACHER_BINARY_PATH",
            "/tinther/v1/completions_bin",
        )

    # ---- construction --------------------------------------------------

    @classmethod
    def from_env(cls, model_ref: str) -> "_HTTPSamplerBackend":
        """Build a backend from ``TINTHER_TEACHER_*`` env vars.

        ``TINTHER_TEACHER_URLS`` (JSON dict) lets multi-teacher configs pick a
        URL per model ref using substring match on the key; this is the MOPD
        pattern in ``train_off_policy.py``. Falls back to
        ``TINTHER_TEACHER_URL``. When a list of URLs is selected, requests are
        round-robined across them.
        """
        base_url = os.environ.get("TINTHER_TEACHER_URL")
        selected_urls: list[str] | None = None
        urls_json = os.environ.get("TINTHER_TEACHER_URLS")

        def _normalize_urls(urls: list[Any]) -> list[str]:
            str_urls = [str(url) for url in urls if str(url)]
            if not str_urls:
                raise TinkerError("TINTHER_TEACHER_URLS list must not be empty")
            return str_urls

        if urls_json:
            try:
                urls_map = json.loads(urls_json)
            except Exception as e:
                raise TinkerError(
                    f"Invalid TINTHER_TEACHER_URLS (must be JSON dict or list): {e}"
                ) from e
            if isinstance(urls_map, list):
                selected_urls = _normalize_urls(urls_map)
            elif isinstance(urls_map, dict):
                for key, url in urls_map.items():
                    if key in str(model_ref):
                        if isinstance(url, list):
                            selected_urls = _normalize_urls(url)
                        else:
                            base_url = str(url)
                        break
            else:
                raise TinkerError("TINTHER_TEACHER_URLS must be a JSON dict or list")
        base_urls = None
        if selected_urls:
            base_urls = selected_urls
            base_url = selected_urls[0]
        if not base_url:
            raise TinkerError(
                "Teacher sampling requested but TINTHER_TEACHER_URL is not set. "
                "Start an external vLLM server (e.g. "
                "`python -m vllm.entrypoints.openai.api_server --model ... --port 8765`) "
                "and export TINTHER_TEACHER_URL=http://127.0.0.1:8765"
            )
        model_name = os.environ.get("TINTHER_TEACHER_MODEL_NAME") or model_ref
        api_key = os.environ.get("TINTHER_TEACHER_API_KEY")
        timeout_s = float(os.environ.get("TINTHER_TEACHER_TIMEOUT", "600"))
        concurrency = int(os.environ.get("TINTHER_TEACHER_CONCURRENCY", "64"))
        return cls(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            timeout_s=timeout_s,
            concurrency=concurrency,
            base_urls=base_urls,
        )

    @classmethod
    def from_student_env(cls, model_ref: str) -> "_HTTPSamplerBackend":
        """Build a backend from ``TINTHER_STUDENT_*`` env vars.

        The student vLLM server is single-endpoint (v1). The model name sent
        in OpenAI requests defaults to whatever the server reports (most
        deployments accept any string here, but ``TINTHER_STUDENT_MODEL_NAME``
        is available as an override).
        """
        base_url = os.environ.get("TINTHER_STUDENT_URL")
        if not base_url:
            raise TinkerError(
                "Student sampling requires TINTHER_STUDENT_URL to be set to the "
                "URL of the vLLM server you launched via vllm_student_server.py "
                "(e.g. http://127.0.0.1:8001)."
            )
        model_name = (
            os.environ.get("TINTHER_STUDENT_MODEL_NAME")
            or str(model_ref)
        )
        api_key = os.environ.get("TINTHER_STUDENT_API_KEY")
        timeout_s = float(os.environ.get("TINTHER_STUDENT_TIMEOUT", "600"))
        concurrency = int(os.environ.get("TINTHER_STUDENT_CONCURRENCY", "64"))
        return cls(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            timeout_s=timeout_s,
            concurrency=concurrency,
        )

    # ---- session management -------------------------------------------

    async def _pick_base_url(self) -> str:
        if len(self._base_urls) == 1:
            return self._base_urls[0]
        url = self._base_urls[self._url_index]
        self._url_index = (self._url_index + 1) % len(self._base_urls)
        return url

    # ---- request helpers ----------------------------------------------

    @staticmethod
    def _parse_prompt_logprobs(raw: list | None) -> list[list[tuple[int, float]] | None] | None:
        """Convert vLLM's response-side prompt_logprobs to tinker's format.

        Input: ``[None | {"tok_id_str": {"logprob": float, ...}}]``
        Output: ``[None | [(int, float), ...]]`` (sorted by logprob desc).
        """
        if raw is None:
            return None
        out: list[list[tuple[int, float]] | None] = []
        for pos in raw:
            if pos is None:
                out.append(None)
                continue
            items: list[tuple[int, float]] = []
            for tok_key, lp_obj in pos.items():
                # JSON serialization coerces dict[int, ...] keys to strings.
                try:
                    tok_id = int(tok_key)
                except (TypeError, ValueError):
                    continue
                if isinstance(lp_obj, dict):
                    lp_val = float(lp_obj.get("logprob", 0.0))
                else:
                    lp_val = float(lp_obj)
                items.append((tok_id, lp_val))
            items.sort(key=lambda x: x[1], reverse=True)
            out.append(items)
        return out

    async def _post_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        base_url = await self._pick_base_url()
        return await self._post(base_url, "/v1/completions", body)

    async def _post_completions_bin(self, body: dict[str, Any]) -> bytes:
        import msgpack

        tokens = np.asarray(body["prompt"], dtype=np.int32)
        header = {
            "version": 1,
            "model": str(body.get("model") or self.model_name),
            "max_tokens": int(body.get("max_tokens", 1)),
            "temperature": float(body.get("temperature", 1.0)),
            "n": int(body.get("n", 1)),
            "logprobs": body.get("logprobs"),
            "prompt_logprobs": int(body.get("prompt_logprobs") or 0),
            "stop": body.get("stop"),
            "stop_token_ids": body.get("stop_token_ids"),
            "arrays": [
                {
                    "name": "prompt_token_ids",
                    "dtype": str(tokens.dtype),
                    "shape": list(tokens.shape),
                    "nbytes": int(tokens.nbytes),
                }
            ],
        }
        header_bytes = msgpack.packb(header, use_bin_type=True)
        payload = b"".join(
            [
                len(header_bytes).to_bytes(4, "little"),
                header_bytes,
                np.ascontiguousarray(tokens).tobytes(),
            ]
        )
        base_url = await self._pick_base_url()
        return await self._post_bytes(base_url, self._binary_path, payload)

    @staticmethod
    def _unpack_arrays_bin(raw: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        import msgpack

        if len(raw) < 4:
            raise TinkerError("Binary teacher response is too short")
        header_len = int.from_bytes(raw[:4], "little")
        header_start = 4
        header_end = header_start + header_len
        header = msgpack.unpackb(raw[header_start:header_end], raw=False)
        arrays: dict[str, np.ndarray] = {}
        offset = header_end
        for desc in header.get("arrays") or []:
            name = str(desc["name"])
            dtype = np.dtype(desc["dtype"])
            shape = tuple(int(x) for x in desc["shape"])
            nbytes = int(desc["nbytes"])
            next_offset = offset + nbytes
            arrays[name] = np.frombuffer(raw[offset:next_offset], dtype=dtype).reshape(shape).copy()
            offset = next_offset
        return header, arrays

    @classmethod
    def _sample_response_from_bin(cls, raw: bytes) -> SampleResponse:
        header, arrays = cls._unpack_arrays_bin(raw)

        seqs: list[_Sequence] = []
        completion_ids = arrays.get("completion_token_ids")
        completion_lps = arrays.get("completion_logprobs")
        completion_counts = arrays.get("completion_counts")
        completion_metadata = header.get("completion_metadata") or []
        if completion_ids is not None and completion_counts is not None:
            for i, count_raw in enumerate(completion_counts.tolist()):
                count = int(count_raw)
                tokens = [int(t) for t in completion_ids[i, :count].tolist()]
                logprobs = None
                if completion_lps is not None:
                    logprobs = [float(x) for x in completion_lps[i, :count].tolist()]
                md = completion_metadata[i] if i < len(completion_metadata) else {}
                seqs.append(
                    _Sequence(
                        tokens=tokens,
                        logprobs=logprobs,
                        stop_reason=str(md.get("finish_reason") or "stop"),
                    )
                )

        topk_prompt: list[list[tuple[int, float]] | None] | None = None
        topk_ids = arrays.get("prompt_topk_token_ids")
        topk_lps = arrays.get("prompt_topk_logprobs")
        topk_counts = arrays.get("prompt_topk_counts")
        if topk_ids is not None and topk_lps is not None and topk_counts is not None:
            topk_prompt = []
            for i, count_raw in enumerate(topk_counts.tolist()):
                count = int(count_raw)
                if count <= 0:
                    topk_prompt.append(None)
                    continue
                topk_prompt.append(
                    [
                        (int(tok), float(lp))
                        for tok, lp in zip(
                            topk_ids[i, :count].tolist(),
                            topk_lps[i, :count].tolist(),
                        )
                        if int(tok) >= 0
                    ]
                )

        return SampleResponse(sequences=seqs, topk_prompt_logprobs=topk_prompt)

    # ---- public surface (matches the SamplingClient contract) --------

    async def sample(self, prompt: ModelInput, num_samples: int, params: SamplingParams, include_prompt_logprobs: bool, topk_prompt_logprobs: int | None) -> SampleResponse:
        """Generate continuations from a tokenized prompt via vLLM.

        Input: prompt tokens, number of samples, sampling params, and optional
        prompt-logprob settings.
        Output: generated token sequences with per-generated-token logprobs,
        plus optional top-k logprobs for prompt positions.
        """
        tokens = prompt.to_ints()
        body: dict[str, Any] = {
            "model": self.model_name,
            "prompt": tokens,
            "max_tokens": int(params.max_tokens),
            "temperature": float(params.temperature),
            "n": int(num_samples),
            "logprobs": 1,
            "return_token_ids": True,
        }
        if include_prompt_logprobs:
            body["prompt_logprobs"] = int(topk_prompt_logprobs or 1)
        if params.stop is not None:
            stop_seq = list(params.stop)
            if stop_seq and isinstance(stop_seq[0], int):
                body["stop_token_ids"] = [int(x) for x in stop_seq]
            elif stop_seq:
                body["stop"] = [str(x) for x in stop_seq]

        if self._use_binary and include_prompt_logprobs:
            async with self._sem:
                raw_payload = await self._post_completions_bin(body)
            return self._sample_response_from_bin(raw_payload)

        async with self._sem:
            payload = await self._post_completions(body)

        choices = payload.get("choices") or []
        seqs: list[_Sequence] = []
        for choice in choices:
            out_tokens = choice.get("token_ids") or []
            lp_info = choice.get("logprobs") or {}
            out_lps = lp_info.get("token_logprobs")
            if out_lps is not None:
                out_lps = [float(x) if x is not None else 0.0 for x in out_lps]
            stop_reason = choice.get("finish_reason") or "stop"
            seqs.append(
                _Sequence(
                    tokens=[int(t) for t in out_tokens],
                    logprobs=out_lps,
                    stop_reason=str(stop_reason),
                )
            )

        topk_prompt = None
        if include_prompt_logprobs and choices:
            topk_prompt = self._parse_prompt_logprobs(choices[0].get("prompt_logprobs"))

        return SampleResponse(sequences=seqs, topk_prompt_logprobs=topk_prompt)

    async def compute_logprobs(self, sequence: ModelInput) -> list[float | None]:
        """Score a fixed token sequence via teacher forcing.

        Input: a complete token sequence, not just a prompt.
        Output: per-position logprobs for existing tokens, aligned as
        ``logprobs[i] = log P(sequence[i] | sequence[:i])``. The first token
        has no previous context, so ``logprobs[0]`` is ``None`` and the output
        length matches the input length.

        Implemented via ``prompt_logprobs=1`` (top-1) on an empty generation;
        we pick the actual prompt-token entry that vLLM includes for each
        scored prompt position.
        """
        tokens = sequence.to_ints()
        if not tokens:
            return []
        body = {
            "model": self.model_name,
            "prompt": tokens,
            "max_tokens": 1,
            "temperature": 0.0,
            "n": 1,
            "logprobs": 1,
            "prompt_logprobs": 1,
            "return_token_ids": True,
        }
        if self._use_binary:
            async with self._sem:
                raw_payload = await self._post_completions_bin(body)
            response = self._sample_response_from_bin(raw_payload)
            raw = response.topk_prompt_logprobs or []
        else:
            async with self._sem:
                payload = await self._post_completions(body)
            choices = payload.get("choices") or []
            if not choices:
                raise TinkerError("compute_logprobs received no completion choices")
            raw = self._parse_prompt_logprobs(choices[0].get("prompt_logprobs")) or []

        # Match the public Tinker contract: one output slot per input token,
        # with no score for the first token because it has no context.
        out: list[float | None] = [None] * len(tokens)
        for i, tok in enumerate(tokens[1:], start=1):
            pos = raw[i] if i < len(raw) else None
            if pos is None:
                raise TinkerError(
                    "compute_logprobs did not receive prompt logprobs for "
                    f"position {i}; cannot compute teacher-forced logprobs."
                )
            match = next((lp for tid, lp in pos if tid == tok), None)
            if match is None:
                # A top-1 entry for another token is not a teacher-forced
                # logprob for this sequence, so fail instead of substituting it.
                available = ", ".join(str(tid) for tid, _ in pos[:5])
                raise TinkerError(
                    "compute_logprobs did not receive the logprob for the "
                    f"actual prompt token at position {i} (token id {tok}). "
                    f"Available token ids: {available or 'none'}."
                )
            out[i] = match
        return out


# ---------------------------------------------------------------------------
# Trainer backend: external train.py server via HTTP
# ---------------------------------------------------------------------------


class _HTTPTrainerBackend(_HTTPBackend):
    """Talks to an out-of-process ``train.py`` server via FastAPI.

    Exposes ``forward_backward``, ``optim_step``, ``save_state``,
    ``save_weights_for_sampler``, ``set_user_metadata``, and
    ``get_tokenizer``. The model + optimizer + LoRA adapter live on the
    torchrun-launched server; this class only encodes/decodes JSON payloads.

    The tokenizer is loaded client-side with HuggingFace ``AutoTokenizer``
    using the same ``model_name`` (or saved-state directory) as the server.
    """

    def __init__(self, model_name: str, lora_rank: int | None, from_state: str | None = None, load_optimizer: bool = False) -> None:
        # ``load_optimizer`` is a server-side concern (the server must be
        # launched with ``--from-state``); the flag is kept for API parity.
        del load_optimizer
        from transformers import AutoTokenizer

        base_url = os.environ.get("TINTHER_TRAIN_URL")
        if not base_url:
            raise TinkerError(
                "_HTTPTrainerBackend requires TINTHER_TRAIN_URL to be set "
                "to the URL of a torchrun-launched train.py server."
            )
        super().__init__(
            timeout_s=float(os.environ.get("TINTHER_TRAIN_TIMEOUT", "1800")),
            default_headers={"Content-Type": "application/json"},
            log_label="train.py",
        )
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.lora_rank = int(lora_rank or 0)
        self._lock = asyncio.Lock()

        self._last_backward_outputs: list[dict[str, TensorData]] | None = None
        self._last_backward_metrics: dict[str, float] | None = None

        tok_source = (
            str(_CheckpointStore.resolve(from_state)) if from_state else model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self._user_metadata: dict[str, str] = {}
        # Validate compatibility synchronously so misconfigurations fail fast.
        self._validate_health()

    def _validate_health(self) -> None:
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(
                f"{self.base_url}/health", timeout=10
            ) as resp:
                info = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
            raise TinkerError(
                f"could not reach train.py at {self.base_url}/health: {e}"
            ) from e
        srv_lora = int(info.get("lora_rank", 0))
        if srv_lora != self.lora_rank:
            raise TinkerError(
                f"train.py server lora_rank={srv_lora} does not match "
                f"requested lora_rank={self.lora_rank}; relaunch the server with "
                f"--lora-rank {self.lora_rank}"
            )
        logger.info(
            "Connected to train.py at %s (model=%s, world_size=%s)",
            self.base_url,
            info.get("model_name"),
            info.get("world_size"),
        )

    # ---- payload helpers ---------------------------------------------

    @staticmethod
    def _tensor_data_to_payload(t: TensorData) -> dict[str, Any]:
        arr = np.asarray(t.data)
        return {"dtype": str(arr.dtype), "shape": list(arr.shape), "data": arr.tolist()}

    @staticmethod
    def _payload_to_tensor_data(payload: dict[str, Any]) -> TensorData:
        arr = np.asarray(payload["data"]).astype(np.dtype(payload["dtype"]))
        arr = arr.reshape(tuple(payload["shape"]))
        return TensorData(data=arr, dtype=str(arr.dtype), shape=tuple(arr.shape))

    # ---- public surface ----------------------------------------------

    def set_user_metadata(self, md: dict[str, str] | None) -> None:
        self._user_metadata = dict(md or {})

    def get_tokenizer(self):
        return self.tokenizer

    async def forward_backward(self, data_D: list[Datum], loss_fn: str, loss_fn_config: dict[str, Any] | None) -> ForwardBackwardOutput:
        import msgpack

        async with self._lock:
            # Build binary payload: msgpack header + concatenated numpy buffers
            datum_descs = []
            buffers: list[bytes] = []
            for d in data_D:
                tokens = d.model_input.to_ints()
                tokens_arr = np.array(tokens, dtype=np.int32)
                buffers.append(tokens_arr.tobytes())
                tensor_descs = []
                for k, v in d.loss_fn_inputs.items():
                    arr = np.asarray(v.data)
                    buffers.append(arr.tobytes())
                    tensor_descs.append({
                        "name": k,
                        "dtype": str(arr.dtype),
                        "shape": list(arr.shape),
                    })
                datum_descs.append({
                    "tokens_len": len(tokens),
                    "tensors": tensor_descs,
                })

            header = {
                "loss_fn": loss_fn,
                "loss_fn_config": loss_fn_config,
                "datums": datum_descs,
            }
            header_bytes = msgpack.packb(header, use_bin_type=True)
            parts = [len(header_bytes).to_bytes(4, "little"), header_bytes] + buffers
            payload_bytes = b"".join(parts)

            resp_raw = await self._post_bytes(
                self.base_url,
                "/forward_backward_bin",
                payload_bytes,
            )

            # Parse binary response
            resp_header_len = int.from_bytes(resp_raw[:4], "little")
            resp_header = msgpack.unpackb(resp_raw[4 : 4 + resp_header_len], raw=False)
            resp_buf = resp_raw[4 + resp_header_len :]

            metrics = {k: float(v) for k, v in (resp_header.get("metrics") or {}).items()}
            loss_fn_outputs: list[dict[str, TensorData]] = []
            buf_offset = 0
            for out_desc in resp_header.get("outputs") or []:
                out_tensors: dict[str, TensorData] = {}
                for k, info in out_desc.items():
                    dtype = np.dtype(info["dtype"])
                    shape = tuple(info["shape"])
                    nbytes = info["nbytes"]
                    arr = np.frombuffer(
                        resp_buf[buf_offset : buf_offset + nbytes], dtype=dtype
                    ).reshape(shape).copy()
                    buf_offset += nbytes
                    out_tensors[k] = TensorData(data=arr, dtype=str(arr.dtype), shape=shape)
                loss_fn_outputs.append(out_tensors)

            self._last_backward_outputs = loss_fn_outputs
            self._last_backward_metrics = metrics
            return ForwardBackwardOutput(
                loss_fn_outputs=loss_fn_outputs, metrics=metrics
            )

    async def optim_step(self, adam: AdamParams) -> OptimStepResponse:
        async with self._lock:
            body = {
                "learning_rate": float(adam.learning_rate),
                "beta1": float(adam.beta1),
                "beta2": float(adam.beta2),
                "eps": float(adam.eps),
                "weight_decay": 0.0,
            }
            result = await super()._post(self.base_url, "/optim_step", body)
            metrics = {k: float(v) for k, v in (result.get("metrics") or {}).items()}
            if self._last_backward_metrics:
                for k, v in self._last_backward_metrics.items():
                    metrics[f"train/{k}"] = float(v)
            return OptimStepResponse(metrics=metrics)

    async def save_state(self, name: str) -> _SavePath:
        async with self._lock:
            result = await super()._post(self.base_url, "/save_state", {"name": name})
            path = result["path"]
            d = _CheckpointStore.resolve(path)
            d.mkdir(parents=True, exist_ok=True)
            _CheckpointStore.write_meta(
                path,
                {
                    "user_metadata": self._user_metadata,
                    "model_name": self.model_name,
                    "lora_rank": self.lora_rank,
                    "ts": time.time(),
                },
            )
            return _SavePath(path=path)

    async def save_weights_for_sampler(self, name: str) -> _SavePath:
        async with self._lock:
            result = await super()._post(self.base_url, "/save_weights_for_sampler", {"name": name})
            path = result["path"]
            # In distillation2, save_weights_for_sampler pushes weights over
            # NCCL to the student vLLM and never writes to disk; the returned
            # tinther://sampler/<uid> is just a handle. We still need a local
            # directory to drop ``meta.json`` so downstream metadata reads
            # work — create it directly from the path components instead of
            # resolving (which insists the directory already exist).
            if path.startswith("tinther://"):
                kind, uid = path[len("tinther://"):].split("/", 1)
                d = _cache_root() / kind / uid
            else:
                d = Path(path)
            d.mkdir(parents=True, exist_ok=True)
            _CheckpointStore.write_meta(
                path,
                {
                    "user_metadata": self._user_metadata,
                    "model_name": self.model_name,
                    "ts": time.time(),
                },
            )
            return _SavePath(path=path)


# ---------------------------------------------------------------------------
# Rest client stub (checkpoint metadata)
# ---------------------------------------------------------------------------


class _TrainingRunInfo:
    def __init__(self, user_metadata: dict[str, str]):
        self.user_metadata = dict(user_metadata)


class _TrainingRunFuture:
    def __init__(self, info: _TrainingRunInfo):
        self._info = info

    def result(self) -> _TrainingRunInfo:
        return self._info


class _RestClientStub:
    """Minimal fake covering checkpoint_utils metadata calls."""

    def get_training_run_by_tinker_path(self, path: str) -> _TrainingRunFuture:
        meta = _CheckpointStore.read_meta(path)
        return _TrainingRunFuture(_TrainingRunInfo(meta.get("user_metadata", {})))

    async def get_training_run_by_tinker_path_async(self, path: str) -> _TrainingRunInfo:
        meta = _CheckpointStore.read_meta(path)
        return _TrainingRunInfo(meta.get("user_metadata", {}))

    async def delete_checkpoint_from_tinker_path_async(self, path: str) -> None:
        _CheckpointStore.delete(path)


# ---------------------------------------------------------------------------
# Public clients
# ---------------------------------------------------------------------------


class SamplingClient:
    """Mirrors tinker.SamplingClient.

    Always backed by ``_HTTPSamplerBackend`` — every sampler (student and
    teacher) is an external vLLM OpenAI-compatible HTTP server.
    """

    def __init__(self, backend: "_HTTPSamplerBackend"):
        self._backend = backend

    async def sample_async(self, prompt: ModelInput, num_samples: int, sampling_params: SamplingParams, include_prompt_logprobs: bool = False, topk_prompt_logprobs: int | None = None) -> SampleResponse:
        return await self._backend.sample(
            prompt, num_samples, sampling_params, include_prompt_logprobs, topk_prompt_logprobs
        )

    async def compute_logprobs_async(self, sequence: ModelInput) -> list[float | None]:
        return await self._backend.compute_logprobs(sequence)


class TrainingClient:
    """Mirrors tinker.TrainingClient.

    Talks to an external ``train.py`` server through
    :class:`_HTTPTrainerBackend`. Sampling clients minted via this class
    target the student vLLM server (``TINTHER_STUDENT_URL``); the trainer
    pushes weights into that server through a side-channel NCCL
    communicator on every ``save_weights_for_sampler`` call, so a fresh
    SamplingClient always reflects the latest student parameters.
    """

    def __init__(self, backend: "_HTTPTrainerBackend"):
        self._backend = backend

    def get_tokenizer(self):
        return self._backend.get_tokenizer()

    async def forward_backward_async(self, data_D: list[Datum], loss_fn: LossFnType, loss_fn_config: dict[str, Any] | None = None) -> APIFuture[ForwardBackwardOutput]:
        task = asyncio.create_task(
            self._backend.forward_backward(data_D, loss_fn, loss_fn_config)
        )
        return APIFuture(task)

    async def optim_step_async(self, adam_params: AdamParams) -> APIFuture[OptimStepResponse]:
        task = asyncio.create_task(self._backend.optim_step(adam_params))
        return APIFuture(task)

    async def save_state_async(self, name: str, ttl_seconds: int | None = None) -> APIFuture[_SavePath]:
        del ttl_seconds
        task = asyncio.create_task(self._backend.save_state(name))
        return APIFuture(task)

    async def save_weights_for_sampler_async(self, name: str, ttl_seconds: int | None = None) -> APIFuture[_SavePath]:
        del ttl_seconds
        task = asyncio.create_task(self._backend.save_weights_for_sampler(name))
        return APIFuture(task)

    async def save_weights_and_get_sampling_client_async(self) -> SamplingClient:
        saved = await self._backend.save_weights_for_sampler(f"adhoc-{int(time.time())}")
        return self._student_sampling_client(saved.path)

    def create_sampling_client(self, sampler_path: str) -> SamplingClient:
        return self._student_sampling_client(sampler_path)

    def _student_sampling_client(self, path: str) -> SamplingClient:
        return SamplingClient(_HTTPSamplerBackend.from_student_env(path))


class ServiceClient:
    """Mirrors tinker.ServiceClient.

    Training clients connect to an external ``train.py`` server via
    HTTP — launch it separately with ``torchrun`` and point clients at it
    through ``TINTHER_TRAIN_URL``.
    """

    def __init__(self, base_url: str | None = None) -> None:
        # base_url is accepted for API compatibility; it has no effect locally.
        self.base_url = base_url

    async def create_lora_training_client_async(
        self,
        model_name: str | None = None,
        rank: int = 0,
        user_metadata: dict[str, str] | None = None,
        *,
        base_model: str | None = None,
    ) -> TrainingClient:
        if model_name is None:
            model_name = base_model
        elif base_model is not None and base_model != model_name:
            raise ValueError(
                f"Received conflicting model_name={model_name!r} and "
                f"base_model={base_model!r}."
            )
        if model_name is None:
            raise TypeError("create_lora_training_client_async requires a model name")

        backend = await asyncio.to_thread(
            _HTTPTrainerBackend, model_name, rank, None, False
        )
        backend.set_user_metadata(user_metadata)
        return TrainingClient(backend)

    async def create_training_client_from_state_async(self, state_path: str, user_metadata: dict[str, str] | None = None) -> TrainingClient:
        meta = _CheckpointStore.read_meta(state_path)
        model_name = meta.get("model_name") or state_path
        lora_rank = meta.get("lora_rank")
        backend = await asyncio.to_thread(
            _HTTPTrainerBackend, model_name, lora_rank, state_path, False
        )
        backend.set_user_metadata(user_metadata)
        return TrainingClient(backend)

    async def create_training_client_from_state_with_optimizer_async(self, state_path: str, user_metadata: dict[str, str] | None = None) -> TrainingClient:
        meta = _CheckpointStore.read_meta(state_path)
        model_name = meta.get("model_name") or state_path
        lora_rank = meta.get("lora_rank")
        backend = await asyncio.to_thread(
            _HTTPTrainerBackend, model_name, lora_rank, state_path, True
        )
        backend.set_user_metadata(user_metadata)
        return TrainingClient(backend)

    def create_sampling_client(self, base_model: str, model_path: str | None = None) -> SamplingClient:
        """Create a sampler for an external (teacher) model.

        Routes through an external vLLM HTTP server configured via
        ``TINTHER_TEACHER_URL`` / ``TINTHER_TEACHER_URLS``. Student
        checkpoints (``tinther://sampler/...``) must go through
        :meth:`TrainingClient.create_sampling_client` instead, which targets
        the student vLLM server (``TINTHER_STUDENT_URL``).
        """
        model_ref = model_path or base_model
        if str(model_ref).startswith("tinther://"):
            raise TinkerError(
                "ServiceClient.create_sampling_client is for external teacher models. "
                "Use TrainingClient.create_sampling_client for student checkpoints "
                "at tinther:// paths (they hit the student vLLM server)."
            )
        return SamplingClient(_HTTPSamplerBackend.from_env(model_ref))

    def create_rest_client(self) -> _RestClientStub:
        return _RestClientStub()


# ---------------------------------------------------------------------------
# `tinker.types` submodule facade + sys.modules aliasing
# ---------------------------------------------------------------------------


def _build_types_module() -> _types_module.ModuleType:
    mod = _types_module.ModuleType("tinker.types")
    mod.LossFnType = LossFnType  # type: ignore[attr-defined]
    mod.SampleResponse = SampleResponse
    mod.ModelInput = ModelInput
    mod.TensorData = TensorData
    mod.Datum = Datum
    mod.SamplingParams = SamplingParams
    mod.AdamParams = AdamParams
    mod.ForwardBackwardOutput = ForwardBackwardOutput
    mod.OptimStepResponse = OptimStepResponse
    mod.StopReason = StopReason  # type: ignore[attr-defined]
    mod.EncodedTextChunk = EncodedTextChunk
    mod.ImageChunk = ImageChunk
    mod.ImageAssetPointerChunk = ImageAssetPointerChunk
    mod.ModelInputChunk = ModelInputChunk
    # Nested submodules referenced as `tinker.types.tensor_data`, etc.
    tensor_data_mod = _types_module.ModuleType("tinker.types.tensor_data")
    tensor_data_mod.TensorData = TensorData
    mod.tensor_data = tensor_data_mod
    image_chunk_mod = _types_module.ModuleType("tinker.types.image_chunk")
    image_chunk_mod.ImageChunk = ImageChunk
    image_chunk_mod.ImageAssetPointerChunk = ImageAssetPointerChunk
    mod.image_chunk = image_chunk_mod
    return mod


types = _build_types_module()


def install_as_tinker() -> None:
    """Register this module as ``tinker`` / ``tinker.types`` in sys.modules.

    Call this once, before importing any ``tinker_cookbook`` module, so
    every downstream ``import tinker`` resolves to ``tinther``.
    """
    this_mod = sys.modules[__name__]
    sys.modules["tinker"] = this_mod
    sys.modules["tinker.types"] = types
    sys.modules["tinker.types.tensor_data"] = types.tensor_data  # type: ignore[attr-defined]
    sys.modules["tinker.types.image_chunk"] = types.image_chunk  # type: ignore[attr-defined]


__all__ = [
    "APIFuture",
    "AdamParams",
    "Datum",
    "EncodedTextChunk",
    "ForwardBackwardOutput",
    "ImageAssetPointerChunk",
    "ImageChunk",
    "LossFnType",
    "ModelInput",
    "ModelInputChunk",
    "OptimStepResponse",
    "SampleResponse",
    "SamplingClient",
    "SamplingParams",
    "ServiceClient",
    "StopReason",
    "TensorData",
    "TinkerError",
    "TrainingClient",
    "_HTTPSamplerBackend",
    "_HTTPTrainerBackend",
    "install_as_tinker",
    "types",
]
