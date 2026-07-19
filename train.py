"""Stateful distributed LLM finetuning runtime.

Run via torchrun on every node. Rank 0 hosts a FastAPI server that exposes
forward_backward / optim_step / save_state / save_weights_for_sampler endpoints.
All ranks run a synchronous command loop driven by torch.distributed broadcasts
from rank 0.

This server is the compute backend for ``tinther._HTTPTrainerBackend``: client
code (e.g. ``train_sft_tinther.py``) submits ``Datum`` lists with arbitrary
``loss_fn_inputs`` shapes, the server dispatches the appropriate loss
(cross_entropy / importance_sampling / ppo) and returns per-position logprobs
plus aggregated metrics. LoRA, save_state, mixed precision, and
``flash_attention_2`` parity with ``tinther._TrainerBackend`` are all here.

``save_weights_for_sampler`` no longer writes to disk. Instead, rank 0 holds
a side-channel ``PyNcclCommunicator`` with the user-launched student vLLM
server (see ``vllm_student_server.py``) and broadcasts the merged base
parameters tensor-by-tensor on every call. The returned ``tinther://sampler/<uid>``
is just a handle for downstream HTTP sampling clients.

Required environment when calling save_weights_for_sampler::

    TINTHER_STUDENT_URL=http://127.0.0.1:8001
    TINTHER_STUDENT_NCCL_MASTER_ADDR=127.0.0.1
    TINTHER_STUDENT_NCCL_MASTER_PORT=29500
    TINTHER_STUDENT_NCCL_WORLD_SIZE=2     # 1 (trainer) + tensor_parallel * pipeline

Example::

    torchrun --nnodes=1 --nproc_per_node=1 train.py \\
        --model HuggingFaceTB/SmolLM2-135M \\
        --precision bf16 \\
        --lora-rank 8 \\
        --host 0.0.0.0 --port 8000

v1 limitations:
  * Multi-rank training requires ``len(datums) % world_size == 0``.
  * Single student vLLM endpoint (set via TINTHER_STUDENT_URL).
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import datetime as _dt
import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import msgpack
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------------
class TensorPayload(BaseModel):
    """JSON envelope for a numeric tensor."""

    dtype: str
    shape: list[int]
    data: list[Any]


class DatumPayload(BaseModel):
    model_input: list[int]
    loss_fn_inputs: dict[str, TensorPayload]


class ForwardBackwardRequest(BaseModel):
    loss_fn: str
    loss_fn_config: Optional[dict[str, Any]] = None
    datums: list[DatumPayload]


class ForwardBackwardResponse(BaseModel):
    metrics: dict[str, float]
    loss_fn_outputs: list[dict[str, TensorPayload]]


class AdamParams(BaseModel):
    learning_rate: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0


class OptimStepResponse(BaseModel):
    metrics: dict[str, float]


class SaveRequest(BaseModel):
    name: str


class SaveResponse(BaseModel):
    path: str


class OkResponse(BaseModel):
    ok: bool = True


class HealthResponse(BaseModel):
    rank: int
    world_size: int
    precision: str
    model_name: str
    lora_rank: int
    cache_dir: str


# ----------------------------------------------------------------------------
# Global state (single session per process)
# ----------------------------------------------------------------------------
class State:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device = torch.device("cpu")
    precision: str = "fp32"
    model_name: str = ""
    lora_rank: int = 0
    max_grad_norm: Optional[float] = None
    cache_dir: Path = Path("/tmp/tinther")
    model: Any = None        # DDP-wrapped nn.Module
    optimizer: Any = None    # torch.optim.AdamW
    tokenizer: Any = None
    last_loss: Optional[torch.Tensor] = None
    cmd_queue: Optional[queue.Queue] = None  # rank 0 only
    request_lock = threading.Lock()
    # Side-channel NCCL communicator with the student vLLM server. Built
    # lazily on the first save_weights_for_sampler call (rank 0 only).
    student_comm: Any = None
    student_world_size: int = 0


STATE = State()
log = logging.getLogger("train")


@dataclasses.dataclass
class Command:
    kind: str  # "forward_backward" | "optim_step" | "save_state" | "save_sampler" | "shutdown"
    payload: Optional[dict[str, Any]] = None


_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _backward_loss_scale(
    ddp_world_size: int,
    global_valid_token_count: float | None,
) -> float:
    """Return the local-loss scale that produces a global token mean in DDP."""
    if ddp_world_size < 1:
        raise ValueError(f"ddp_world_size must be positive; got {ddp_world_size}")
    if global_valid_token_count is None:
        return float(ddp_world_size)
    if not np.isfinite(global_valid_token_count) or global_valid_token_count <= 0:
        raise ValueError(
            "global_valid_token_count must be positive and finite; "
            f"got {global_valid_token_count!r}"
        )
    return ddp_world_size / global_valid_token_count


def _infer_global_valid_token_count(datums_all: list[dict]) -> float:
    """Infer action-token count from the zero-logprob observation sentinel.

    ``assemble_training_data`` stores exactly ``0.0`` at prompt/observation
    positions and sampled log probabilities at action positions. This avoids
    transmitting the training mask, but an action token whose sampled log
    probability is exactly zero cannot be distinguished from an observation.
    """
    valid_token_count = 0
    for datum_idx, datum in enumerate(datums_all):
        logprobs_payload = datum["loss_fn_inputs"].get("logprobs")
        if logprobs_payload is None:
            raise ValueError(
                "valid-token inference requires logprobs for every datum; "
                f"datum {datum_idx} has none"
            )

        shape = tuple(int(dim) for dim in logprobs_payload["shape"])
        try:
            logprobs = np.asarray(logprobs_payload["data"], dtype=np.float32).reshape(shape)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid logprobs payload for datum {datum_idx}") from exc

        if not np.all(np.isfinite(logprobs)):
            raise ValueError(f"logprobs for datum {datum_idx} contain non-finite values")
        valid_token_count += int(np.count_nonzero(logprobs != 0.0))

    if valid_token_count <= 0:
        raise ValueError("no valid training tokens inferred from logprobs")
    return float(valid_token_count)


# ----------------------------------------------------------------------------
# Loss functions (mirror tinther._LossFns)
# ----------------------------------------------------------------------------
class _LossFns:
    @staticmethod
    def _validate_sequence_shapes(
        loss_name: str,
        logits: torch.Tensor,
        targets: torch.Tensor,
        allowed_target_ndims: tuple[int, ...],
    ) -> None:
        """Validate per-datum tensor ranks and the shared sequence length N."""
        if logits.ndim != 2:
            raise ValueError(
                f"{loss_name} requires logits with shape (N, V); "
                f"got {tuple(logits.shape)}"
            )
        if targets.ndim not in allowed_target_ndims:
            allowed = " or ".join(
                "(N,)" if ndim == 1 else "(N, K)" for ndim in allowed_target_ndims
            )
            raise ValueError(
                f"{loss_name} requires target_tokens with shape {allowed}; "
                f"got {tuple(targets.shape)}"
            )
        if targets.shape[0] != logits.shape[0]:
            raise ValueError(
                f"{loss_name} requires target_tokens sequence length to match logits; "
                f"got {targets.shape[0]} target positions and {logits.shape[0]} logit positions"
            )

    @staticmethod
    def _gather_target_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        if targets.ndim == 1:
            return logp.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1)
        return logp.gather(-1, targets.long())

    @staticmethod
    def cross_entropy(logits, inputs, loss_fn_config):
        del loss_fn_config
        targets = inputs["target_tokens"]
        _LossFns._validate_sequence_shapes(
            "cross_entropy", logits, targets, allowed_target_ndims=(1, 2)
        )
        weights = inputs.get("weights")
        if weights is None:
            weights = torch.ones_like(targets, dtype=logits.dtype)
        elif weights.shape != targets.shape:
            raise ValueError(
                "cross_entropy requires weights and target_tokens to have identical "
                f"shapes; got {tuple(weights.shape)} and {tuple(targets.shape)}"
            )
        tgt_lp = _LossFns._gather_target_logits(logits, targets)
        if targets.ndim == 1:
            per_tok = -weights.to(tgt_lp.dtype) * tgt_lp
            out_lp = tgt_lp
        else:
            per_tok = -(weights.to(tgt_lp.dtype) * tgt_lp).sum(-1)
            out_lp = tgt_lp[..., 0]
        loss = per_tok.sum()
        loss_value = float(loss.detach().item())
        metrics = {
            "loss:sum": loss_value,
            # Keep the legacy key for existing tinther consumers.
            "total_loss": loss_value,
            "n_tokens": float(weights.sum().detach().item()),
        }
        return loss, out_lp.detach(), metrics

    @staticmethod
    def importance_sampling(logits, inputs, loss_fn_config):
        del loss_fn_config
        targets = inputs["target_tokens"].long()
        old_lp = inputs["logprobs"].to(logits.dtype)
        adv = inputs["advantages"].to(logits.dtype)
        _LossFns._validate_sequence_shapes(
            "importance_sampling", logits, targets, allowed_target_ndims=(1,)
        )
        if targets.shape != old_lp.shape or targets.shape != adv.shape:
            raise ValueError(
                "importance_sampling requires target_tokens, logprobs, and advantages "
                f"to have identical shapes; got {targets.shape}, {old_lp.shape}, "
                f"and {adv.shape}"
            )

        new_lp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        ratio = torch.exp(new_lp - old_lp)
        loss = -(ratio * adv).sum()
        loss_value = float(loss.detach().item())
        metrics = {
            "loss:sum": loss_value,
            # Keep the legacy key for existing tinther consumers.
            "total_loss": loss_value,
        }
        return loss, new_lp.detach(), metrics

    @staticmethod
    def ppo(logits, inputs, loss_fn_config):
        clip_eps = float((loss_fn_config or {}).get("clip_eps", 0.2))
        targets = inputs["target_tokens"].long()
        old_lp = inputs["logprobs"].to(logits.dtype)
        adv = inputs["advantages"].to(logits.dtype)
        _LossFns._validate_sequence_shapes("ppo", logits, targets, allowed_target_ndims=(1,))
        if targets.shape != old_lp.shape or targets.shape != adv.shape:
            raise ValueError(
                "ppo requires target_tokens, logprobs, and advantages to have identical "
                f"shapes; got {targets.shape}, {old_lp.shape}, and {adv.shape}"
            )

        new_lp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        ratio = torch.exp(new_lp - old_lp)
        unclipped = ratio * adv
        clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv
        loss = -torch.minimum(unclipped, clipped).sum()
        loss_value = float(loss.detach().item())
        metrics = {
            "loss:sum": loss_value,
            # Keep the legacy key for existing tinther consumers.
            "total_loss": loss_value,
        }
        return loss, new_lp.detach(), metrics

    DISPATCH: dict[str, Any] = {}


_LossFns.DISPATCH = {
    "cross_entropy": _LossFns.cross_entropy,
    "importance_sampling": _LossFns.importance_sampling,
    "ppo": _LossFns.ppo,
}


# ----------------------------------------------------------------------------
# Tensor (de)serialization
# ----------------------------------------------------------------------------
def _payload_to_torch(payload: dict, device: torch.device) -> torch.Tensor:
    """Reconstruct a torch tensor from a TensorPayload dict."""
    data = payload["data"]
    if isinstance(data, np.ndarray):
        arr = data
    else:
        arr = np.asarray(data).astype(np.dtype(payload["dtype"]))
        arr = arr.reshape(tuple(payload["shape"]))
    return torch.from_numpy(arr.copy()).to(device)


def _torch_to_payload(t: torch.Tensor) -> dict:
    t = t.detach().to("cpu").contiguous()
    if t.dtype == torch.bfloat16:
        t = t.to(torch.float32)
    arr = t.numpy()
    return {"dtype": str(arr.dtype), "shape": list(arr.shape), "data": arr.tolist()}


# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
def setup_distributed() -> None:
    STATE.rank = int(os.environ["RANK"])
    STATE.world_size = int(os.environ["WORLD_SIZE"])
    STATE.local_rank = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(STATE.local_rank)
    STATE.device = torch.device(f"cuda:{STATE.local_rank}")
    timeout_s = int(os.environ.get("TINTHER_TRAIN_NCCL_TIMEOUT_S", "3600"))
    dist.init_process_group(
        backend="nccl",
        timeout=_dt.timedelta(seconds=timeout_s),
    )


def _resolve_attn_implementation(requested: str) -> tuple[Optional[str], bool]:
    """Returns (attn_impl, allow_fallback). Mirrors tinther._TrainerBackend."""
    if requested == "auto":
        try:
            from transformers.utils import is_flash_attn_2_available

            if is_flash_attn_2_available():
                return "flash_attention_2", True
        except Exception:
            pass
        return None, True
    if requested in ("flash_attention_2", "fa2"):
        return "flash_attention_2", False
    if requested in ("sdpa", "eager"):
        return requested, False
    raise ValueError(f"unsupported --attn-implementation: {requested!r}")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        f"{name} must be a boolean value like 0/1, true/false, or on/off; got {raw!r}"
    )


def build_model(
    model_name: str,
    precision: str,
    lora_rank: int,
    attn_impl: str,
    from_state: Optional[str],
) -> None:
    # Resolve checkpoint source (model_name string is preserved for meta).
    base_source = from_state if from_state else model_name
    tok_source = base_source

    # Tokenizer (server only uses pad_token_id; clients load their own).
    STATE.tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if STATE.tokenizer.pad_token_id is None:
        STATE.tokenizer.pad_token_id = STATE.tokenizer.eos_token_id

    dtype = _DTYPE[precision] if precision in _DTYPE else torch.float32
    chosen_attn, allow_fallback = _resolve_attn_implementation(attn_impl)

    model_kwargs: dict[str, Any] = {"torch_dtype": dtype, "trust_remote_code": True}
    if chosen_attn is not None:
        model_kwargs["attn_implementation"] = chosen_attn

    try:
        base = AutoModelForCausalLM.from_pretrained(base_source, **model_kwargs)
    except Exception as e:
        if chosen_attn is not None and allow_fallback and any(
            needle in str(e).lower()
            for needle in ("flash attention", "flash_attn", "attn_implementation", "attention implementation")
        ):
            log.warning(
                "Auto-selected attn_implementation=%s failed (%s); retrying with default.",
                chosen_attn,
                e,
            )
            model_kwargs.pop("attn_implementation", None)
            base = AutoModelForCausalLM.from_pretrained(base_source, **model_kwargs)
        else:
            raise

    if getattr(base.config, "use_cache", None):
        base.config.use_cache = False
    if _env_bool("TINTHER_GRADIENT_CHECKPOINTING", True):
        base.gradient_checkpointing_enable()
        log.info("gradient checkpointing enabled")
    else:
        log.info("gradient checkpointing disabled by TINTHER_GRADIENT_CHECKPOINTING")
    base.to(STATE.device)

    if lora_rank > 0:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=2 * lora_rank,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        base = get_peft_model(base, lora_config)
        base.to(STATE.device)

    from torch.nn.parallel import DistributedDataParallel as DDP

    STATE.model = DDP(base, device_ids=[STATE.local_rank])
    trainable = [p for p in STATE.model.parameters() if p.requires_grad]
    STATE.optimizer = torch.optim.AdamW(trainable, lr=1e-5)

    # Optionally restore optimizer state from a saved-state directory.
    if from_state:
        opt_path = Path(from_state) / "optim.pt"
        if opt_path.exists():
            try:
                STATE.optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
                log.info("restored optimizer state from %s", opt_path)
            except Exception as e:
                log.warning("could not restore optimizer state from %s: %s", opt_path, e)


# ----------------------------------------------------------------------------
# Per-rank operations
# ----------------------------------------------------------------------------
def _resolve_grad_accum_steps() -> int:
    """TINTHER_GRAD_ACCUM_STEPS: number of micro-batches to split each
    forward_backward call into. 1 (or unset) = single-shot behavior."""
    raw = os.environ.get("TINTHER_GRAD_ACCUM_STEPS")
    if raw is None or raw == "":
        return 1
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(
            f"TINTHER_GRAD_ACCUM_STEPS must be an int, got {raw!r}"
        ) from e
    if value < 1:
        raise ValueError("TINTHER_GRAD_ACCUM_STEPS must be >= 1")
    return value


def _slice_for_rank(datums: list[dict]) -> tuple[list[dict], int]:
    """Return (this_rank_slice, n_global)."""
    n_global = len(datums)
    if n_global % STATE.world_size != 0:
        raise RuntimeError(
            f"len(datums)={n_global} not divisible by world_size={STATE.world_size}"
        )
    per = n_global // STATE.world_size
    start = STATE.rank * per
    return datums[start : start + per], n_global


def _materialize_local_inputs(local: list[dict]) -> tuple[
    torch.Tensor, torch.Tensor, list[int], list[dict[str, torch.Tensor]]
]:
    """Pad token lists into (B, S) and decode loss_fn_inputs to torch on device."""
    pad_id = STATE.tokenizer.pad_token_id
    token_lists = [d["model_input"] for d in local]
    seq_lens = [len(t) for t in token_lists]
    max_seq = max(seq_lens) if seq_lens else 0

    input_ids = torch.full(
        (len(local), max_seq), pad_id, dtype=torch.long, device=STATE.device
    )
    attention_mask = torch.zeros(
        (len(local), max_seq), dtype=torch.long, device=STATE.device
    )
    for i, toks in enumerate(token_lists):
        n = len(toks)
        input_ids[i, :n] = torch.tensor(toks, dtype=torch.long, device=STATE.device)
        attention_mask[i, :n] = 1

    loss_inputs_per_datum: list[dict[str, torch.Tensor]] = []
    for d in local:
        decoded = {
            k: _payload_to_torch(v, STATE.device) for k, v in d["loss_fn_inputs"].items()
        }
        loss_inputs_per_datum.append(decoded)
    return input_ids, attention_mask, seq_lens, loss_inputs_per_datum


def do_forward_backward(payload: dict) -> dict:
    """Forward + backward on this rank's slice, gather outputs back to rank 0.

    When ``TINTHER_GRAD_ACCUM_STEPS`` (K) is set, the per-rank batch is split
    into K micro-batches; each micro-batch runs an independent forward+backward
    so peak activation memory scales with the micro-batch size rather than the
    full local batch. Loss implementations return token sums, which are summed
    over datums and micro-batches. Because default DDP averages gradients across
    ranks, every micro-batch loss is multiplied by the DDP world size. For
    importance-sampling and PPO losses, the full-request valid-token count is
    inferred from the sampled-logprob sentinel and used for every micro-batch,
    so accumulation reproduces a global token-mean loss. DDP all-reduce is
    suppressed via ``no_sync()`` on all but the last micro-batch so collectives
    fire once per forward_backward call.
    """
    loss_fn = payload["loss_fn"]
    loss_fn_config = payload.get("loss_fn_config") or None
    datums_all: list[dict] = payload["datums"]

    if loss_fn not in _LossFns.DISPATCH:
        raise ValueError(f"unknown loss_fn: {loss_fn}")
    loss_impl = _LossFns.DISPATCH[loss_fn]

    global_valid_token_count = (
        _infer_global_valid_token_count(datums_all)
        if loss_fn in {"importance_sampling", "ppo"}
        else None
    )

    local, n_global = _slice_for_rank(datums_all)
    STATE.model.train()
    n_local = len(local)
    if n_local == 0:
        raise RuntimeError(
            f"empty rank slice (n_global={n_global}, world_size={STATE.world_size})"
        )
    ddp_world_size = dist.get_world_size()
    backward_loss_scale = _backward_loss_scale(
        ddp_world_size,
        global_valid_token_count,
    )

    k_requested = _resolve_grad_accum_steps()
    k_eff = max(1, min(k_requested, n_local))

    idx_groups: list[list[int]] = [
        list(g) for g in np.array_split(np.arange(n_local), k_eff)
    ]
    idx_groups = [g for g in idx_groups if len(g) > 0]

    local_metric_sums: dict[str, float] = {}
    loss_fn_outputs_local: list[dict[str, dict] | None] = [None] * n_local

    for mb_i, idxs in enumerate(idx_groups):
        is_last = mb_i == len(idx_groups) - 1
        sync_ctx = (
            contextlib.nullcontext() if is_last else STATE.model.no_sync()          # DDP no_sync to suppress all-reduce until the last micro-batch
        )
        with sync_ctx:
            mb_local = [local[i] for i in idxs]
            input_ids, attention_mask, seq_lens, loss_inputs_per_datum = (
                _materialize_local_inputs(mb_local)
            )

            if STATE.precision != "fp32":
                with torch.autocast(device_type="cuda", dtype=_DTYPE[STATE.precision]):
                    out = STATE.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                out = STATE.model(input_ids=input_ids, attention_mask=attention_mask)
            batch_logits = out.logits  # (B, S, V)

            mb_loss_sum = torch.zeros((), device=STATE.device)
            for j, datum_idx in enumerate(idxs):
                logits = batch_logits[j, : seq_lens[j]]                             # (seq_lens[j], vocab_size)
                inputs = loss_inputs_per_datum[j]                                   # dict[str, torch.Tensor]

                loss, per_pos_lp, metrics = loss_impl(logits, inputs, loss_fn_config)
                mb_loss_sum = mb_loss_sum + loss

                loss_fn_outputs_local[datum_idx] = {
                    "logprobs": _torch_to_payload(per_pos_lp)
                }
                for k, v in metrics.items():
                    local_metric_sums[k] = local_metric_sums.get(k, 0.0) + float(v)

            # DDP averages gradients across ranks. Scale every micro-batch by W
            # so no_sync accumulation reproduces a global sum. If N is supplied,
            # use the same full-request denominator for every micro-batch:
            # (1 / W) * sum_r grad((W / N) * local_loss_r)
            #     == grad(global_loss / N).
            backward_loss = mb_loss_sum * backward_loss_scale
            backward_loss.backward()

    STATE.last_loss = None

    return {
        "metrics_local_sums": local_metric_sums,
        "n_local": n_local,
        "n_global": n_global,
        "loss_fn_outputs_local": loss_fn_outputs_local,
    }


def do_optim(adam: dict) -> dict:
    for g in STATE.optimizer.param_groups:
        g["lr"] = float(adam["learning_rate"])
        g["betas"] = (float(adam["beta1"]), float(adam["beta2"]))
        g["eps"] = float(adam["eps"])
        g["weight_decay"] = float(adam.get("weight_decay", 0.0))

    grad_norm = 0.0
    n_p = 0
    for p in STATE.model.parameters():
        if p.grad is not None:
            grad_norm += float(p.grad.detach().norm(2).item()) ** 2
            n_p += 1
    grad_norm = float(grad_norm**0.5) if n_p else 0.0

    if STATE.max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(STATE.model.parameters(), STATE.max_grad_norm)

    STATE.optimizer.step()
    STATE.optimizer.zero_grad(set_to_none=True)

    return {
        "metrics": {
            "optim/lr": float(adam["learning_rate"]),
            "optim/grad_norm": grad_norm,
        }
    }


def _unwrap(model):
    """Return the inner nn.Module (peeling the DDP wrapper)."""
    inner = getattr(model, "module", model)
    return inner


def do_save_state(payload: dict) -> dict:
    name = payload["name"]
    uid = f"{name}-{uuid.uuid4().hex[:6]}"
    d = STATE.cache_dir / "state" / uid
    if STATE.rank == 0:
        d.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    inner = _unwrap(STATE.model)
    if STATE.rank == 0:
        inner.save_pretrained(str(d))
        STATE.tokenizer.save_pretrained(str(d))
        try:
            if STATE.optimizer is not None:
                torch.save(STATE.optimizer.state_dict(), str(d / "optim.pt"))
        except Exception as e:
            log.warning("could not save optimizer state: %s", e)
        meta = {
            "model_name": STATE.model_name,
            "lora_rank": STATE.lora_rank,
            "ts": time.time(),
        }
        with (d / "meta.json").open("w") as f:
            json.dump(meta, f)
    dist.barrier()
    return {"path": f"tinther://state/{uid}"}


def _student_url() -> str:
    url = os.environ.get("TINTHER_STUDENT_URL")
    if not url:
        raise RuntimeError(
            "save_weights_for_sampler requires TINTHER_STUDENT_URL to point at the "
            "tinther student vLLM server (started via vllm_student_server.py)."
        )
    return url.rstrip("/")


def _student_nccl_config() -> tuple[str, int, int]:
    addr = os.environ.get("TINTHER_STUDENT_NCCL_MASTER_ADDR")
    port = int(os.environ.get("TINTHER_STUDENT_NCCL_MASTER_PORT", "0"))
    world_size = int(os.environ.get("TINTHER_STUDENT_NCCL_WORLD_SIZE", "0"))
    if not addr or port <= 0 or world_size < 2:
        raise RuntimeError(
            "save_weights_for_sampler requires TINTHER_STUDENT_NCCL_MASTER_ADDR / "
            "TINTHER_STUDENT_NCCL_MASTER_PORT / TINTHER_STUDENT_NCCL_WORLD_SIZE "
            "(world_size = 1 + tensor_parallel * pipeline)."
        )
    return addr, port, world_size


def _post_student(path: str, body: Optional[dict] = None) -> dict:
    url = f"{_student_url()}{path}"
    data = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    timeout = float(os.environ.get("TINTHER_STUDENT_HTTP_TIMEOUT_S", "1800"))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"student vLLM at {url} unreachable: {e}") from e


def _ensure_student_comm() -> None:
    """Build the trainer-side PyNcclCommunicator on rank 0.

    Both sides must enter ``StatelessProcessGroup.create`` concurrently for
    the TCPStore rendezvous to complete. We kick off the server-side init
    RPC in a background thread, then run our own ``create`` in the
    foreground; once both have rendezvoused the worker RPC returns and we
    proceed to build the communicator.
    """
    if STATE.rank != 0 or STATE.student_comm is not None:
        return

    addr, port, world_size = _student_nccl_config()

    rpc_exc: list[BaseException] = []

    def _drive_init_rpc() -> None:
        try:
            _post_student("/trainer/init_weight_update_pg")
        except BaseException as e:  # re-raised after join below
            rpc_exc.append(e)

    rpc_thread = threading.Thread(
        target=_drive_init_rpc, name="tinther-rpc-init", daemon=True
    )
    rpc_thread.start()

    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    timeout_s = int(os.environ.get("TINTHER_STUDENT_NCCL_TIMEOUT_S", "1800"))
    pg = StatelessProcessGroup.create(
        host=addr,
        port=port,
        rank=0,
        world_size=world_size,
        store_timeout=timeout_s,
    )

    # Build the PyNcclCommunicator on the trainer side BEFORE joining the RPC
    # thread: the student-side ``init_weight_update_pg`` RPC also blocks inside
    # ``PyNcclCommunicator.__init__`` waiting for this side, and our HTTP RPC
    # only returns after the worker RPC completes. Joining the thread before
    # constructing our own communicator deadlocks both sides.
    device = torch.device(f"cuda:{STATE.local_rank}")
    STATE.student_comm = PyNcclCommunicator(pg, device=device)

    rpc_thread.join()
    if rpc_exc:
        raise rpc_exc[0]
    STATE.student_world_size = world_size
    log.info(
        "tinther weight-update communicator ready on trainer rank 0 (peers=%d, %s:%s)",
        world_size - 1,
        addr,
        port,
    )


def _torch_dtype_str(t: torch.Tensor) -> str:
    s = str(t.dtype)
    return s[len("torch."):] if s.startswith("torch.") else s


def _push_weights_to_student(state_dict_iter) -> int:
    """Broadcast each tensor in ``state_dict_iter`` to vLLM workers.

    For each (name, tensor): announce metadata via HTTP so the server fires
    ``recv_weight`` on every worker, then perform the NCCL broadcast in lock
    step. We push tensors one at a time; the server-side dispatch is also
    sequential, so order is well-defined.
    """
    if STATE.student_comm is None:
        raise RuntimeError("student communicator not initialized")
    n = 0
    metas: list[dict] = []
    tensors: list[torch.Tensor] = []
    for name, tensor in state_dict_iter:
        if not isinstance(tensor, torch.Tensor):
            continue
        t = tensor.detach()
        if not t.is_cuda:
            t = t.to(STATE.device)
        t = t.contiguous()
        metas.append(
            {"name": name, "dtype": _torch_dtype_str(t), "shape": list(t.shape)}
        )
        tensors.append(t)

    # Prime the worker side: each entry triggers one collective_rpc(
    # "recv_weight", ...), which blocks each worker inside its NCCL
    # broadcast. We then satisfy those broadcasts in the same order.
    # The HTTP call drives the loop on the server; we run our broadcasts
    # in a background thread so both sides progress concurrently.
    import threading as _threading

    rpc_exc: list[BaseException] = []

    def _drive_rpc():
        try:
            _post_student("/trainer/update_weights", {"weights": metas})
        except BaseException as e:  # pragma: no cover - re-raised below
            rpc_exc.append(e)

    rpc_thread = _threading.Thread(target=_drive_rpc, name="tinther-rpc-update", daemon=True)
    rpc_thread.start()

    stream = torch.cuda.current_stream()
    for t in tensors:
        STATE.student_comm.broadcast(t, src=0, stream=stream)
        n += 1
    stream.synchronize()

    rpc_thread.join()
    if rpc_exc:
        raise rpc_exc[0]
    return n


def do_save_sampler(payload: dict) -> dict:
    """Push current weights into the student vLLM via NCCL.

    We don't write anything to disk: ``tinther://sampler/<uid>`` is now a
    pure handle that downstream code uses to construct an HTTP sampling
    client pointed at the student server.
    """
    name = payload["name"]
    uid = f"{name}-{uuid.uuid4().hex[:6]}"

    inner = _unwrap(STATE.model)
    if STATE.rank == 0:
        _ensure_student_comm()
        try:
            from peft import PeftModel
        except Exception:  # pragma: no cover
            PeftModel = None  # type: ignore

        if PeftModel is not None and isinstance(inner, PeftModel):
            merged = copy.deepcopy(inner).merge_and_unload()
            try:
                n = _push_weights_to_student(merged.named_parameters())
            finally:
                del merged
        else:
            n = _push_weights_to_student(inner.named_parameters())
        # Match TRL: invalidate the engine's prefix cache so subsequent
        # prompts don't reuse KV blocks computed under the old weights.
        # See trl/generation/vllm_generation.py: after update_named_param
        # loop, it calls vllm_client.reset_prefix_cache().
        try:
            _post_student("/trainer/reset_prefix_cache")
        except Exception as e:
            log.warning("could not reset student prefix cache: %s", e)
        log.info("pushed %d tensors to student vLLM (sampler=%s)", n, uid)
    dist.barrier()
    return {"path": f"tinther://sampler/{uid}"}


# ----------------------------------------------------------------------------
# Worker loop (runs on every rank; rank 0 also pumps from cmd_queue)
# ----------------------------------------------------------------------------
def run_worker_loop() -> None:
    while True:
        if STATE.rank == 0:
            pkg = STATE.cmd_queue.get()
            cmd: Command = pkg["cmd"]
            dist.broadcast_object_list([cmd], src=0)
        else:
            holder: list = [None]
            dist.broadcast_object_list(holder, src=0)
            cmd = holder[0]
            pkg = None

        try:
            if cmd.kind == "forward_backward":
                # cmd.payload was already broadcast as part of the cmd object.
                payload = cmd.payload
                local_result = do_forward_backward(payload)

                # Gather per-datum outputs back to rank 0.
                gathered: list = [None] * STATE.world_size if STATE.rank == 0 else None
                dist.gather_object(local_result, object_gather_list=gathered, dst=0)

                if STATE.rank == 0:
                    metrics_sums: dict[str, float] = {}
                    n_global = local_result["n_global"]
                    all_outputs: list[dict[str, dict]] = [None] * n_global  # type: ignore
                    for r, g in enumerate(gathered):  # type: ignore
                        per_rank = n_global // STATE.world_size
                        offset = r * per_rank
                        for i, out in enumerate(g["loss_fn_outputs_local"]):
                            all_outputs[offset + i] = out
                        for k, v in g["metrics_local_sums"].items():
                            metrics_sums[k] = metrics_sums.get(k, 0.0) + float(v)

                    # Loss functions emit additive metrics (for example
                    # ``loss:sum``), so gathering across ranks is a sum as well.
                    aggregated = metrics_sums
                    pkg["future"].set_result(
                        {"metrics": aggregated, "loss_fn_outputs": all_outputs}
                    )

            elif cmd.kind == "optim_step":
                result = do_optim(cmd.payload or {})
                if STATE.rank == 0:
                    pkg["future"].set_result(result)

            elif cmd.kind == "save_state":
                result = do_save_state(cmd.payload or {})
                if STATE.rank == 0:
                    pkg["future"].set_result(result)

            elif cmd.kind == "save_sampler":
                result = do_save_sampler(cmd.payload or {})
                if STATE.rank == 0:
                    pkg["future"].set_result(result)

            elif cmd.kind == "shutdown":
                if STATE.rank == 0 and STATE.student_comm is not None:
                    try:
                        _post_student("/trainer/close_weight_update_pg")
                    except Exception as e:  # best effort
                        log.warning("could not close student weight-update PG: %s", e)
                    STATE.student_comm = None
                if STATE.rank == 0:
                    pkg["future"].set_result(None)
                break

            else:
                raise ValueError(f"unknown command: {cmd.kind}")

        except Exception as e:
            log.exception("rank %d failed on %s", STATE.rank, cmd.kind)
            if STATE.rank == 0 and pkg is not None:
                pkg["future"].set_exception(e)
            raise


# ----------------------------------------------------------------------------
# FastAPI app (rank 0 only)
# ----------------------------------------------------------------------------
app = FastAPI(title="LLM Finetuning Runtime")


def _submit(cmd: Command) -> Any:
    fut: Future = Future()
    pkg = {"cmd": cmd, "future": fut}
    STATE.cmd_queue.put(pkg)
    return fut.result()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        rank=STATE.rank,
        world_size=STATE.world_size,
        precision=STATE.precision,
        model_name=STATE.model_name,
        lora_rank=STATE.lora_rank,
        cache_dir=str(STATE.cache_dir),
    )


@app.post("/forward_backward", response_model=ForwardBackwardResponse)
def forward_backward(req: ForwardBackwardRequest) -> ForwardBackwardResponse:
    with STATE.request_lock:
        try:
            payload = {
                "loss_fn": req.loss_fn,
                "loss_fn_config": req.loss_fn_config,
                "datums": [
                    {
                        "model_input": d.model_input,
                        "loss_fn_inputs": {
                            k: v.model_dump() for k, v in d.loss_fn_inputs.items()
                        },
                    }
                    for d in req.datums
                ],
            }
        except Exception as e:
            raise HTTPException(400, f"failed to deserialize datums: {e}")

        if len(req.datums) % STATE.world_size != 0:
            raise HTTPException(
                400,
                f"len(datums)={len(req.datums)} not divisible by world_size={STATE.world_size}",
            )

        result = _submit(Command(kind="forward_backward", payload=payload))
        return ForwardBackwardResponse(
            metrics=result["metrics"],
            loss_fn_outputs=[
                {k: TensorPayload(**v) for k, v in out.items()}
                for out in result["loss_fn_outputs"]
            ],
        )


@app.post("/forward_backward_bin")
async def forward_backward_bin(request: Request) -> Response:
    """Binary forward_backward: msgpack header + concatenated numpy buffers."""
    with STATE.request_lock:
        t0 = time.perf_counter()
        raw = await request.body()
        header_len = int.from_bytes(raw[:4], "little")
        header = msgpack.unpackb(raw[4 : 4 + header_len], raw=False)
        buf = raw[4 + header_len :]

        loss_fn = header["loss_fn"]
        loss_fn_config = header.get("loss_fn_config")
        datum_descs = header["datums"]

        datums = []
        offset = 0
        for desc in datum_descs:
            tokens_len = desc["tokens_len"]
            tokens = np.frombuffer(buf[offset : offset + tokens_len * 4], dtype=np.int32).tolist()
            offset += tokens_len * 4
            lfi = {}
            for tensor_desc in desc["tensors"]:
                name = tensor_desc["name"]
                dtype = np.dtype(tensor_desc["dtype"])
                shape = tuple(tensor_desc["shape"])
                nbytes = int(np.prod(shape)) * dtype.itemsize
                arr = np.frombuffer(buf[offset : offset + nbytes], dtype=dtype).reshape(shape)
                lfi[name] = {"dtype": str(arr.dtype), "shape": list(arr.shape), "data": arr}
                offset += nbytes
            datums.append({"model_input": tokens, "loss_fn_inputs": lfi})

        payload = {
            "loss_fn": loss_fn,
            "loss_fn_config": loss_fn_config,
            "datums": datums,
        }
        t_deser = time.perf_counter()

        if len(datums) % STATE.world_size != 0:
            raise HTTPException(
                400,
                f"len(datums)={len(datums)} not divisible by world_size={STATE.world_size}",
            )

        result = _submit(Command(kind="forward_backward", payload=payload))
        t_compute = time.perf_counter()

        resp_header = {"metrics": result["metrics"], "outputs": []}
        resp_buffers = []
        for out in result["loss_fn_outputs"]:
            out_desc = {}
            for k, v in out.items():
                arr = np.asarray(v["data"]).astype(np.dtype(v["dtype"])).reshape(v["shape"])
                out_desc[k] = {"dtype": str(arr.dtype), "shape": list(arr.shape), "nbytes": arr.nbytes}
                resp_buffers.append(arr.tobytes())
            resp_header["outputs"].append(out_desc)

        resp_header_bytes = msgpack.packb(resp_header, use_bin_type=True)
        parts = [len(resp_header_bytes).to_bytes(4, "little"), resp_header_bytes] + resp_buffers
        t_ser = time.perf_counter()
        log.info(
            "forward_backward_bin timing: deser=%.2fs compute=%.2fs serialize=%.2fs total=%.2fs",
            t_deser - t0, t_compute - t_deser, t_ser - t_compute, t_ser - t0,
        )
        return Response(content=b"".join(parts), media_type="application/octet-stream")


@app.post("/optim_step", response_model=OptimStepResponse)
def optim_step(params: AdamParams) -> OptimStepResponse:
    with STATE.request_lock:
        result = _submit(Command(kind="optim_step", payload=params.model_dump()))
        return OptimStepResponse(metrics=result["metrics"])


@app.post("/save_state", response_model=SaveResponse)
def save_state(req: SaveRequest) -> SaveResponse:
    with STATE.request_lock:
        result = _submit(Command(kind="save_state", payload={"name": req.name}))
        return SaveResponse(path=result["path"])


@app.post("/save_weights_for_sampler", response_model=SaveResponse)
def save_weights_for_sampler(req: SaveRequest) -> SaveResponse:
    with STATE.request_lock:
        result = _submit(Command(kind="save_sampler", payload={"name": req.name}))
        return SaveResponse(path=result["path"])


@app.post("/shutdown", response_model=OkResponse)
def shutdown() -> OkResponse:
    with STATE.request_lock:
        _submit(Command(kind="shutdown"))
        return OkResponse()


# ----------------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model id or local path")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        default="auto",
    )
    parser.add_argument("--from-state", default=None, help="resume from a saved state dir")
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("TINTHER_CACHE_DIR", "/tmp/tinther"),
        help="directory for state/ and sampler/ checkpoints",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    setup_distributed()
    STATE.precision = args.precision
    STATE.model_name = args.model
    STATE.lora_rank = args.lora_rank
    STATE.max_grad_norm = args.max_grad_norm
    STATE.cache_dir = Path(args.cache_dir)
    STATE.cache_dir.mkdir(parents=True, exist_ok=True)

    build_model(
        model_name=args.model,
        precision=args.precision,
        lora_rank=args.lora_rank,
        attn_impl=args.attn_implementation,
        from_state=args.from_state,
    )

    log.info(
        "rank=%d world_size=%d precision=%s lora_rank=%d cache_dir=%s",
        STATE.rank,
        STATE.world_size,
        STATE.precision,
        STATE.lora_rank,
        STATE.cache_dir,
    )

    if STATE.rank == 0:
        STATE.cmd_queue = queue.Queue()
        config = uvicorn.Config(app, host=args.host, port=args.port, log_level=args.log_level)
        server = uvicorn.Server(config)
        threading.Thread(target=server.run, daemon=True).start()
        log.info("FastAPI listening on %s:%d", args.host, args.port)

    try:
        run_worker_loop()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
