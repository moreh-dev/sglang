"""Gates for the in-tree HIP indexer-logits kernels of the GLM-5.3-Flash k-pool
indexer on gfx950 (kernels/ops/attention/dsa/hip_gfx950/). Each is opt-in by
env var; a decline is logged once with its reason and the aiter path stays."""

from __future__ import annotations

import functools
import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def hip_fp8_mqa_logits_enabled() -> bool:
    """SGLANG_DSA_HIP_FP8_MQA_LOGITS=1 on gfx950 with a kernel that builds."""
    if not is_hip() or not envs.SGLANG_DSA_HIP_FP8_MQA_LOGITS.get():
        return False
    from sglang.kernels.ops.attention.dsa.hip_gfx950 import fp8_mqa_logits

    if not fp8_mqa_logits.supported_hardware():
        logger.warning(
            "SGLANG_DSA_HIP_FP8_MQA_LOGITS=1 needs gfx950; using aiter's Triton "
            "fp8_mqa_logits"
        )
        return False
    if fp8_mqa_logits.module_or_none() is None:
        logger.warning(
            "SGLANG_DSA_HIP_FP8_MQA_LOGITS=1 but the kernel failed to build; "
            "using aiter's Triton fp8_mqa_logits"
        )
        return False
    logger.info("DSA kpool: prefill indexer logits by the in-tree HIP fp8_mqa_logits kernel")
    return True


def hip_fp8_mqa_logits(
    *,
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor | None:
    """The HIP prefill logits when this shape fits it (n_heads 32, head_dim
    128), else None so the caller keeps its aiter path. The gathered K buffer
    is uint8 bytes of fp8 e4m3fn; the kernel reads it as fp8."""
    from sglang.kernels.ops.attention.dsa.hip_gfx950 import fp8_mqa_logits

    if q_fp8.ndim != 3 or not fp8_mqa_logits.is_supported(q_fp8.shape[1], q_fp8.shape[2]):
        return None
    fp8 = torch.float8_e4m3fn
    if q_fp8.dtype != fp8:
        q_fp8 = q_fp8.view(fp8)
    if k_fp8.dtype != fp8:
        k_fp8 = k_fp8.view(fp8)
    return fp8_mqa_logits.fp8_mqa_logits(
        q_fp8.contiguous(),
        k_fp8.contiguous(),
        k_scale.reshape(-1).contiguous(),
        weights.contiguous(),
        starts.to(torch.int32).contiguous(),
        ends.to(torch.int32).contiguous(),
        clean_logits=clean_logits,
    )


@functools.lru_cache(maxsize=1)
def hip_fp8_paged_mqa_logits_enabled() -> bool:
    """SGLANG_DSA_HIP_FP8_PAGED_MQA_LOGITS=1 on gfx950 with a kernel that builds."""
    if not is_hip() or not envs.SGLANG_DSA_HIP_FP8_PAGED_MQA_LOGITS.get():
        return False
    from sglang.kernels.ops.attention.dsa.hip_gfx950 import fp8_paged_mqa_logits

    if not fp8_paged_mqa_logits.supported_hardware():
        logger.warning(
            "SGLANG_DSA_HIP_FP8_PAGED_MQA_LOGITS=1 needs gfx950; using aiter's "
            "paged MQA logits"
        )
        return False
    if fp8_paged_mqa_logits.module_or_none() is None:
        logger.warning(
            "SGLANG_DSA_HIP_FP8_PAGED_MQA_LOGITS=1 but the kernel failed to build; "
            "using aiter's paged MQA logits"
        )
        return False
    logger.info(
        "DSA kpool: decode indexer logits by the in-tree HIP fp8_paged_mqa_logits "
        "kernel for batches of >= %d rows",
        envs.SGLANG_DSA_HIP_FP8_PAGED_MQA_LOGITS_MIN_ROWS.get(),
    )
    return True


def hip_fp8_paged_mqa_logits(
    *,
    q_fp8: torch.Tensor,  # [rows, 32, 128] fp8
    kv_cache_fp8: torch.Tensor,  # [pages, 64, 1, 132] uint8, preshuffled keys
    weights: torch.Tensor,  # [rows, 32] fp32
    seq_lens: torch.Tensor,  # [rows] int32, pooled context length per row
    block_tables: torch.Tensor,  # [rows, pages] int32, pooled page table
    max_seq_len: int,
) -> torch.Tensor | None:
    """The HIP decode logits [rows, max_seq_len] when this call fits the kernel
    (n_heads 32, head_dim 128, 64-token preshuffled pages, enough rows), else
    None so the caller keeps its aiter path. Each query row is its own request
    (next_n = 1), so the kernel's [batch] lengths are per row. Only the causal
    window is written, as on the aiter path."""
    from sglang.kernels.ops.attention.dsa.hip_gfx950 import fp8_paged_mqa_logits

    rows = q_fp8.shape[0]
    if (
        q_fp8.ndim != 3
        or rows < envs.SGLANG_DSA_HIP_FP8_PAGED_MQA_LOGITS_MIN_ROWS.get()
        or not fp8_paged_mqa_logits.is_supported(
            q_fp8.shape[1], q_fp8.shape[2], kv_cache_fp8.shape[1]
        )
        or kv_cache_fp8.shape[-1] != 132
    ):
        return None
    if q_fp8.dtype != torch.float8_e4m3fn:
        q_fp8 = q_fp8.view(torch.float8_e4m3fn)
    return fp8_paged_mqa_logits.fp8_paged_mqa_logits(
        q_fp8.contiguous().unsqueeze(1),
        kv_cache_fp8,
        weights.contiguous(),
        seq_lens.to(torch.int32).contiguous(),
        block_tables.to(torch.int32).contiguous(),
        max_seq_len,
    )
