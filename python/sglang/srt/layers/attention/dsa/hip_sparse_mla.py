"""Gate for the in-tree HIP FP8 sparse MLA attention core on gfx950
(kernels/ops/attention/dsa/hip_gfx950/sparse_mla_fp8.py), an opt-in
replacement for the tilelang DSA backend's ``tilelang_sparse_fwd`` on the
FP8 latent cache. Declines are logged once with their reason."""

from __future__ import annotations

import functools
import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

_DECLINED: set[str] = set()


def _decline(reason: str) -> None:
    if reason not in _DECLINED:
        _DECLINED.add(reason)
        logger.warning("HIP FP8 sparse MLA not used: %s; using tilelang", reason)


@functools.lru_cache(maxsize=1)
def hip_sparse_mla_fp8_enabled() -> bool:
    """SGLANG_DSA_HIP_SPARSE_MLA_FP8=1 on gfx950 with kernels that build."""
    if not is_hip() or not envs.SGLANG_DSA_HIP_SPARSE_MLA_FP8.get():
        return False
    from sglang.kernels.ops.attention.dsa.hip_gfx950 import sparse_mla_fp8

    if not sparse_mla_fp8.supported_hardware():
        logger.warning("SGLANG_DSA_HIP_SPARSE_MLA_FP8=1 needs gfx950; using tilelang")
        return False
    if sparse_mla_fp8.module_or_none() is None:
        logger.warning(
            "SGLANG_DSA_HIP_SPARSE_MLA_FP8=1 but the kernels failed to build; "
            "using tilelang"
        )
        return False
    logger.info(
        "DSA: sparse MLA attention by the in-tree HIP FP8 kernel (fused BF16 "
        "query conversion, BF16 output)"
    )
    return True


def hip_sparse_mla_fp8_fwd(
    *,
    q_all: torch.Tensor,  # [tokens, heads, head_dim] bf16 (or fp8)
    kv_cache: torch.Tensor,  # [slots, 1, head_dim] latent cache
    page_table_1: torch.Tensor,  # [tokens, K] int32 slot ids, -1 padded
    sm_scale: float,
    v_head_dim: int,
) -> torch.Tensor | None:
    """The HIP attention output [tokens, 64, 512] when this call fits the
    kernel (all 64 heads on this rank, 512-dim latent without RoPE, fp8 e4m3fn
    cache, <= 2176 ids per row, <= 16384 rows), else None so the caller keeps
    tilelang. The tilelang fp8 path casts q to fp8 without a scale, so the
    kernel runs with q_scale = kv_scale = 1 as well."""
    from sglang.kernels.ops.attention.dsa.hip_gfx950 import sparse_mla_fp8

    fp8 = torch.float8_e4m3fn
    tokens, heads, head_dim = q_all.shape
    if heads != sparse_mla_fp8.NUM_HEADS:
        _decline(f"{heads} query heads on this rank, the kernel needs 64 (attention DP)")
        return None
    if head_dim != sparse_mla_fp8.HEAD_DIM or v_head_dim != sparse_mla_fp8.HEAD_DIM:
        _decline(f"head_dim {head_dim} / v_head_dim {v_head_dim}, the kernel needs 512 / 512")
        return None
    if kv_cache.dtype != fp8:
        _decline(f"KV cache dtype {kv_cache.dtype}, the kernel reads float8_e4m3fn")
        return None
    if page_table_1.shape[-1] > sparse_mla_fp8.MAX_CAPACITY:
        _decline(f"{page_table_1.shape[-1]} ids per row, the kernel takes <= 2176")
        return None
    if tokens > sparse_mla_fp8.MAX_QUERIES:
        return None  # per-call shape; the next chunk may fit
    if q_all.dtype not in (torch.bfloat16, fp8):
        q_all = q_all.to(torch.bfloat16)
    out, _ = sparse_mla_fp8.sparse_mla_fp8_fwd(
        q_all,
        kv_cache.view(-1, head_dim),
        page_table_1.to(torch.int32).contiguous(),
        sm_scale,
    )
    return out
