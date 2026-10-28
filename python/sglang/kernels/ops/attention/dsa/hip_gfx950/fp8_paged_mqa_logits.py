"""In-tree HIP decode indexer logits for GLM-5.3-Flash on gfx950.

JIT build + load of ``kernels/jit/csrc/dsa_gfx950/fp8_paged_mqa_logits.cuh`` on
the first gated call. Same contract as aiter's ``deepgemm_fp8_paged_mqa_logits``
over the pooled index cache ``[pages, 64, 1, 132]`` (16x16 preshuffled keys
then fp32 scales; block_size 1 reads a plain cache):

    out[m, p] = sum_h relu(q[m,h,:] . k[p,:]) * weights[m,h] * kv_scale[p]

Only the causal window of each row is written: p <= context_lens[b] - next_n
+ n for [B] lengths, p < context_lens[b, n] for [B, next_n] ones; the rest of
``out`` is left as is (the k-pool top-k is range aware). n_heads=32,
head_dim=128, block_size 1 / 64 only.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

NUM_HEADS = 32
HEAD_DIM = 128
SUPPORTED_KV_BLOCK_SIZES = (1, 64)


@cache_once
def module() -> Module:
    # -fno-honor-nans: relu = single v_max. The device code leaves a few
    # unused constants.
    return load_jit(
        "dsa_gfx950_fp8_paged_mqa_logits",
        cuda_files=["dsa_gfx950/fp8_paged_mqa_logits.cuh"],
        cuda_wrappers=[("fp8_paged_mqa_logits", "Fp8PagedMqaLogitsKernel::run")],
        extra_cuda_cflags=[
            "-O3",
            "-fno-honor-nans",
            "-Wno-unused-variable",
            "-Wno-unused-result",
        ],
    )


@cache_once
def module_or_none() -> Optional[Module]:
    try:
        return module()
    except Exception as e:  # noqa: BLE001 - a build failure must never be fatal
        logger.warning(
            "gfx950 HIP fp8_paged_mqa_logits unavailable, build failed: %s", e
        )
        return None


def supported_hardware() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    return str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0] == "gfx950"


def is_supported(num_heads: int, head_dim: int, kv_block_size: int = 1) -> bool:
    return (
        supported_hardware()
        and num_heads == NUM_HEADS
        and head_dim == HEAD_DIM
        and kv_block_size in SUPPORTED_KV_BLOCK_SIZES
    )


def fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,  # [B, next_n, 32, 128] float8_e4m3fn
    kv_cache_fp8: torch.Tensor,  # [pages, block_size, 1, 132] uint8 / fp8
    weights: torch.Tensor,  # [B * next_n, 32] fp32
    context_lens: torch.Tensor,  # [B] or [B, next_n] int32
    block_tables: torch.Tensor,  # [B, max_blocks] int32
    max_model_len: int,
    *,
    out: torch.Tensor | None = None,
    chunk_k: int = 0,
    split_kv: int = 0,
    num_warps: int = 0,
    total_cu_count: int = 0,
    rows_per_block: int = 0,
) -> torch.Tensor:
    """Tunables <= 0 select the host heuristics (fitted on MI355X)."""
    if out is None:
        out = torch.empty(
            (q_fp8.shape[0] * q_fp8.shape[1], max_model_len),
            dtype=torch.float32,
            device=q_fp8.device,
        )
    module().fp8_paged_mqa_logits(
        q_fp8,
        kv_cache_fp8,
        weights,
        context_lens,
        block_tables,
        out,
        max_model_len,
        chunk_k,
        split_kv,
        num_warps,
        total_cu_count,
        rows_per_block,
    )
    return out
