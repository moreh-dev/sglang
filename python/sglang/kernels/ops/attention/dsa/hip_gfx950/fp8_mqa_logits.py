"""In-tree HIP prefill indexer logits for GLM-5.3-Flash on gfx950.

JIT build + load of ``kernels/jit/csrc/dsa_gfx950/fp8_mqa_logits.cuh`` on the
first gated call. Same contract as aiter's Triton ``fp8_mqa_logits``:

    out[m, n] = sum_h relu(q[m,h,:] . k[n,:]) * weights[m,h] * kv_scale[n]

for n in [ks[m], ke[m]); -inf elsewhere when ``clean_logits``. n_heads=32,
head_dim=128 only (the GLM-5.x indexer shape); ``is_supported`` gates on that.
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


@cache_once
def module() -> Module:
    # -fno-honor-nans: relu = single v_max; -fno-slp-vectorize: keep the
    # epilogue scalar (packed v_pk_fma cannot co-issue with MFMA);
    # -disable-machine-sink: the last row's MFMA pair must not sink past the
    # loop exit branch. The device code leaves a few unused constants.
    return load_jit(
        "dsa_gfx950_fp8_mqa_logits",
        cuda_files=["dsa_gfx950/fp8_mqa_logits.cuh"],
        cuda_wrappers=[("fp8_mqa_logits", "Fp8MqaLogitsKernel::run")],
        extra_cuda_cflags=[
            "-O3",
            "-fno-honor-nans",
            "-fno-slp-vectorize",
            "-mllvm",
            "-disable-machine-sink",
            "-Wno-unused-variable",
            "-Wno-unused-result",
        ],
    )


@cache_once
def module_or_none() -> Optional[Module]:
    try:
        return module()
    except Exception as e:  # noqa: BLE001 - a build failure must never be fatal
        logger.warning("gfx950 HIP fp8_mqa_logits unavailable, build failed: %s", e)
        return None


def supported_hardware() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    return str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0] == "gfx950"


def is_supported(num_heads: int, head_dim: int) -> bool:
    return supported_hardware() and num_heads == NUM_HEADS and head_dim == HEAD_DIM


def fp8_mqa_logits(
    q_fp8: torch.Tensor,  # [M, 32, 128] float8_e4m3fn
    k_fp8: torch.Tensor,  # [N, 128] float8_e4m3fn
    kv_scale: torch.Tensor,  # [N] fp32
    weights: torch.Tensor,  # [M, 32] fp32
    cu_seqlen_ks: torch.Tensor,  # [M] int32
    cu_seqlen_ke: torch.Tensor,  # [M] int32
    *,
    clean_logits: bool = True,
    split_n: int = 0,
    num_warps: int = 0,
    total_cu_count: int = 0,
    reverse_rows: int = -1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(
            (q_fp8.shape[0], k_fp8.shape[0]), dtype=torch.float32, device=q_fp8.device
        )
    module().fp8_mqa_logits(
        q_fp8,
        k_fp8,
        kv_scale,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        out,
        split_n,
        num_warps,
        total_cu_count,
        clean_logits,
        reverse_rows,
    )
    return out
