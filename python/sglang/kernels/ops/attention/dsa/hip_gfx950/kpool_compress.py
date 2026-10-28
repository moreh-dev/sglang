"""In-tree HIP writers for the GLM-5.3-Flash k-pool index cache on gfx950.

JIT build + load of ``kernels/jit/csrc/dsa_gfx950/kpool_compress.cuh`` on the
first gated call (never in a timed region or a graph capture). The two entry
points mirror the Triton writers in ``srt/layers/attention/dsa/kpool_fp8_index``
byte for byte and share their cache layout: ``[num_pages, page_size, 132]``
uint8 with the 16x16 preshuffled key tile followed by the fp32 scales.

``kpool_compress_insert``  prefill: K / gate read in place, one window of
                           ``pool_size`` tokens per lane group, only closing
                           pools (``slot_mapping >= 0``) are compressed.
``kpool_decode_update``    decode: each token's K / gate stashed into the
                           request's tail ring ``[req, 2, ring, 128]``; the
                           pool-closing token compresses from the ring.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

HEAD_DIM = 128
CACHE_ROW_BYTES = HEAD_DIM + 4


@cache_once
def module() -> Module:
    # --amdgpu-kernarg-preload-count: the decode kernel's first loads take
    # their arguments from preloaded user SGPRs (hipcc >= 7.2 on gfx950).
    return load_jit(
        "dsa_gfx950_kpool_compress",
        cuda_files=["dsa_gfx950/kpool_compress.cuh"],
        cuda_wrappers=[
            ("kpool_compress_insert", "KpoolCompressKernel::insert"),
            ("kpool_decode_update", "KpoolCompressKernel::decode_update"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "-mllvm",
            "--amdgpu-kernarg-preload-count=32",
            "-Wno-unused-result",
        ],
    )


@cache_once
def module_or_none() -> Optional[Module]:
    """Build once; on failure log and return None so the gate can fall back."""
    try:
        return module()
    except Exception as e:  # noqa: BLE001 - a build failure must never be fatal
        logger.warning("gfx950 HIP kpool writers unavailable, build failed: %s", e)
        return None


def supported_hardware() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    arch = str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0]
    return arch in ("gfx950", "gfx942")


def kpool_pick_lanes(n_tokens: int, pool_size: int) -> int:
    """Lanes per pool the insert kernel picks for ``n_tokens`` (``lanes_per_pool=0``)."""
    n_pools = n_tokens // pool_size
    if n_pools <= 8192:
        return 64
    if n_pools <= 65536:
        return 32
    return 16


def kpool_compress_insert(
    kv_cache: torch.Tensor,  # [num_pages, page_size, 132] uint8
    k: torch.Tensor,  # [n_tokens, 128] bf16, rows may be strided
    gate_score: torch.Tensor,  # [n_tokens, 128] bf16
    ape: torch.Tensor,  # [pool_size, 128] fp32
    slot_mapping: torch.Tensor,  # [n_tokens] int32 / int64, >= 0 on closing tokens
    pool_size: int,
    round_scale: bool = True,
    lanes_per_pool: int = 0,  # 0: auto, else 16 / 32 / 64
) -> None:
    module().kpool_compress_insert(
        kv_cache, k, gate_score, ape, slot_mapping, pool_size, round_scale, lanes_per_pool
    )


def kpool_decode_update(
    kv_cache: torch.Tensor,  # [num_pages, page_size, 132] uint8
    tail_kv_cache: torch.Tensor,  # [req, 2, ring, 128] bf16, ring a power of two
    tail_slot_mapping: torch.Tensor,  # [num_requests, next_n] int32 / int64
    key: torch.Tensor,  # [num_requests, next_n, 128] bf16
    slot_score: torch.Tensor,  # [num_requests, next_n, 128] bf16
    ape: torch.Tensor,  # [pool_size, 128] fp32
    slot_mapping: torch.Tensor,  # [num_requests, next_n], dtype of tail_slot_mapping
    positions: torch.Tensor,  # [num_requests, next_n], dtype of tail_slot_mapping
    pool_size: int,
    round_scale: bool = True,
) -> None:
    module().kpool_decode_update(
        kv_cache,
        tail_kv_cache,
        tail_slot_mapping,
        key,
        slot_score,
        ape,
        slot_mapping,
        positions,
        pool_size,
        round_scale,
    )
