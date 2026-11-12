"""In-tree HIP prefill indexer logits + top-k for GLM-5.3-Flash on gfx950.

JIT build + load of ``kernels/jit/csrc/dsa_gfx950/fp8_mqa_topk.cuh`` on the first
gated call. Each row's top ``TOPK`` key ids of

    logits[m, n] = sum_h relu(q[m,h,:] . k[n,:]) * weights[m,h] * kv_scale[n]

for n in [ks[m], ke[m]), without storing the fp32 [rows, keys] logits; the .cuh
describes the pipeline. The result is the exact top-k of the logits its scan
computes (ties broken arbitrarily). n_heads=32, head_dim=128, TOPK=512 only;
``is_supported`` gates on the shape. Device code: moreh-dev/rocm-aiter#18.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

# The configuration the kernels were validated in; fp8_mqa_topk.cuh rejects others.
TOPK = 512
NUM_HEADS = 32
HEAD_DIM = 128
SAMPLE_STRIDE = 32
# Candidate slots per (split, row); a row overflowing them takes the exact fallback.
CANDIDATE_CAP = 4096
FALLBACK_SCRATCH_ROWS = 128
# The kernels address the keys through 32-bit buffer offsets.
MAX_KEYS = (2**31 - 1) // HEAD_DIM
# The rocm-aiter launcher's flags: the kernels' hand-placed waits were checked on
# the ISA these produce, so they are part of the kernel, not a tuning choice.
_CUDA_CFLAGS = [
    "-O3",
    "-fno-honor-nans",
    "-fno-slp-vectorize",
    "-mllvm",
    "-disable-machine-sink",
    "-Wno-unused-result",
]


@cache_once
def module() -> Module:
    return load_jit(
        "dsa_gfx950_fp8_mqa_topk",
        cuda_files=["dsa_gfx950/fp8_mqa_topk.cuh"],
        cuda_wrappers=[
            ("plan", "Fp8MqaTopkKernel::plan"),
            ("run", "Fp8MqaTopkKernel::run"),
        ],
        extra_cuda_cflags=_CUDA_CFLAGS,
    )


@cache_once
def module_or_none() -> Optional[Module]:
    try:
        return module()
    except Exception as e:  # noqa: BLE001 - a build failure must never be fatal
        logger.warning("gfx950 HIP fp8_mqa_topk unavailable, build failed: %s", e)
        return None


def supported_hardware() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    return (
        str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0] == "gfx950"
    )


def is_supported(num_heads: int, head_dim: int) -> bool:
    return supported_hardware() and num_heads == NUM_HEADS and head_dim == HEAD_DIM


# Chunked prefill changes rows x keys on every forward, while all indexer layers
# of one forward share it; the bound only caps the cache's memory.
@functools.lru_cache(maxsize=256)
def _plan(rows: int, keys: int) -> int:
    """Packed scan grid (split | reverse << 8) from the kernels' makespan model."""
    return int(module().plan(rows, keys))


def fp8_mqa_topk(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    force_fallback: bool = False,
    fallback_row_count: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Each row's top-``TOPK`` key ids in [starts, ends), relative to starts.

    q_fp8 is [rows, 32, 128] and k_fp8 [keys, 128], both fp8 e4m3fn; k_scale is
    [keys] and weights [rows, 32] fp32; starts / ends are [rows] int32. A row with
    fewer valid keys returns all of them, then -1. Returns [rows, TOPK] int32.

    Test hooks: force_fallback sends every row through the exact recompute, and
    fallback_row_count (a zeroed [1] int32) counts the rows that took it.
    """
    rows, keys = q_fp8.shape[0], k_fp8.shape[0]
    device = q_fp8.device
    out = torch.empty((rows, TOPK), dtype=torch.int32, device=device)
    if rows == 0:
        return out
    if keys == 0:
        return out.fill_(-1)

    stride = SAMPLE_STRIDE
    k_scale = k_scale.reshape(-1)
    starts = starts.to(torch.int32)
    ends = ends.to(torch.int32)
    sample_k = k_fp8[::stride].contiguous()
    sample_keys = sample_k.shape[0]
    scan_plan = _plan(rows, keys)
    split = scan_plan & 0xFF
    rows_padded = rows + (rows & 1)

    module().run(
        q_fp8,
        k_fp8,
        k_scale,
        weights,
        starts,
        ends,
        sample_k,
        k_scale[::stride].contiguous(),
        (starts + (stride - 1)) // stride,
        (ends + (stride - 1)) // stride,
        torch.empty((rows, sample_keys), dtype=torch.float32, device=device),
        torch.empty((rows,), dtype=torch.float32, device=device),
        torch.empty(
            (split * rows_padded * CANDIDATE_CAP,), dtype=torch.int64, device=device
        ),
        torch.empty((split * rows,), dtype=torch.int32, device=device),
        torch.empty((rows,), dtype=torch.int32, device=device),
        # Flagged-row count; the tau kernel zeroes it.
        torch.empty((1,), dtype=torch.int32, device=device),
        torch.empty(
            (FALLBACK_SCRATCH_ROWS * keys,), dtype=torch.float32, device=device
        ),
        out,
        _plan(rows, sample_keys),
        scan_plan,
        stride,
        CANDIDATE_CAP,
        int(force_fallback),
        fallback_row_count,
    )
    return out
