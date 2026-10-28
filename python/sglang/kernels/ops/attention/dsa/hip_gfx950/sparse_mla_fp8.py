"""In-tree HIP FP8 sparse MLA attention core for GLM-5.3-Flash on gfx950.

JIT build + load of ``kernels/jit/csrc/dsa_gfx950/sparse_mla_fp8.cuh`` (four
kernel translation units plus the entry unit) on the first gated call. Same
contract as aiter's ``sparse_mla_fp8_fwd``: BF16 or FP8 ``q`` [Q, 64, 512]
(BF16 is encoded as FP8 inside the kernel, ``q / q_scale``), the flat FP8
latent pool ``kv`` [slots, 512], ``kv_indices`` [Q, K] physical slot ids with
negative / out-of-range ids masked (K <= 2176, Q <= 16384), BF16 output and an
optional natural-log LSE. Dispatch depends on shapes only, so a warmed call is
CUDA-graph capturable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

NUM_HEADS = 64
HEAD_DIM = 512
MAX_QUERIES = 16384
MAX_CAPACITY = 2176


@cache_once
def module() -> Module:
    # -fno-gpu-flush-denormals-to-zero keeps the gradual underflow of the FP8
    # encode that the kernel's numerical contract documents; the device code
    # leaves a few unused constants.
    return load_jit(
        "dsa_gfx950_sparse_mla_fp8",
        cuda_files=["dsa_gfx950/sparse_mla_fp8.cuh"],
        cuda_wrappers=[
            ("sparse_mla_fp8_fwd", "SparseMlaFp8Kernel::fwd"),
            ("sparse_mla_fp8_fwd_lse", "SparseMlaFp8Kernel::fwd_lse"),
        ],
        cuda_units=[[f"SGL_SPARSE_MLA_UNIT={unit}"] for unit in (1, 2, 3, 4)],
        extra_cuda_cflags=[
            "-O3",
            "-fno-gpu-flush-denormals-to-zero",
            "-Wno-unused-variable",
            "-Wno-unused-result",
        ],
    )


@cache_once
def module_or_none() -> Optional[Module]:
    try:
        return module()
    except Exception as e:  # noqa: BLE001 - a build failure must never be fatal
        logger.warning("gfx950 HIP sparse_mla_fp8 unavailable, build failed: %s", e)
        return None


def supported_hardware() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    return str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0] == "gfx950"


def split_count(queries: int) -> int:
    """KV splits of the kernel's dispatch for ``queries`` rows (sizes the
    workspace; the layout itself is private to the kernel)."""
    for limit, splits in ((6, 17), (12, 9), (21, 6), (23, 4), (42, 6), (64, 4), (128, 2)):
        if queries <= limit:
            return splits
    return 1


def workspace_rows(queries: int) -> int:
    splits = split_count(queries)
    return queries * splits * 64 if splits > 1 else 0


def workspace(queries: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Opaque split workspace (BF16 partial O, FP32 partial LSE); needs no
    initialization. Reuse one pair per query count."""
    rows = workspace_rows(queries)
    return (
        torch.empty((rows * HEAD_DIM,), dtype=torch.bfloat16, device=device),
        torch.empty((rows,), dtype=torch.float32, device=device),
    )


def sparse_mla_fp8_fwd(
    q: torch.Tensor,  # [Q, 64, 512] bf16 or float8_e4m3fn
    kv_buffer: torch.Tensor,  # float8_e4m3fn [slots, 512], [pages, page, 512] or [slots, 1, 1, 512]
    kv_indices: torch.Tensor,  # int32 [Q, K] or [Q, 1, K]
    softmax_scale: float,
    *,
    q_scale: float = 1.0,
    kv_scale: float = 1.0,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    return_lse: bool = False,
    workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if kv_buffer.ndim == 4 and kv_buffer.shape[1:3] == (1, 1):
        kv_buffer = kv_buffer.view(kv_buffer.shape[0], kv_buffer.shape[-1])
    elif kv_buffer.ndim == 3:
        kv_buffer = kv_buffer.view(-1, kv_buffer.shape[-1])
    if kv_indices.ndim == 3 and kv_indices.shape[1] == 1:
        kv_indices = kv_indices.view(kv_indices.shape[0], kv_indices.shape[2])
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    if lse is not None and not return_lse:
        raise ValueError("supplying lse requires return_lse=True")
    if return_lse and lse is None:
        lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    if workspace is None:
        workspace = globals()["workspace"](q.shape[0], q.device)
    partial_o, partial_lse = workspace
    mod = module()
    if lse is None:
        mod.sparse_mla_fp8_fwd(
            q, kv_buffer, kv_indices, out, partial_o, partial_lse,
            float(softmax_scale), float(q_scale), float(kv_scale),
        )
    else:
        mod.sparse_mla_fp8_fwd_lse(
            q, kv_buffer, kv_indices, out, lse, partial_o, partial_lse,
            float(softmax_scale), float(q_scale), float(kv_scale),
        )
    return out, lse
