"""Gate for the in-tree HIP prefill indexer logits + top-k of the GLM-5.3-Flash
k-pool indexer on gfx950 (kernels/ops/attention/dsa/hip_gfx950/). Opt-in by env
var; a decline is logged once with its reason and the logits + top-k path stays."""

from __future__ import annotations

import functools
import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def hip_fp8_mqa_topk_enabled() -> bool:
    """SGLANG_DSA_HIP_FP8_MQA_TOPK=1 on gfx950 with a kernel that builds."""
    if not is_hip() or not envs.SGLANG_DSA_HIP_FP8_MQA_TOPK.get():
        return False
    from sglang.kernels.ops.attention.dsa.hip_gfx950 import fp8_mqa_topk

    if not fp8_mqa_topk.supported_hardware():
        logger.warning(
            "SGLANG_DSA_HIP_FP8_MQA_TOPK=1 needs gfx950; using fp8_mqa_logits "
            "+ the k-pool top-k"
        )
        return False
    if fp8_mqa_topk.module_or_none() is None:
        logger.warning(
            "SGLANG_DSA_HIP_FP8_MQA_TOPK=1 but the kernel failed to build; "
            "using fp8_mqa_logits + the k-pool top-k"
        )
        return False
    logger.info(
        "DSA kpool: prefill indexer logits + top-k by the in-tree HIP "
        "fp8_mqa_topk kernel (rocm-aiter PR #18)"
    )
    return True


def hip_fp8_mqa_topk(
    *,
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    topk: int,
) -> torch.Tensor | None:
    """Each row's top-``topk`` key ids in [starts, ends), relative to starts, when
    this call fits the kernel (n_heads 32, head_dim 128, topk 512, eager), else
    None so the caller computes logits. The gathered K buffer is uint8 bytes of
    fp8 e4m3fn; the kernel reads it as fp8."""
    from sglang.kernels.ops.attention.dsa.hip_gfx950 import fp8_mqa_topk

    if (
        q_fp8.ndim != 3
        or not fp8_mqa_topk.is_supported(q_fp8.shape[1], q_fp8.shape[2])
        or topk != fp8_mqa_topk.TOPK
        or k_fp8.shape[0] > fp8_mqa_topk.MAX_KEYS
        # Grids and workspaces are sized on the host per call.
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    fp8 = torch.float8_e4m3fn
    if q_fp8.dtype != fp8:
        q_fp8 = q_fp8.view(fp8)
    if k_fp8.dtype != fp8:
        k_fp8 = k_fp8.view(fp8)
    return fp8_mqa_topk.fp8_mqa_topk(
        q_fp8.contiguous(),
        k_fp8.contiguous(),
        k_scale.reshape(-1).contiguous(),
        weights.contiguous(),
        starts.to(torch.int32).contiguous(),
        ends.to(torch.int32).contiguous(),
    )
