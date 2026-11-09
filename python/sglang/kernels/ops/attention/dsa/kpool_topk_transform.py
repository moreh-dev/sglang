from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

SUPPORTED_GROUP_TOPK = (128, 160, 192, 224, 256, 512)


@cache_once
def _jit_kpool_topk_transform_module(group_topk: int) -> Module:
    assert group_topk in SUPPORTED_GROUP_TOPK, (
        "fast_kpool_topk_transform supports pool-level topk "
        f"{SUPPORTED_GROUP_TOPK}, got {group_topk}"
    )
    return load_jit(
        f"kpool_topk_transform_{group_topk}",
        cuda_files=["dsa/kpool_topk_transform.cuh"],
        cuda_wrappers=[
            ("kpool_topk_transform", "KpoolTopKTransformKernel::transform"),
            ("kpool_groups_transform", "KpoolTopKTransformKernel::transform_groups"),
        ],
        extra_cuda_cflags=[f"-DSGL_GROUP_TOPK={group_topk}"],
    )


def _check_transform_args(
    *,
    rows: int,
    kpool: int,
    topk: int,
    page_table: Optional[torch.Tensor],
    topk_indices_offset: Optional[torch.Tensor],
    seq_lens: Optional[torch.Tensor],
    page_table_row_index: Optional[torch.Tensor],
) -> int:
    assert topk % kpool == 0
    group_topk = topk // kpool
    assert group_topk in SUPPORTED_GROUP_TOPK, (
        "fast_kpool_topk_transform supports pool-level topk "
        f"{SUPPORTED_GROUP_TOPK}, got {group_topk}"
    )
    assert page_table is None or topk_indices_offset is None
    assert page_table_row_index is None or page_table is not None
    if seq_lens is not None:
        assert seq_lens.dim() == 1
        assert seq_lens.shape[0] == rows
    if page_table_row_index is not None:
        assert page_table_row_index.dim() == 1
        assert page_table_row_index.shape[0] == rows
    return group_topk


def fast_kpool_topk_transform_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    kpool: int,
    topk: int,
    page_table: Optional[torch.Tensor] = None,
    topk_indices_offset: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    page_table_row_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert score.dim() == 2
    group_topk = _check_transform_args(
        rows=score.shape[0],
        kpool=kpool,
        topk=topk,
        page_table=page_table,
        topk_indices_offset=topk_indices_offset,
        seq_lens=seq_lens,
        page_table_row_index=page_table_row_index,
    )

    out_cols = topk + (kpool - 1 if seq_lens is not None else 0)
    dst_token_indices = score.new_empty((score.shape[0], out_cols), dtype=torch.int32)

    module = _jit_kpool_topk_transform_module(group_topk)
    module.kpool_topk_transform(
        score,
        lengths,
        dst_token_indices,
        kpool,
        page_table,
        topk_indices_offset,
        row_starts,
        seq_lens,
        page_table_row_index,
    )
    return dst_token_indices


def kpool_transform_selected_groups(
    groups: torch.Tensor,
    lengths: torch.Tensor,
    kpool: int,
    topk: int,
    page_table: Optional[torch.Tensor] = None,
    topk_indices_offset: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    page_table_row_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``fast_kpool_topk_transform_fused`` with the pool ranking done by the caller.

    ``groups`` is ``[rows, topk // kpool]`` int32: row ``r``'s selected pool ids,
    relative to its row start and in any order. It is read only for rows longer
    than ``topk // kpool`` pools; shorter rows take every pool, as in the radix path.
    """
    assert groups.dim() == 2
    group_topk = _check_transform_args(
        rows=groups.shape[0],
        kpool=kpool,
        topk=topk,
        page_table=page_table,
        topk_indices_offset=topk_indices_offset,
        seq_lens=seq_lens,
        page_table_row_index=page_table_row_index,
    )
    assert groups.shape[1] == group_topk

    out_cols = topk + (kpool - 1 if seq_lens is not None else 0)
    dst_token_indices = groups.new_empty((groups.shape[0], out_cols), dtype=torch.int32)

    module = _jit_kpool_topk_transform_module(group_topk)
    module.kpool_groups_transform(
        groups,
        lengths,
        dst_token_indices,
        kpool,
        page_table,
        topk_indices_offset,
        seq_lens,
        page_table_row_index,
    )
    return dst_token_indices
