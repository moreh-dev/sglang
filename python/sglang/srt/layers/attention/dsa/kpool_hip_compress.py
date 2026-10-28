"""In-tree HIP writers for the pooled DSA index cache (GLM-5.3-Flash on gfx950).

Enabled with ``SGLANG_DSA_KPOOL_HIP_COMPRESS=1``. The kernels
(``kernels/jit/csrc/dsa_gfx950/kpool_compress.cuh``) are byte-exact with the
Triton writers in ``kpool_fp8_index.py`` and share their cache layout:
``[num_index_pages, index_page_size, 132]`` uint8 with the 16x16 preshuffled
key tile followed by the fp32 scales. They take the compression destination
per token (``slot_mapping``), so this module builds that from the sglang
metadata; the decode side does it in one Triton launch that is CUDA-graph safe.
"""

from __future__ import annotations

import functools
import logging

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

INDEX_HEAD_DIM = 128
_CACHE_ROW_BYTES = INDEX_HEAD_DIM + 4


@functools.lru_cache(maxsize=1)
def kpool_hip_compress_enabled() -> bool:
    if not is_hip() or not envs.SGLANG_DSA_KPOOL_HIP_COMPRESS.get():
        return False
    from sglang.kernels.ops.attention.dsa.hip_gfx950 import kpool_compress

    if not kpool_compress.supported_hardware():
        logger.warning(
            "SGLANG_DSA_KPOOL_HIP_COMPRESS=1 needs gfx950 / gfx942; "
            "using the Triton kpool writers"
        )
        return False
    # Fail loudly: a silent fallback would make a benchmark of this path a lie.
    kpool_compress.module()
    logger.info("DSA kpool: index cache written by the in-tree HIP kpool kernels")
    return True


def kpool_hip_tail_ring(index_kpool: int, tail_extra_slots: int) -> int:
    """The HIP tail ring must be a power of two and a whole number of pools;
    sglang's Triton writers index the ring with ``pos % tail_width`` just like
    the kernel, so both agree as long as every writer sees the same width."""
    width = index_kpool + tail_extra_slots
    assert width % index_kpool == 0 and width & (width - 1) == 0, (
        f"SGLANG_DSA_KPOOL_HIP_COMPRESS needs a tail ring that is a power of two "
        f"and a multiple of index_kpool={index_kpool}, got {width} "
        f"(tail_extra_slots={tail_extra_slots})"
    )
    return width


def _kv_cache_view(buf: torch.Tensor, index_page_size: int) -> torch.Tensor:
    assert buf.dtype == torch.uint8
    assert buf.shape[1] == index_page_size * _CACHE_ROW_BYTES, (
        f"index buffer row has {buf.shape[1]} bytes, expected "
        f"{index_page_size} x {_CACHE_ROW_BYTES}"
    )
    return buf.view(buf.shape[0], index_page_size, _CACHE_ROW_BYTES)


@triton.jit
def _kpool_hip_decode_meta_kernel(
    block_tables_ptr,
    req_pool_indices_ptr,
    positions_ptr,
    seq_lens_ptr,
    out_cache_loc_ptr,
    slot_map_ptr,
    tail_slot_ptr,
    pos_out_ptr,
    batch,
    bt_stride_0,
    bt_stride_1,
    block_table_cols,
    req_pool_size,
    POOL_SIZE: tl.constexpr,
    INDEX_PAGE_SIZE: tl.constexpr,
    RING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = rows < batch
    req = tl.load(req_pool_indices_ptr + rows, mask=m, other=-1).to(tl.int64)
    pos = tl.load(positions_ptr + rows, mask=m, other=-1).to(tl.int64)
    seq_len = tl.load(seq_lens_ptr + rows, mask=m, other=0).to(tl.int64)
    cache_loc = tl.load(out_cache_loc_ptr + rows, mask=m, other=0).to(tl.int64)

    # Same admission as _kpool_decode_update_and_maybe_write_cache_kernel.
    req_valid = (req >= 0) & (req < req_pool_size)
    pos_valid = m & req_valid & (cache_loc != 0) & (pos >= 0) & (pos < seq_len)
    safe_pos = tl.maximum(pos, 0)
    pool_id = safe_pos // POOL_SIZE
    closing = pos_valid & (safe_pos % POOL_SIZE == POOL_SIZE - 1)

    token_page_row = (pool_id // INDEX_PAGE_SIZE) * POOL_SIZE
    token_page_row = tl.minimum(tl.maximum(token_page_row, 0), block_table_cols - 1)
    packed_page = tl.load(
        block_tables_ptr + rows.to(tl.int64) * bt_stride_0 + token_page_row * bt_stride_1,
        mask=m,
        other=0,
    ).to(tl.int64)
    loc = (packed_page // POOL_SIZE) * INDEX_PAGE_SIZE + pool_id % INDEX_PAGE_SIZE

    tl.store(slot_map_ptr + rows, tl.where(closing, loc, -1).to(tl.int32), mask=m)
    tl.store(tail_slot_ptr + rows, tl.where(pos_valid, req * RING, -1).to(tl.int32), mask=m)
    tl.store(pos_out_ptr + rows, tl.where(pos_valid, safe_pos, -1).to(tl.int32), mask=m)


def hip_kpool_decode_update(
    *,
    pool,
    buf: torch.Tensor,
    tail_kv: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    block_tables: torch.Tensor,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    round_scale: bool,
) -> None:
    """One decode token per request: stash into the tail ring, compress on the
    pool-closing token. Two launches (metadata + kernel), both graph-capturable."""
    from sglang.kernels.ops.attention.dsa.hip_gfx950.kpool_compress import (
        kpool_decode_update,
    )

    batch = key.shape[0]
    if batch == 0:
        return
    assert key.shape[1] == INDEX_HEAD_DIM and key.dtype == torch.bfloat16
    assert slot_score.shape == key.shape and slot_score.dtype == torch.bfloat16
    assert tail_kv.dim() == 4 and tail_kv.shape[1] == 2
    assert block_tables.ndim == 2 and block_tables.shape[0] >= batch
    ring = tail_kv.shape[2]

    key = key.contiguous()
    slot_score = slot_score.contiguous()
    meta = torch.empty((3, batch), dtype=torch.int32, device=key.device)
    BLOCK = 128
    _kpool_hip_decode_meta_kernel[(triton.cdiv(batch, BLOCK),)](
        block_tables,
        req_pool_indices,
        positions,
        seq_lens,
        out_cache_loc,
        meta[0],
        meta[1],
        meta[2],
        batch,
        block_tables.stride(0),
        block_tables.stride(1),
        block_tables.shape[1],
        tail_kv.shape[0],
        POOL_SIZE=pool.index_kpool,
        INDEX_PAGE_SIZE=pool.index_page_size,
        RING=ring,
        BLOCK=BLOCK,
    )
    kpool_decode_update(
        _kv_cache_view(buf, pool.index_page_size),
        tail_kv,
        meta[1].view(batch, 1),
        key.view(batch, 1, INDEX_HEAD_DIM),
        slot_score.view(batch, 1, INDEX_HEAD_DIM),
        ape,
        meta[0].view(batch, 1),
        meta[2].view(batch, 1),
        pool.index_kpool,
        round_scale,
    )


def hip_kpool_insert_from_plan(
    *,
    pool,
    buf: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    chunk_src: torch.Tensor,
    n_from_tail: torch.Tensor,
    write_loc: torch.Tensor,
    ape: torch.Tensor,
    round_scale: bool,
) -> None:
    """Prefill: compress every pool of the extend plan whose rows all lie in
    this chunk. Pools that start in the tail ring are left to the Triton
    assemble kernel (see ``_compress_write_extend``)."""
    from sglang.kernels.ops.attention.dsa.hip_gfx950.kpool_compress import (
        kpool_compress_insert,
    )

    kpool = pool.index_kpool
    n_tokens = chunk_k.shape[0]
    if n_tokens < kpool or chunk_src.shape[0] == 0:
        return
    chunk_k = chunk_k.contiguous()
    chunk_score = chunk_score.contiguous()
    assert chunk_k.shape == (n_tokens, INDEX_HEAD_DIM) and chunk_k.dtype == torch.bfloat16
    assert chunk_score.shape == chunk_k.shape and chunk_score.dtype == torch.bfloat16

    in_chunk = n_from_tail == 0
    chunk_src = chunk_src.to(torch.int64)
    # A pool that starts in the tail has no closing token to mark here; park
    # its row on the request's first token, which no in-chunk pool closes on.
    closing_tok = torch.where(in_chunk, chunk_src + (kpool - 1), chunk_src)
    loc = torch.where(in_chunk, write_loc.to(torch.int64), -1)
    slot_mapping = torch.full((n_tokens,), -1, dtype=torch.int64, device=chunk_k.device)
    slot_mapping[closing_tok] = loc
    kpool_compress_insert(
        _kv_cache_view(buf, pool.index_page_size),
        chunk_k,
        chunk_score,
        ape,
        slot_mapping,
        kpool,
        round_scale,
        0,
    )
