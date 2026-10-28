"""In-tree HIP k-pool index-cache writers (kernels/jit/csrc/dsa_gfx950/kpool_compress.cuh)
on GLM-5.3-Flash geometry: byte-exact against the Triton writers of
kpool_fp8_index.py, and within one fp8 ulp of an fp32 torch reference.

Two levels. The kernel entry points (insert, decode_update) run against a
torch reference of the pool math and the preshuffled cache layout, and the
insert against the two-pass Triton writer byte for byte. The srt wrappers
(kpool_hip_compress.py) replay decode steps and extend plans through both the
Triton and the HIP path on identical state and compare cache and tail rings.
"""

import math
import random
import unittest

import torch

from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    kpool_assemble_softmax_rotate_write_cache,
    kpool_decode_update_and_maybe_write_cache,
    kpool_softmax_rotate_write_cache,
)
from sglang.srt.layers.attention.dsa.kpool_plan import (
    _append_compress_rows,
    _KPoolCpuPlan,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=300, suite="stage-b-test-1-gpu-small-amd-mi35x")

KPOOL = 4
PAGE = 256  # sglang forces 64 * index_kpool for kpool models
IPS = PAGE // KPOOL  # index page size
HEAD = 128
FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
HAD_SCALE = 0.08838834764831845  # 1/sqrt(128), the fp32 constant of the kernels
TILE = 16
DEV = "cuda"


def _gfx950() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    return str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0] in (
        "gfx950",
        "gfx942",
    )


class Pool:
    index_kpool = KPOOL
    index_page_size = IPS
    index_head_dim = HEAD
    page_size = PAGE
    tail_extra_slots = 0


def rand_bf16(*shape):
    return (torch.randn(*shape, device=DEV) * 2).to(torch.bfloat16)


# --------------------------------------------------------------------------
# torch reference of the pool math and the cache layout
# --------------------------------------------------------------------------


def hadamard128(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    h = 1
    while h < n:
        v = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        a, b = v[..., 0, :], v[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2).reshape(*x.shape)
        h *= 2
    return x


def compress_pools(k_rows, g_rows, ape, round_scale=True):
    u = g_rows.float() + ape.float()[None]
    e = torch.exp(u - u.amax(dim=1, keepdim=True))
    z = (e * k_rows.float()).sum(1) / e.sum(1)
    x = z.to(torch.bfloat16).float()
    y = (hadamard128(x) * HAD_SCALE).to(torch.bfloat16).float()
    a = y.abs().amax(1).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(a / FP8_MAX))) if round_scale else a / FP8_MAX
    q = (y / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q.view(torch.uint8), scale


def _k_offsets(tok, page):
    d = torch.arange(HEAD, device=DEV)
    if page > 1:
        return (
            (tok // TILE)[..., None] * (TILE * HEAD)
            + (d // TILE)[None] * (TILE * TILE)
            + (tok % TILE)[..., None] * TILE
            + (d % TILE)[None]
        )
    return d[None].expand(*tok.shape, HEAD)


def write_cache(kv, loc, q_u8, scale):
    nb, page, cs = kv.shape
    flat = kv.view(nb, page * cs)
    pg, tok = loc // page, loc % page
    flat[pg[:, None], _k_offsets(tok, page)] = q_u8
    soff = page * HEAD + tok[:, None] * 4 + torch.arange(4, device=DEV)[None]
    flat[pg[:, None], soff] = scale.contiguous().view(torch.uint8).view(-1, 4)


def read_cache(kv):
    nb, page, cs = kv.shape
    flat = kv.view(nb, page * cs)
    k = flat[:, : page * HEAD][:, _k_offsets(torch.arange(page, device=DEV), page)]
    scale = flat[:, page * HEAD :].contiguous().view(torch.float32).view(nb, page)
    return k.view(FP8).float() * scale[..., None], scale


def ref_insert(kv, k, gate, ape, slot, pool, round_scale=True):
    n = k.shape[0]
    t = torch.arange(n, device=DEV)
    closing = (slot >= 0) & (t >= pool - 1)
    tc = t[closing]
    if tc.numel() == 0:
        return
    idx = tc[:, None] - (pool - 1) + torch.arange(pool, device=DEV)[None]
    q, s = compress_pools(k[idx], gate[idx], ape, round_scale)
    write_cache(kv, slot[closing].long(), q, s)


def ref_decode(kv, tail, tail_slot, key, score, ape, slot_map, pos, pool, round_scale=True):
    """The ordered walk: per request, token by token, read ring then stash."""
    B, next_n = key.shape[:2]
    ring = tail.shape[2]
    ts_c, sm_c, pos_c = tail_slot.cpu(), slot_map.cpu(), pos.cpu()
    for b in range(B):
        for i in range(next_n):
            p, ts, loc = int(pos_c[b, i]), int(ts_c[b, i]), int(sm_c[b, i])
            if p >= 0 and loc >= 0 and p % pool == pool - 1:
                blk, slot = max(ts, 0) // ring, p % pool
                k_rows = torch.empty(pool, HEAD, dtype=torch.bfloat16, device=DEV)
                g_rows = torch.empty_like(k_rows)
                for s in range(pool):
                    if s == slot:
                        k_rows[s], g_rows[s] = key[b, i], score[b, i]
                    else:
                        phys = (p - slot + s) % ring
                        k_rows[s], g_rows[s] = tail[blk, 0, phys], tail[blk, 1, phys]
                q, sc = compress_pools(k_rows[None], g_rows[None], ape, round_scale)
                write_cache(kv, torch.tensor([loc], device=DEV), q, sc)
            if p >= 0 and ts >= 0:
                tail[ts // ring, 0, p % ring] = key[b, i]
                tail[ts // ring, 1, p % ring] = score[b, i]


def assert_cache_close(tc, kv_ref, kv_got):
    y_ref, s_ref = read_cache(kv_ref)
    y_got, s_got = read_cache(kv_got)
    torch.testing.assert_close(y_got, y_ref, rtol=0.13, atol=2e-3)
    torch.testing.assert_close(s_got, s_ref, rtol=1e-6, atol=0)


def make_prefill(lens, pool, page, slot_dtype=torch.int64, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    slots = []
    n_pools = sum(length // pool for length in lens)
    num_blocks = max(1, math.ceil(n_pools / page)) + 2
    perm = torch.randperm(num_blocks * page, device=DEV, generator=g)[:n_pools].tolist()
    for length in lens:
        for t in range(length):
            slots.append(perm.pop() if t % pool == pool - 1 else -1)
    n = len(slots)
    k = torch.randn(n, HEAD, generator=g, device=DEV).bfloat16()
    gate = (torch.randn(n, HEAD, generator=g, device=DEV) * 3).bfloat16()
    ape = torch.randn(pool, HEAD, generator=g, device=DEV)
    slot = torch.tensor(slots, dtype=slot_dtype, device=DEV)
    kv = torch.zeros(num_blocks, page, HEAD + 4, dtype=torch.uint8, device=DEV)
    return k, gate, ape, slot, kv


def make_decode(B, next_n, pool, page=64, seed=0, ring=None, idx_dtype=torch.int32):
    g = torch.Generator(device=DEV).manual_seed(seed)
    ring = ring or pool
    nblocks = B + 2
    tail = torch.randn(nblocks, 2, ring, HEAD, generator=g, device=DEV).bfloat16()
    ctx = torch.randint(pool, 100000, (B,), generator=g, device=DEV)
    pos = (ctx[:, None] + torch.arange(next_n, device=DEV)[None]).to(torch.int32)
    blocks = torch.arange(B, dtype=torch.int32, device=DEV)[:, None]
    tail_slot = (blocks * ring + pos % ring).to(torch.int32)
    n_pages = math.ceil(B * next_n / page) + 2
    locs = torch.randperm(n_pages * page, generator=g, device=DEV)[: B * next_n]
    locs = locs.view(B, next_n).to(torch.int32)
    completes = pos % pool == pool - 1
    slot_map = torch.where(completes, locs, torch.full_like(locs, -1))
    key = torch.randn(B, next_n, HEAD, generator=g, device=DEV).bfloat16()
    score = torch.randn(B, next_n, HEAD, generator=g, device=DEV).bfloat16()
    ape = torch.randn(pool, HEAD, generator=g, device=DEV)
    kv = torch.zeros(n_pages, page, HEAD + 4, dtype=torch.uint8, device=DEV)
    return dict(kv=kv, tail=tail, tail_slot=tail_slot.to(idx_dtype), key=key, score=score,
                ape=ape, slot_map=slot_map.to(idx_dtype), pos=pos.to(idx_dtype))


def unique_page_table(rows, cols, num_pages):
    """page_table_64 whose read rows (multiples of KPOOL) map to distinct index
    pages, so no two pools ever share a write slot."""
    need = rows * ((cols + KPOOL - 1) // KPOOL)
    assert need <= num_pages, f"need {need} index pages, have {num_pages}"
    ids = torch.randperm(num_pages, device=DEV)[:need].view(rows, -1)
    pt = torch.zeros(rows, cols, dtype=torch.int32, device=DEV)
    pt[:, ::KPOOL] = (ids * KPOOL).to(torch.int32)[:, : pt[:, ::KPOOL].shape[1]]
    return pt


@unittest.skipUnless(_gfx950(), "gfx950 / gfx942 only")
class TestKpoolCompressHipKernels(CustomTestCase):
    """The kernel entry points against the torch reference and the Triton writers."""

    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.attention.dsa.hip_gfx950 import kpool_compress

        cls.ops = kpool_compress
        cls.ops.module()  # build once, outside the timed cases

    def test_insert_matches_reference_and_triton(self):
        cases = [
            ([1024], 4, 64, torch.int64, 0),
            ([1024], 4, 1, torch.int64, 0),
            ([2048], 4, 64, torch.int64, 16),
            ([2048], 4, 64, torch.int64, 32),
            ([1, 7, 130, 1000, 33, 4, 5], 4, 32, torch.int32, 0),
            ([8192] * 8, 4, 64, torch.int64, 0),
            ([131072] * 2, 4, 64, torch.int64, 0),
        ]
        for lens, pool, page, slot_dtype, lanes in cases:
            with self.subTest(lens=lens[:3], page=page, slot_dtype=slot_dtype, lanes=lanes):
                k, gate, ape, slot, kv = make_prefill(lens, pool, page, slot_dtype)
                ref = kv.clone()
                ref_insert(ref, k, gate, ape, slot, pool)
                got = kv.clone()
                self.ops.kpool_compress_insert(got, k, gate, ape, slot, pool, True, lanes)
                torch.cuda.synchronize()
                assert_cache_close(self, ref, got)
                if page == IPS:
                    # byte for byte against sglang's two-pass Triton writer
                    exact = kv.clone()
                    t = torch.arange(k.shape[0], device=DEV)
                    sel = ((slot >= 0) & (t >= pool - 1)).nonzero().flatten()
                    idx = sel[:, None] - (pool - 1) + torch.arange(pool, device=DEV)[None]
                    kpool_softmax_rotate_write_cache(
                        Pool(), exact.view(exact.shape[0], -1), k[idx].contiguous(),
                        gate[idx].contiguous(), ape, slot[sel].long().contiguous(),
                        round_scale=True,
                    )
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(exact, got), "HIP insert differs from Triton")

    def test_decode_matches_reference(self):
        cases = [
            (4, 1, 4, None, torch.int32),
            (3, 4, 4, None, torch.int32),
            (8, 2, 4, 8, torch.int32),
            (5, 4, 4, 8, torch.int64),
            (4, 7, 4, 16, torch.int32),
            (3, 16, 4, 32, torch.int32),
            (64, 1, 4, None, torch.int64),
            (32, 16, 4, None, torch.int32),
        ]
        for B, next_n, pool, ring, idx_dtype in cases:
            with self.subTest(B=B, next_n=next_n, ring=ring, idx_dtype=idx_dtype):
                d = make_decode(B, next_n, pool, 64, seed=B * 100 + next_n, ring=ring,
                                idx_dtype=idx_dtype)
                kv_ref, tail_ref = d["kv"].clone(), d["tail"].clone()
                ref_decode(kv_ref, tail_ref, d["tail_slot"], d["key"], d["score"], d["ape"],
                           d["slot_map"], d["pos"], pool)
                kv_got, tail_got = d["kv"].clone(), d["tail"].clone()
                self.ops.kpool_decode_update(kv_got, tail_got, d["tail_slot"], d["key"],
                                             d["score"], d["ape"], d["slot_map"], d["pos"],
                                             pool, True)
                torch.cuda.synchronize()
                assert_cache_close(self, kv_ref, kv_got)
                self.assertTrue(torch.equal(tail_ref, tail_got), "tail ring mismatch")

    def test_rejects_bad_metadata(self):
        d = make_decode(4, 1, 4, 64, seed=9, ring=4)
        with self.assertRaisesRegex(Exception, "power of two"):
            bad = torch.randn(6, 2, 12, HEAD, device=DEV).bfloat16()
            self.ops.kpool_decode_update(d["kv"], bad, d["tail_slot"], d["key"], d["score"],
                                         d["ape"], d["slot_map"], d["pos"], 4, True)
        with self.assertRaisesRegex(Exception, "pool_size"):
            k, gate, ape, slot, kv = make_prefill([64], 4, 64)
            self.ops.kpool_compress_insert(kv, k, gate, torch.randn(5, HEAD, device=DEV),
                                           slot, 5, True, 0)


@unittest.skipUnless(_gfx950(), "gfx950 / gfx942 only")
class TestKpoolHipCompressWrappers(CustomTestCase):
    """The srt wrappers (SGLANG_DSA_KPOOL_HIP_COMPRESS path) against the Triton
    writers on identical decode steps and extend plans."""

    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.attention.dsa.hip_gfx950 import kpool_compress
        from sglang.srt.layers.attention.dsa import kpool_hip_compress

        kpool_compress.module()
        cls.wrap = kpool_hip_compress

    def _decode(self, seed, batch, valid, steps, num_pages):
        torch.manual_seed(seed)
        random.seed(seed)
        pool = Pool()
        R = 97
        buf1 = torch.randint(0, 256, (num_pages, IPS * (HEAD + 4)), dtype=torch.uint8, device=DEV)
        buf2 = buf1.clone()
        tail_kv = rand_bf16(R, 2, KPOOL, HEAD)
        tail_k1 = tail_kv[:, 0].clone().contiguous()
        tail_s1 = tail_kv[:, 1].clone().contiguous()
        tail_kv2 = tail_kv.clone()
        ape = (torch.randn(KPOOL, HEAD, device=DEV) * 0.5).float()
        req = torch.full((batch,), -1, dtype=torch.int32, device=DEV)
        req[:valid] = torch.randperm(R - 1, device=DEV)[:valid].to(torch.int32)
        pos0 = torch.randint(0, 4096, (batch,), device=DEV, dtype=torch.int64)
        cols = (4096 + steps) // 64 + 2  # page_table_64 rows are 64-token pages
        bt = unique_page_table(batch, cols, num_pages)
        for step in range(steps):
            pos = pos0 + step
            seq_lens = (pos + 1).to(torch.int32)
            seq_lens[valid:] = 0
            out_cache_loc = torch.randint(1, 1 << 20, (batch,), device=DEV, dtype=torch.int64)
            out_cache_loc[valid:] = 0
            key = rand_bf16(batch, HEAD)
            score = rand_bf16(batch, HEAD)
            kpool_decode_update_and_maybe_write_cache(
                pool=pool, buf=buf1, tail_k=tail_k1, tail_score=tail_s1, key=key,
                slot_score=score, ape=ape, block_tables=bt, req_pool_indices=req,
                positions=pos, seq_lens=seq_lens, out_cache_loc=out_cache_loc, round_scale=True,
            )
            self.wrap.hip_kpool_decode_update(
                pool=pool, buf=buf2, tail_kv=tail_kv2, key=key, slot_score=score, ape=ape,
                block_tables=bt, req_pool_indices=req, positions=pos, seq_lens=seq_lens,
                out_cache_loc=out_cache_loc, round_scale=True,
            )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(buf1, buf2), "index cache differs after decode steps")
        self.assertTrue(torch.equal(tail_k1, tail_kv2[:, 0]), "tail K differs")
        self.assertTrue(torch.equal(tail_s1, tail_kv2[:, 1]), "tail gate differs")

    def test_decode_steps(self):
        for seed, batch, valid, steps, num_pages in [
            (0, 64, 48, 9, 4096),
            (1, 1, 1, 12, 64),
            (2, 96, 96, 5, 4096),
        ]:
            with self.subTest(seed=seed, batch=batch, valid=valid):
                self._decode(seed, batch, valid, steps, num_pages)

    def _extend(self, seed, reqs, num_pages, max_q, misaligned_frac):
        torch.manual_seed(seed)
        random.seed(seed)
        pool = Pool()
        R = 65
        q_lens, seq_lens, req_ids = [], [], []
        for i in range(reqs):
            q = random.randint(1, max_q)
            prefix = random.randint(0, 2048)
            if random.random() > misaligned_frac:
                prefix -= prefix % KPOOL
            q_lens.append(q)
            seq_lens.append(prefix + q)
            req_ids.append(i + 1)
        n_tokens = sum(q_lens)
        cpu = _KPoolCpuPlan()
        _append_compress_rows(cpu, KPOOL, reqs, q_lens, seq_lens, req_ids)
        cols = max(seq_lens) // 64 + 2
        page_table = unique_page_table(reqs, cols, num_pages)
        t = lambda xs, dt: torch.tensor(xs, dtype=dt, device=DEV)  # noqa: E731
        pool_req = t(cpu.pool_req, torch.int64)
        pool_id = t(cpu.pool_pool_id, torch.int64)
        chunk_src = t(cpu.pool_chunk_src, torch.int64)
        batch_idx = t(cpu.pool_batch_idx, torch.int64)
        n_from_tail = t(cpu.pool_n_from_tail, torch.int32)
        tail_base = t(cpu.pool_tail_logical_base, torch.int32)
        packed = page_table[batch_idx, (pool_id // IPS) * KPOOL].to(torch.int64) // KPOOL
        write_loc = packed * IPS + pool_id % IPS
        any_from_tail = any(n > 0 for n in cpu.pool_n_from_tail)

        buf1 = torch.randint(0, 256, (num_pages, IPS * (HEAD + 4)), dtype=torch.uint8, device=DEV)
        buf2 = buf1.clone()
        tail_kv = rand_bf16(R, 2, KPOOL, HEAD)
        tail_k1 = tail_kv[:, 0].clone().contiguous()
        tail_s1 = tail_kv[:, 1].clone().contiguous()
        tail_kv2 = tail_kv.clone()
        ape = (torch.randn(KPOOL, HEAD, device=DEV) * 0.5).float()
        key = rand_bf16(n_tokens, HEAD)
        score = rand_bf16(n_tokens, HEAD)

        kpool_assemble_softmax_rotate_write_cache(
            pool=pool, buf=buf1, chunk_k=key, chunk_score=score, tail_k=tail_k1,
            tail_score=tail_s1, req_pool_idx=pool_req, n_from_tail=n_from_tail,
            chunk_src_start=chunk_src, tail_logical_base=tail_base, ape=ape, loc=write_loc,
            round_scale=True,
        )
        self.wrap.hip_kpool_insert_from_plan(
            pool=pool, buf=buf2, chunk_k=key, chunk_score=score, chunk_src=chunk_src,
            n_from_tail=n_from_tail, write_loc=write_loc, ape=ape, round_scale=True,
        )
        if any_from_tail:
            # exactly what _compress_write_extend does on the HIP path
            kpool_assemble_softmax_rotate_write_cache(
                pool=pool, buf=buf2, chunk_k=key, chunk_score=score, tail_k=tail_kv2[:, 0],
                tail_score=tail_kv2[:, 1], req_pool_idx=pool_req, n_from_tail=n_from_tail,
                chunk_src_start=chunk_src, tail_logical_base=tail_base, ape=ape, loc=write_loc,
                write_mask=n_from_tail > 0, round_scale=True,
            )
        torch.cuda.synchronize()
        # the two-pass Triton writer has the kernel's arithmetic order; the
        # online-softmax assemble writer may differ by one fp8 ulp
        sel = (n_from_tail == 0).nonzero().flatten()
        buf3 = buf2.clone()
        idx = chunk_src[sel][:, None] + torch.arange(KPOOL, device=DEV)[None]
        kpool_softmax_rotate_write_cache(
            pool, buf3, key[idx].contiguous(), score[idx].contiguous(), ape,
            write_loc[sel].contiguous(), round_scale=True,
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(buf3, buf2), "HIP insert differs from the two-pass Triton writer")
        self.assertTrue(torch.equal(tail_k1, tail_kv2[:, 0]), "tail K must be untouched")
        ya, sa = read_cache(buf1.view(num_pages, IPS, HEAD + 4))
        yb, sb = read_cache(buf2.view(num_pages, IPS, HEAD + 4))
        pg, tok = write_loc // IPS, write_loc % IPS
        torch.testing.assert_close(yb[pg, tok], ya[pg, tok], rtol=0.13, atol=2e-3)
        torch.testing.assert_close(sb[pg, tok], sa[pg, tok], rtol=0, atol=0)

    def test_extend_plans(self):
        for seed, reqs, num_pages, max_q, frac in [
            (0, 8, 2048, 2048, 0.0),
            (1, 16, 2048, 1024, 0.5),
            (2, 4, 256, 7, 1.0),
            (3, 2, 2048, 8192, 0.0),
            (4, 48, 4096, 200, 0.3),
        ]:
            with self.subTest(seed=seed, reqs=reqs, max_q=max_q, misaligned=frac):
                self._extend(seed, reqs, num_pages, max_q, frac)


if __name__ == "__main__":
    unittest.main()
