"""In-tree HIP decode indexer logits (kernels/jit/csrc/dsa_gfx950/fp8_paged_mqa_logits.cuh)
on GLM-5.3-Flash geometry over the pooled index cache ([pages, 64, 1, 132],
16x16 preshuffled keys then fp32 scales).

Every case is graded against an fp32 torch reference that reads the plain
(unshuffled) cache; the kernel writes only each row's causal window, so the
output is pre-filled with -inf and the mask is checked exactly. Context
lengths come in both kernel forms ([B] per sequence, [B, next_n] per row),
and aiter's deepgemm_fp8_paged_mqa_logits is compared where importable (the
production reference on ROCm: same mask, logits within accumulation order).
"""

import random
import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=240, suite="stage-b-test-1-gpu-small-amd-mi35x")

DEV = "cuda"
NUM_HEADS = 32
HEAD_DIM = 128
FP8 = torch.float8_e4m3fn
TILE = 16


def _gfx950() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    return str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0] == "gfx950"


def kv_cache_cast_to_fp8(x):
    """Co-pack a bf16 [blocks, block_size, 1, 128] cache: per block the fp8 key
    bytes then the block_size fp32 scales."""
    num_blocks, block_size, one, head_dim = x.shape
    amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = amax / 240.0
    x_fp8 = (x * (1.0 / sf)).to(FP8)
    out = torch.empty((num_blocks, block_size * (head_dim + 4)), device=DEV, dtype=torch.uint8)
    out[:, : block_size * head_dim] = x_fp8.view(num_blocks, -1).view(torch.uint8)
    out[:, block_size * head_dim :] = sf.view(num_blocks, block_size).view(torch.uint8)
    return out.view(num_blocks, block_size, 1, head_dim + 4)


def preshuffle_kv_data(kv):
    """The 16x16 MFMA preshuffle of the key bytes that kpool_fp8_index writes:
    byte (tok, d) of a block moves to (tok // 16) * 2048 + (d // 16) * 256 +
    (tok % 16) * 16 + d % 16; the scale tail is untouched."""
    num_blocks, block_size, one, index_dim = kv.shape
    flat = kv.reshape(num_blocks, block_size * index_dim).clone()
    keys = flat[:, : block_size * HEAD_DIM].view(num_blocks, block_size, HEAD_DIM)
    shuffled = (
        keys.view(num_blocks, block_size // TILE, TILE, HEAD_DIM // TILE, TILE)
        .permute(0, 1, 3, 2, 4)
        .reshape(num_blocks, block_size * HEAD_DIM)
    )
    flat[:, : block_size * HEAD_DIM] = shuffled
    return flat.view(num_blocks, block_size, one, index_dim)


def ref_logits(q_fp8, kv, weights, context_lens, block_tables, max_model_len):
    batch_size, next_n, _, dim = q_fp8.shape
    num_blocks, block_size, _, index_dim = kv.shape
    flat = kv.reshape(num_blocks, block_size * index_dim)
    keys = flat[:, : block_size * dim].contiguous().view(num_blocks, block_size, dim)
    scales = flat[:, block_size * dim :].contiguous().view(torch.float32).view(num_blocks, block_size)
    q = q_fp8.float()
    out = torch.full((batch_size * next_n, max_model_len), float("-inf"), dtype=torch.float32, device=DEV)
    row_lens = context_lens.dim() == 2
    for b in range(batch_size):
        if row_lens:
            ends = context_lens[b].to(torch.int64)
        else:
            ends = int(context_lens[b]) - next_n + 1 + torch.arange(next_n, device=DEV)
        n_tok = int(ends.max().item())
        if n_tok <= 0:
            continue
        pos = torch.arange(n_tok, device=DEV)
        blk = block_tables[b, pos // block_size].to(torch.int64)
        tok = pos % block_size
        k = keys[blk, tok].view(FP8).float() * scales[blk, tok][:, None]
        s = torch.relu(torch.einsum("nhd,pd->nhp", q[b], k))
        s = torch.einsum("nhp,nh->np", s, weights[b * next_n : (b + 1) * next_n])
        s = s.masked_fill(pos[None, :] >= ends[:, None], float("-inf"))
        out[b * next_n : (b + 1) * next_n, :n_tok] = s
    return out


def build_inputs(batch_size, next_n, avg_kv_len, block_size, var_ratio=0.3, seed=0, per_row_lens=False):
    torch.manual_seed(seed)
    random.seed(seed)
    max_model_len = 2 * avg_kv_len
    if var_ratio == 0.0:
        context_lens = torch.full((batch_size,), avg_kv_len, dtype=torch.int32, device=DEV)
    else:
        lo = max(1, int((1 - var_ratio) * avg_kv_len))
        hi = int((1 + var_ratio) * avg_kv_len) + 1
        context_lens = torch.randint(lo, hi, (batch_size,), device=DEV).to(torch.int32)
    context_lens = context_lens.clamp(min=next_n)
    blocks_per_seq = (context_lens.to(torch.int64) + block_size - 1) // block_size
    max_block_len = int(blocks_per_seq.max().item())
    num_blocks = int(blocks_per_seq.sum().item())
    q = torch.randn((batch_size, next_n, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=DEV)
    kv = torch.randn((num_blocks, block_size, 1, HEAD_DIM), dtype=torch.bfloat16, device=DEV)
    weights = torch.randn((batch_size * next_n, NUM_HEADS), dtype=torch.float32, device=DEV)
    # a shuffled block pool catches a kernel that walks the cache linearly
    pool = list(range(num_blocks))
    random.shuffle(pool)
    pool_t = torch.tensor(pool, device=DEV, dtype=torch.int32)
    col = torch.arange(max_block_len, device=DEV, dtype=torch.int64)
    starts = torch.cumsum(blocks_per_seq, 0) - blocks_per_seq
    block_tables = torch.where(
        col[None, :] < blocks_per_seq[:, None],
        pool_t[(starts[:, None] + col[None, :]) % num_blocks],
        torch.zeros((), device=DEV, dtype=torch.int32),
    ).to(torch.int32)
    if per_row_lens:
        g = torch.Generator(device=DEV).manual_seed(seed + 1)
        drop = torch.randint(0, 3, (batch_size, next_n), generator=g, device=DEV)
        context_lens = (context_lens[:, None] - drop.cumsum(1).flip(1)).clamp(min=1).to(torch.int32).contiguous()
    return q.to(FP8), kv_cache_cast_to_fp8(kv), weights, context_lens, block_tables, max_model_len


def calc_diff(x, y):
    x, y = x.double(), y.double()
    return 1 - 2 * (x * y).sum() / (x * x + y * y).sum()


def _aiter_paged_logits():
    try:
        from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
    except Exception:  # noqa: BLE001 - aiter is optional for this test
        return None
    return deepgemm_fp8_paged_mqa_logits


@unittest.skipUnless(_gfx950(), "gfx950 only")
class TestFp8PagedMqaLogitsHip(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.attention.dsa.hip_gfx950 import fp8_paged_mqa_logits

        cls.ops = fp8_paged_mqa_logits
        cls.ops.module()
        fn = _aiter_paged_logits()
        cls.triton = staticmethod(fn) if fn is not None else None

    def _run(self, inp, block_size, **tunables):
        q_fp8, kv, weights, context_lens, block_tables, max_model_len = inp
        ref = ref_logits(q_fp8, kv, weights, context_lens, block_tables, max_model_len)
        kv_kernel = preshuffle_kv_data(kv) if block_size > 1 else kv
        out = torch.full_like(ref, float("-inf"))
        got = self.ops.fp8_paged_mqa_logits(q_fp8, kv_kernel, weights, context_lens, block_tables,
                                            max_model_len, out=out, **tunables)
        self.assertIs(got, out)
        torch.cuda.synchronize()
        ref_mask, out_mask = ref == float("-inf"), out == float("-inf")
        self.assertTrue(torch.equal(out_mask, ref_mask), "-inf mask mismatch")
        self.assertFalse(torch.isnan(out).any().item())
        o, r = out.masked_fill(out_mask, 0), ref.masked_fill(ref_mask, 0)
        torch.testing.assert_close(o, r, rtol=1e-2, atol=5.0)
        self.assertLess(float(calc_diff(o, r)), 1e-3)
        return out, kv_kernel

    def test_block64_shapes(self):
        """Production layout: 64-token preshuffled pages, one length per sequence."""
        for batch_size in (1, 4, 16, 64):
            for next_n in (1, 2, 4, 6):
                for avg_kv_len, var_ratio in ((1024, 0.3), (8192, 0.3), (8192, 0.0)):
                    with self.subTest(batch_size=batch_size, next_n=next_n, kv_len=avg_kv_len, var=var_ratio):
                        self._run(build_inputs(batch_size, next_n, avg_kv_len, 64, var_ratio), 64)

    def test_long_context_and_block1(self):
        for batch_size, next_n, avg_kv_len, block_size in [
            (1, 1, 32768, 64), (4, 4, 32768, 64), (1, 2, 131072, 64), (16, 6, 32768, 64),
            (4, 1, 1024, 1), (16, 4, 1024, 1), (2, 6, 8192, 1),
        ]:
            with self.subTest(batch_size=batch_size, next_n=next_n, kv_len=avg_kv_len, block=block_size):
                self._run(build_inputs(batch_size, next_n, avg_kv_len, block_size, 0.3), block_size)

    def test_per_row_lens(self):
        """[B, next_n] context lengths (one per query row, as the k-pool indexer's
        pooled lengths are); next_n 5 / 7 round up to the R = 6 / 8 kernels."""
        for batch_size, next_n, avg_kv_len in [(4, 2, 1024), (16, 4, 8192), (8, 6, 8192), (2, 5, 32768), (1, 7, 8192)]:
            with self.subTest(batch_size=batch_size, next_n=next_n, kv_len=avg_kv_len):
                inp = build_inputs(batch_size, next_n, avg_kv_len, 64, 0.3, per_row_lens=True)
                self.assertEqual(tuple(inp[3].shape), (batch_size, next_n))
                self._run(inp, 64)

    def test_tunables(self):
        """Explicit grid choices give the same logits as the host heuristics."""
        inp = build_inputs(8, 6, 8192, 64, 0.3, seed=3)
        for tunables in [dict(num_warps=8, chunk_k=256), dict(num_warps=2, chunk_k=128, split_kv=3),
                         dict(rows_per_block=1), dict(rows_per_block=3, chunk_k=64), dict(rows_per_block=6)]:
            with self.subTest(**tunables):
                self._run(inp, 64, **tunables)

    def test_matches_aiter(self):
        """The production reference on ROCm, on the decode shape the indexer uses
        (one row per request, next_n 1): same mask, logits within accumulation order."""
        if self.triton is None:
            self.skipTest("aiter deepgemm_fp8_paged_mqa_logits not importable")
        for batch_size, avg_kv_len in [(16, 1024), (64, 8192), (4, 32768)]:
            with self.subTest(batch_size=batch_size, kv_len=avg_kv_len):
                inp = build_inputs(batch_size, 1, avg_kv_len, 64, 0.3, seed=11)
                q_fp8, kv, weights, context_lens, block_tables, max_model_len = inp
                hip, kv_kernel = self._run(inp, 64)
                tri = torch.full_like(hip, float("-inf"))
                self.triton(q_fp8, kv_kernel, weights, tri, context_lens, block_tables, max_model_len,
                            Preshuffle=True, KVBlockSize=64)
                torch.cuda.synchronize()
                mask = hip == float("-inf")
                self.assertTrue(torch.equal(mask, tri == float("-inf")))
                torch.testing.assert_close(hip.masked_fill(mask, 0), tri.masked_fill(mask, 0), rtol=1e-3, atol=1e-2)

    def test_rejects_bad_inputs(self):
        q_fp8, kv, weights, context_lens, block_tables, max_model_len = build_inputs(2, 1, 1024, 64, 0.3)
        kv_kernel = preshuffle_kv_data(kv)
        self.assertFalse(self.ops.is_supported(32, 128, 32))
        with self.assertRaisesRegex(Exception, "n_heads=32"):
            self.ops.fp8_paged_mqa_logits(q_fp8[:, :, :16].contiguous(), kv_kernel, weights, context_lens,
                                          block_tables, max_model_len)
        with self.assertRaisesRegex(Exception, "block_size must be 1"):
            self.ops.fp8_paged_mqa_logits(q_fp8, kv_kernel.reshape(-1, 32, 1, 132), weights, context_lens,
                                          block_tables, max_model_len)
        with self.assertRaisesRegex(Exception, "out must be"):
            self.ops.fp8_paged_mqa_logits(q_fp8, kv_kernel, weights, context_lens, block_tables, max_model_len,
                                          out=torch.empty(2, max_model_len - 1, dtype=torch.float32, device=DEV))


if __name__ == "__main__":
    unittest.main()
