"""Exactness of the in-tree HIP fp8_mqa_topk (k-pool prefill indexer logits +
top-k, Step 0) against fp32 torch logits.

The op never stores its logits, so the reference recomputes them; fp8 products
are exact and only the fp32 summation order differs, so a returned key may
disagree with the reference only within rounding of the row's k-th value."""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci

FP8 = torch.float8_e4m3fn
NUM_HEADS = 32
HEAD_DIM = 128
KPOOL = 4


def _skip_reason():
    if not torch.cuda.is_available() or torch.version.hip is None:
        return "needs a ROCm GPU"
    from sglang.kernels.ops.attention.dsa.hip_gfx950 import fp8_mqa_topk

    if not fp8_mqa_topk.supported_hardware():
        return "kernels are gfx950-only"
    if fp8_mqa_topk.module_or_none() is None:
        return "fp8_mqa_topk module failed to build"
    return None


SKIP = _skip_reason()

# gfx950 kernels: the stage-b mi35x suite, as test_dsa_gfx950_fused_indexer.py.
register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _inputs(rows: int, keys: int, *, requests: int = 1, seed: int = 0):
    """One prefill chunk in the k-pool layout: row t of a request is the token at
    position 4 * keys - rows + t and sees pools [0, (pos + 1) // 4) of its own
    request; requests are stacked along the key axis."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(rows, NUM_HEADS, HEAD_DIM, device="cuda", generator=g)
    q_fp8 = (q / q.abs().amax(-1, keepdim=True) * 448.0).to(FP8)
    weights = (torch.rand(rows, NUM_HEADS, device="cuda", generator=g) - 0.2) * 0.1
    k = torch.randn(keys, HEAD_DIM, device="cuda", generator=g)
    k_scale = torch.exp2(torch.ceil(torch.log2(k.abs().amax(-1) / 448.0)))
    k_fp8 = (k / k_scale[:, None]).to(FP8)
    starts = torch.empty(rows, dtype=torch.int32)
    ends = torch.empty(rows, dtype=torch.int32)
    per_request = rows // requests
    for r in range(requests):
        a = r * per_request
        b = rows if r == requests - 1 else a + per_request
        pos = torch.arange(keys * KPOOL - (b - a), keys * KPOOL)
        starts[a:b] = r * keys
        ends[a:b] = r * keys + ((pos + 1) // KPOOL).clamp(max=keys)
    return (
        q_fp8,
        k_fp8.repeat(requests, 1).contiguous(),
        k_scale.repeat(requests).contiguous(),
        weights.contiguous(),
        starts.cuda(),
        ends.cuda(),
    )


@unittest.skipIf(SKIP is not None, SKIP or "")
class TestFp8MqaTopkHipGfx950(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.attention.dsa.hip_gfx950 import fp8_mqa_topk

        cls.op = fp8_mqa_topk

    def _assert_exact(self, q_fp8, k_fp8, k_scale, weights, starts, ends, out):
        topk, keys = out.shape[1], k_fp8.shape[0]
        cols = torch.arange(keys, device="cuda")
        slots = torch.arange(topk, device="cuda")
        for r0 in range(0, out.shape[0], 32):
            r1 = min(out.shape[0], r0 + 32)
            logits = torch.einsum(
                "mhd,nd->mhn", q_fp8[r0:r1].float(), k_fp8.float()
            ).relu_()
            logits = (logits * weights[r0:r1, :, None]).sum(1) * k_scale
            lo = starts[r0:r1].long().clamp(0, keys)[:, None]
            hi = ends[r0:r1].long().clamp(0, keys)[:, None]
            window = (cols >= lo) & (cols < hi)
            logits = logits.masked_fill(~window, float("-inf"))
            want = window.sum(1).clamp(max=topk)

            ids = out[r0:r1].long()
            valid = ids >= 0
            # The row's keys first, then -1 padding.
            self.assertTrue(torch.equal(valid, slots[None, :] < want[:, None]))
            key = (ids + lo).clamp(0, keys - 1)
            self.assertTrue(bool(((key >= lo) & (key < hi) | ~valid).all()))
            hits = torch.zeros(window.shape, dtype=torch.int32, device="cuda")
            hits.scatter_add_(1, key, valid.int())
            self.assertLessEqual(int(hits.max()), 1, "duplicate keys")
            chosen = hits > 0

            ranked = logits.sort(dim=1, descending=True).values
            kth = ranked.gather(1, (want - 1).clamp(min=0)[:, None])
            finite = logits.masked_fill(~window, 0)
            tol = 1e-5 * finite.abs().amax(1, keepdim=True)
            picked = logits.gather(1, key)
            self.assertFalse(bool((valid & (picked < kth - tol)).any()), "low key")
            self.assertFalse(bool((window & ~chosen & (logits > kth + tol)).any()))

    def test_matches_reference(self):
        """One case per path a chunk can take: an odd row count whose rows hold
        225..600 keys (rows below 512 take every key, then -1), many splits merged
        by the select, and a two-request prefix layout with nonzero row starts."""
        for rows, keys, requests in ((1501, 600, 1), (96, 120000, 1), (2048, 20000, 2)):
            with self.subTest(rows=rows, keys=keys, requests=requests):
                inputs = _inputs(rows, keys, requests=requests)
                fallback_rows = torch.zeros(1, dtype=torch.int32, device="cuda")
                self.assertTrue(self.op.is_supported(NUM_HEADS, HEAD_DIM))
                out = self.op.fp8_mqa_topk(*inputs, fallback_row_count=fallback_rows)
                self._assert_exact(*inputs, out)
                # Random keys always leave enough candidates above the sampled tau.
                self.assertEqual(int(fallback_rows), 0)

    def test_fallback_rows_are_exact(self):
        """The exact recompute serves only rows whose candidates cannot prove
        their top-k, which random data never produces; force every row there."""
        inputs = _inputs(300, 5000)
        fallback_rows = torch.zeros(1, dtype=torch.int32, device="cuda")
        out = self.op.fp8_mqa_topk(
            *inputs, force_fallback=True, fallback_row_count=fallback_rows
        )
        self._assert_exact(*inputs, out)
        self.assertEqual(int(fallback_rows), 300)


if __name__ == "__main__":
    unittest.main()
