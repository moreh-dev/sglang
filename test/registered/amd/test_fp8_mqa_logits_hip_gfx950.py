"""In-tree HIP prefill indexer logits (kernels/jit/csrc/dsa_gfx950/fp8_mqa_logits.cuh)
on GLM-5.3-Flash geometry (n_heads 32, head_dim 128).

Every case is graded against an fp32 torch reference over exactly the fp8
inputs the kernel consumes, with the output poisoned with NaN first so an
unwritten position fails the -inf mask check. The window modes are the ones
the k-pool indexer hands the kernel: causal, cp, misaligned starts, empty rows
(cu_ends below zero or below cu_starts), bounds past seq_len_kv and several
requests packed into one chunk. Where aiter's Triton kernel is importable its
output is compared too (same mask, logits within fp32 accumulation order).
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=240, suite="stage-b-test-1-gpu-small-amd-mi35x")

DEV = "cuda"
NUM_HEADS = 32
HEAD_DIM = 128
FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max
_REF_SCORE_BUDGET = 512 << 20  # fp32 bytes of the reference's [heads, chunk, s_k] scores

SHAPES = [(1, 1), (1, 113), (17, 76), (61, 1024), (128, 1024), (1024, 1560), (64, 2048),
          (64, 8192), (128, 64), (1024, 1000), (4096, 32768), (1024, 131072)]
WINDOWS = ["causal", "cp", "misaligned", "empty", "past_end", "multi_req"]


def _gfx950() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    return str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0] == "gfx950"


def per_token_cast_to_fp8(x):
    amax = x.abs().float().amax(dim=1, keepdim=True).clamp(1e-4)
    sf = amax / FP8_MAX
    return (x * (1.0 / sf)).to(FP8), sf.squeeze(1)


def ref_logits(q_fp8, k_fp8, scales, weights, ks, ke):
    q = q_fp8.float()
    kv = k_fp8.float() * scales[:, None]
    s_q, s_k = q.shape[0], kv.shape[0]
    col = torch.arange(s_k, device=DEV)[None, :]
    mask = (col >= ks[:, None]) & (col < ke[:, None])
    out = torch.empty((s_q, s_k), dtype=torch.float32, device=DEV)
    chunk = max(1, _REF_SCORE_BUDGET // max(1, NUM_HEADS * s_k * 4))
    for m0 in range(0, s_q, chunk):
        m1 = min(m0 + chunk, s_q)
        score = torch.einsum("mhd,nd->hmn", q[m0:m1], kv)
        part = (score.relu() * weights[m0:m1].T[:, :, None]).sum(0)
        out[m0:m1] = part.masked_fill(~mask[m0:m1], float("-inf"))
        del score, part
    return out, int(mask.sum())


def _num_requests(s_q, s_k):
    return max(1, min(4, s_q // 2, s_k // 2))


def make_windows(s_q, s_k, window):
    rows = torch.arange(s_q, device=DEV)
    i32 = torch.int32
    if window == "causal":
        return (torch.zeros(s_q, dtype=i32, device=DEV),
                (torch.arange(s_q, device=DEV) + (s_k - s_q)).to(i32))
    if window == "cp":
        assert s_k % s_q == 0 and s_q % 2 == 0
        chunk, cp_size = s_q // 2, s_k // s_q
        cp_id = cp_size // 3
        i = torch.arange(chunk, device=DEV)
        ke = torch.cat([cp_id * chunk + i, (cp_size * 2 - 1 - cp_id) * chunk + i])
        return torch.zeros(s_q, dtype=i32, device=DEV), ke.to(i32)
    if window == "misaligned":
        ks = ((rows * 53 + 100) % max(1, s_k // 2)).to(i32)
        return ks, torch.minimum(ks + max(1, s_k // 3), torch.full_like(ks, s_k)).to(i32)
    if window == "empty":
        ks = torch.where(rows % 3 == 1, min(100, s_k), 0)
        ke = torch.where(rows % 3 == 0, -1 - (rows % 7),
                         torch.where(rows % 3 == 1, 0, torch.minimum(rows + 1, ks + s_k)))
        return ks.to(i32), ke.to(i32)
    if window == "past_end":
        ks = torch.where(rows % 2 == 0, s_k + 17, 0)
        ke = torch.where(rows % 2 == 0, s_k + 64, torch.minimum(rows + 1, ks + s_k))
        return ks.to(i32), ke.to(i32)
    if window == "multi_req":
        nreq = _num_requests(s_q, s_k)
        rows_per_req = (s_q + nreq - 1) // nreq
        kv_per_req = s_k // nreq
        req = torch.clamp(rows // rows_per_req, max=nreq - 1)
        ks = req * kv_per_req
        local = rows - req * rows_per_req
        ke = torch.minimum(torch.clamp(ks + (kv_per_req - rows_per_req) + local + 1, min=0),
                           ks + kv_per_req)
        return ks.to(i32), torch.maximum(ke, ks).to(i32)
    raise ValueError(window)


def make_inputs(s_q, s_k, window, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(s_q, NUM_HEADS, HEAD_DIM, generator=g, device=DEV).bfloat16()
    kv = torch.randn(s_k, HEAD_DIM, generator=g, device=DEV).bfloat16()
    k_fp8, scales = per_token_cast_to_fp8(kv)
    weights = torch.randn(s_q, NUM_HEADS, generator=g, device=DEV)
    ks, ke = make_windows(s_q, s_k, window)
    return q.to(FP8), k_fp8, scales, weights, ks, ke


def rehydrate(out, ks, ke):
    s_k = out.shape[1]
    col = torch.arange(s_k, device=DEV)[None, :]
    keep = (col >= ks.clamp(0, s_k)[:, None]) & (col < ke.clamp(0, s_k)[:, None])
    return torch.where(keep, out, torch.full_like(out, float("-inf")))


def calc_diff(x, y):
    x, y = x.double(), y.double()
    return 1 - 2 * (x * y).sum() / (x * x + y * y).sum()


def _cases():
    for s_q, s_k in SHAPES:
        for window in WINDOWS:
            if window == "cp" and (s_k % s_q != 0 or s_q % 2 != 0):
                continue
            if window == "multi_req" and _num_requests(s_q, s_k) < 2:
                continue
            for clean in (True, False):
                yield s_q, s_k, window, clean


def _aiter_triton_logits():
    try:
        from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits
    except Exception:  # noqa: BLE001 - aiter is optional for this test
        return None
    return fp8_mqa_logits


@unittest.skipUnless(_gfx950(), "gfx950 only")
class TestFp8MqaLogitsHip(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.attention.dsa.hip_gfx950 import fp8_mqa_logits

        cls.ops = fp8_mqa_logits
        cls.ops.module()
        fn = _aiter_triton_logits()
        # a plain function on the class would bind self; keep it a staticmethod
        cls.triton = staticmethod(fn) if fn is not None else None

    def _check(self, out, ref, n_valid, tag):
        out_mask, ref_mask = out == float("-inf"), ref == float("-inf")
        self.assertTrue(torch.equal(out_mask, ref_mask), f"{tag}: -inf mask mismatch")
        self.assertFalse(torch.isnan(out).any().item(), f"{tag}: unwritten (NaN) positions")
        o, r = out.masked_fill(out_mask, 0), ref.masked_fill(ref_mask, 0)
        torch.testing.assert_close(o, r, rtol=1e-2, atol=1e-2)
        if n_valid > 1:
            self.assertLess(float(calc_diff(o, r)), 1e-3, f"{tag}: calc_diff too large")

    def test_windows_against_reference(self):
        for s_q, s_k, window, clean in _cases():
            with self.subTest(s_q=s_q, s_k=s_k, window=window, clean=clean):
                q_fp8, k_fp8, scales, weights, ks, ke = make_inputs(s_q, s_k, window)
                ref, n_valid = ref_logits(q_fp8, k_fp8, scales, weights, ks, ke)
                out = torch.full((s_q, s_k), float("nan"), dtype=torch.float32, device=DEV)
                got = self.ops.fp8_mqa_logits(q_fp8, k_fp8, scales, weights, ks, ke,
                                              clean_logits=clean, out=out)
                self.assertIs(got, out)
                torch.cuda.synchronize()
                if not clean:
                    got = rehydrate(got, ks, ke)
                self._check(got, ref, n_valid, f"{s_q}x{s_k} {window} clean={clean}")

    def test_matches_aiter_triton(self):
        """The production reference on ROCm: same mask, logits within fp32 accumulation order."""
        if self.triton is None:
            self.skipTest("aiter Triton fp8_mqa_logits not importable")
        for s_q, s_k, window in [(128, 1024, "causal"), (1024, 1560, "multi_req"),
                                 (64, 8192, "misaligned"), (4096, 32768, "causal")]:
            with self.subTest(s_q=s_q, s_k=s_k, window=window):
                q_fp8, k_fp8, scales, weights, ks, ke = make_inputs(s_q, s_k, window, seed=5)
                hip = self.ops.fp8_mqa_logits(q_fp8, k_fp8, scales, weights, ks, ke, clean_logits=True)
                tri = self.triton(q_fp8, k_fp8, scales, weights, ks, ke, clean_logits=True)[:, :s_k]
                torch.cuda.synchronize()
                mask = hip == float("-inf")
                self.assertTrue(torch.equal(mask, tri == float("-inf")))
                torch.testing.assert_close(hip.masked_fill(mask, 0), tri.masked_fill(mask, 0),
                                           rtol=1e-3, atol=1e-2)

    def test_tunables_and_shape_guard(self):
        """Explicit grid choices give the heuristics' logits; other head shapes are refused."""
        q_fp8, k_fp8, scales, weights, ks, ke = make_inputs(1024, 8192, "multi_req", seed=1)
        ref, n_valid = ref_logits(q_fp8, k_fp8, scales, weights, ks, ke)
        for num_warps, split_n, rev in [(4, 0, -1), (8, 3, 0), (8, 16, 1)]:
            with self.subTest(num_warps=num_warps, split_n=split_n, reverse_rows=rev):
                out = torch.full((1024, 8192), float("nan"), dtype=torch.float32, device=DEV)
                self.ops.fp8_mqa_logits(q_fp8, k_fp8, scales, weights, ks, ke, num_warps=num_warps,
                                        split_n=split_n, reverse_rows=rev, out=out)
                torch.cuda.synchronize()
                self._check(out, ref, n_valid, f"warps={num_warps} split={split_n}")
        self.assertFalse(self.ops.is_supported(64, 128))
        with self.assertRaisesRegex(Exception, "n_heads=32"):
            self.ops.fp8_mqa_logits(q_fp8[:, :16].contiguous(), k_fp8, scales, weights, ks, ke)
        with self.assertRaisesRegex(Exception, "out must be"):
            self.ops.fp8_mqa_logits(q_fp8, k_fp8, scales, weights, ks, ke,
                                    out=torch.empty(1024, 8191, dtype=torch.float32, device=DEV))


if __name__ == "__main__":
    unittest.main()
