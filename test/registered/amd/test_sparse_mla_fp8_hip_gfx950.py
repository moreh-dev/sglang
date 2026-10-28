"""In-tree HIP FP8 sparse MLA attention core (kernels/jit/csrc/dsa_gfx950/sparse_mla_fp8.cuh)
on GLM-5.3-Flash geometry (64 heads x 512 latent dims, FP8 K/V pool, BF16 q
converted in the kernel, BF16 out).

Two independent references grade every case: ideal FP64 attention after the
scalar Q encoding (quantization error of FP8 P and BF16 split partials
included: per-head relative RMS <= 0.07, LSE to 5e-4), and an FP64
re-implementation of the kernel's arithmetic (FP8 P at scale 1/256, BF16
split-store rounding, the kernel's split count) on dyadic inputs whose QK
sums are exact (relative RMS <= 0.005). Also covered: the FP8-q entry equals
the BF16 one bit for bit, every dispatch family and capacity, padded and
tail-only selections, CUDA-graph replay with inputs rewritten in place, the
metadata rejections, and agreement with the tilelang DSA kernel sglang runs
on this cache today.
"""

import math
import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=420, suite="stage-b-test-1-gpu-small-amd-mi35x")

DEV = "cuda"
FP8 = torch.float8_e4m3fn
TOLERANCE = 0.07
DISPATCH_QUERIES = [1, 6, 7, 8, 12, 13, 21, 22, 23, 24, 42, 43, 64, 65, 128, 129]
CAPACITIES = [0, 1, 63, 64, 127, 128, 129, 2048, 2051, 2111, 2112, 2113, 2176]


def _gfx950() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    return str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0] == "gfx950"


def inputs(n, capacity=2112, *, strided=False, seed=2031071, padded=False):
    torch.manual_seed(seed + n + capacity)
    shape = (n, 65, 520) if strided else (n, 64, 512)
    q = torch.randn(shape, device=DEV, dtype=torch.bfloat16)[:, :64, :512]
    kv = torch.randn((8192, 512), device=DEV, dtype=torch.bfloat16).to(FP8)
    ids = torch.randint(len(kv), (n, capacity), device=DEV, dtype=torch.int32)
    if capacity:
        if n > 1:
            ids[0].fill_(-1)
        if padded:
            ids[:, 2051:] = -1
        if n > 1:
            ids[1, : min(9, capacity)] = len(kv) - 1
            ids[1, 10::97] = len(kv) + 17
    return q, kv, ids


def reference_rows(n):
    if n <= 129:
        return list(range(n))
    rows = {0, 1, n - 1}
    rows.update(i * (n - 1) // 31 for i in range(32))
    for boundary in range(2048, n, 2048):
        rows.update([boundary - 1, boundary])
    return sorted(rows)


def assert_reference(tc, out, lse, q, kv, ids, *, q_scale=1.0, kv_scale=1.0, scale=1 / 16, all_rows=False):
    """Independent ideal FP64 attention after scalar Q encoding, in bounded batches."""
    tc.assertEqual(out.shape, q.shape)
    tc.assertEqual(out.dtype, torch.bfloat16)
    tc.assertTrue(torch.isfinite(out).all().item())
    rows = list(range(len(q))) if all_rows else reference_rows(len(q))
    for start in range(0, len(rows), 16):
        batch = rows[start : start + 16]
        encoded = q[batch] if q.dtype == FP8 else (q[batch].float() / q_scale).to(FP8)
        decoded = encoded.double() * q_scale
        selected = ids[batch].long()
        valid = (selected >= 0) & (selected < len(kv))
        values = kv.view(torch.uint8)[selected.clamp(0, len(kv) - 1)].view(kv.dtype).double() * kv_scale
        values[~valid] = 0
        scores = torch.bmm(decoded, values.transpose(1, 2)) * scale
        scores.masked_fill_(~valid[:, None, :], -math.inf)
        expected_lse = scores.logsumexp(-1)
        nonempty = valid.any(-1)
        probability = torch.where(nonempty[:, None, None], scores.softmax(-1), 0.0)
        expected = torch.bmm(probability, values)
        error = out[batch].double() - expected
        rrms = error.square().mean(-1).sqrt() / expected.square().mean(-1).sqrt().clamp_min(1e-3)
        tc.assertLessEqual(float(rrms.max()), TOLERANCE)
        torch.testing.assert_close(out[batch].double(), expected, atol=TOLERANCE, rtol=TOLERANCE)
        if lse is not None:
            torch.testing.assert_close(lse[batch].double(), expected_lse, atol=5e-4, rtol=1e-5)
        tc.assertEqual(int(torch.count_nonzero(out[batch][~nonempty])), 0)
        if lse is not None:
            tc.assertTrue(torch.isneginf(lse[batch][~nonempty]).all().item())


def arithmetic_inputs(n, capacity, *, seed):
    """Dyadic Q/KV keep QK sums exact, isolating P and partial-store rounding."""
    q, kv, ids = inputs(n, capacity, strided=True, seed=seed)
    q.copy_((torch.randint(-8, 9, q.shape, device=DEV).float() / 8).to(q.dtype))
    kv.copy_((torch.randint(-8, 9, kv.shape, device=DEV).float() / 8).to(kv.dtype))
    return q, kv, ids


def arithmetic_reference(split_count, q, kv, ids, *, q_scale=1.0, kv_scale=1.0, scale=1 / 16):
    """FP64 matmuls with the kernel's FP8 P (scale 1/256), BF16 split-store
    rounding and split count; no device layouts."""
    splits = split_count(len(q))
    outputs, lses = [], []
    for row in range(len(q)):
        encoded = q[row] if q.dtype == FP8 else (q[row].float() / q_scale).to(FP8)
        decoded = encoded.double() * q_scale
        selected = ids[row].long()
        valid = (selected >= 0) & (selected < len(kv))
        values = kv.view(torch.uint8)[selected.clamp(0, len(kv) - 1)].view(kv.dtype).double() * kv_scale
        values[~valid] = 0
        scores = (decoded @ values.T) * scale
        scores[:, ~valid] = -math.inf
        parts, part_lses = [], []
        tile_count = (len(selected) + 127) // 128
        quotient, remainder = divmod(tile_count, splits)
        begin = 0
        for split in range(splits):
            end = begin + quotient + (split < remainder)
            maximum = torch.full((64,), -math.inf, device=DEV, dtype=torch.float64)
            denominator = torch.zeros_like(maximum)
            numerator = torch.zeros((64, 512), device=DEV, dtype=torch.float64)
            for tile in range(begin, end):
                lo, hi = tile * 128, min((tile + 1) * 128, len(selected))
                block = scores[:, lo:hi]
                new_maximum = torch.maximum(maximum, block.amax(-1))
                alpha = torch.where(torch.isfinite(maximum), (maximum - new_maximum).exp(), 0.0)
                probability = torch.where(torch.isfinite(new_maximum[:, None]), (block - new_maximum[:, None]).exp(), 0.0)
                rounded_p = (probability.float() * 256).to(FP8).double() / 256
                numerator = numerator * alpha[:, None] + rounded_p @ values[lo:hi]
                denominator = denominator * alpha + probability.sum(-1)
                maximum = new_maximum
            partial = numerator / denominator.clamp_min(1e-300)[:, None]
            if splits > 1:
                partial = partial.to(torch.bfloat16).double()
            parts.append(partial)
            part_lses.append(maximum + denominator.log())
            begin = end
        logs = torch.stack(part_lses)
        lse = logs.logsumexp(0)
        weights = torch.where(torch.isfinite(lse)[None, :], (logs - lse).exp(), 0.0)
        outputs.append((torch.stack(parts) * weights[:, :, None]).sum(0).to(torch.bfloat16))
        lses.append(lse)
    return torch.stack(outputs), torch.stack(lses)


def assert_arithmetic_reference(tc, out, lse, expected, expected_lse):
    error = out.double() - expected.double()
    rrms = error.square().mean(-1).sqrt() / expected.double().square().mean(-1).sqrt().clamp_min(1e-3)
    tc.assertLessEqual(float(rrms.max()), 0.005)
    torch.testing.assert_close(out.double(), expected.double(), atol=0.002, rtol=0.005)
    torch.testing.assert_close(lse.double(), expected_lse, atol=1e-5, rtol=1e-6)


def _tilelang():
    try:
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd
    except Exception:  # noqa: BLE001 - tilelang is optional for this test
        return None
    return tilelang_sparse_fwd


@unittest.skipUnless(_gfx950(), "gfx950 only")
class TestSparseMlaFp8Hip(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.attention.dsa.hip_gfx950 import sparse_mla_fp8

        cls.ops = sparse_mla_fp8
        cls.ops.module()
        fn = _tilelang()
        cls.tilelang = staticmethod(fn) if fn is not None else None

    def test_quantized_reference(self):
        for n in [1, 2, 3, 6, 7, 8, 12, 13, 21, 22, 23, 24, 42, 43, 64, 65, 128, 129, 2048, 4096]:
            with self.subTest(n=n):
                q, kv, ids = inputs(n)
                out, lse = self.ops.sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, return_lse=True)
                assert_reference(self, out, lse, q, kv, ids)
                control, control_lse = self.ops.sparse_mla_fp8_fwd(q.to(FP8), kv, ids, 1 / 16, return_lse=True)
                self.assertTrue(torch.equal(out, control), "FP8-q entry differs from the BF16 one")
                self.assertTrue(torch.equal(lse, control_lse))

    def test_capacity_stride_scale(self):
        """Strided q rows, the 4-D kv / 3-D index views, decoding scales != 1."""
        for n in [1, 8, 13, 24, 65, 129]:
            for capacity in CAPACITIES:
                for padded in (False, True):
                    with self.subTest(n=n, capacity=capacity, padded=padded):
                        q, kv, ids = inputs(n, capacity, strided=True, padded=padded)
                        kwargs = {"q_scale": 0.5, "kv_scale": 1.25, "return_lse": True}
                        out, lse = self.ops.sparse_mla_fp8_fwd(q, kv.view(-1, 1, 1, 512), ids[:, None, :], 1 / 16, **kwargs)
                        assert_reference(self, out, lse, q, kv, ids, q_scale=0.5, kv_scale=1.25)
                        control, control_lse = self.ops.sparse_mla_fp8_fwd((q.float() / 0.5).to(FP8), kv, ids, 1 / 16, **kwargs)
                        self.assertTrue(torch.equal(out, control))
                        self.assertTrue(torch.equal(lse, control_lse))

    def test_changed_graph(self):
        """A captured call replays on rewritten inputs, including masked tails."""
        for n in [1, 8, 13, 24, 65, 129, 2048]:
            with self.subTest(n=n):
                q, kv, ids = inputs(n)
                workspace = self.ops.workspace(n, q.device)
                out = torch.empty_like(q)
                lse = torch.empty(q.shape[:2], dtype=torch.float32, device=DEV)
                kwargs = {"out": out, "lse": lse, "return_lse": True, "workspace": workspace}
                self.ops.sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, **kwargs)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured, captured_lse = self.ops.sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, **kwargs)
                for replay in range(4):
                    q.normal_()
                    kv.copy_(torch.randn(kv.shape, device=DEV, dtype=torch.bfloat16).to(kv.dtype))
                    ids.random_(0, len(kv))
                    if replay == 0:
                        ids[:, 2051:] = -1
                    elif replay == 1:
                        ids[:, :128] = len(kv) - 1
                    elif replay == 2:
                        ids[:, 3::11] = -1
                    else:
                        ids.fill_(-1)
                    out.fill_(math.nan)
                    lse.fill_(math.nan)
                    for t in workspace:
                        t.fill_(math.nan)
                    graph.replay()
                    torch.cuda.synchronize()
                    expected, expected_lse = self.ops.sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, return_lse=True)
                    self.assertTrue(torch.equal(captured, expected))
                    self.assertTrue(torch.equal(captured_lse, expected_lse))
                    assert_reference(self, out, lse, q, kv, ids)

    def test_large_query_chunk_equivalence(self):
        """Every row agrees with the 2048-row launches of the same inputs."""
        for n in [4096, 16384]:
            with self.subTest(n=n):
                q, kv, ids = inputs(n, 2176, strided=True)
                kwargs = {"q_scale": 0.5, "kv_scale": 1.25, "return_lse": True}
                out, lse = self.ops.sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, **kwargs)
                assert_reference(self, out, lse, q, kv, ids, q_scale=0.5, kv_scale=1.25, all_rows=True)
                for start in range(0, n, 2048):
                    chunk_out, chunk_lse = self.ops.sparse_mla_fp8_fwd(
                        q[start : start + 2048], kv, ids[start : start + 2048], 1 / 16, **kwargs
                    )
                    self.assertTrue(torch.equal(out[start : start + 2048], chunk_out))
                    self.assertTrue(torch.equal(lse[start : start + 2048], chunk_lse))

    def test_arithmetic_reference_valid_tail(self):
        for n in DISPATCH_QUERIES:
            for capacity in (2112, 2176):
                for tail_only in (False, True):
                    with self.subTest(n=n, capacity=capacity, tail_only=tail_only):
                        q, kv, ids = arithmetic_inputs(n, capacity, seed=38119)
                        ids[:, 2051:] = torch.randint(len(kv), (n, capacity - 2051), device=DEV, dtype=torch.int32)
                        if tail_only:
                            ids[:, :2051] = -1
                        kwargs = {"q_scale": 0.5, "kv_scale": 1.25, "return_lse": True}
                        out, lse = self.ops.sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, **kwargs)
                        expected, expected_lse = arithmetic_reference(self.ops.split_count, q, kv, ids, q_scale=0.5, kv_scale=1.25)
                        assert_arithmetic_reference(self, out, lse, expected, expected_lse)
                        assert_reference(self, out, lse, q, kv, ids, q_scale=0.5, kv_scale=1.25)

    def test_arithmetic_gate_rejects_small_error(self):
        q, kv, ids = arithmetic_inputs(8, 2176, seed=91737)
        expected, expected_lse = arithmetic_reference(self.ops.split_count, q, kv, ids)
        out, lse = self.ops.sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, return_lse=True)
        assert_arithmetic_reference(self, out, lse, expected, expected_lse)
        with self.assertRaises(AssertionError):
            assert_arithmetic_reference(self, out.float() * 1.02, lse, expected, expected_lse)

    def test_matches_tilelang(self):
        """The DSA kernel sglang runs on this cache today (q cast to fp8, no
        scales): same top-level answer within the FP8 P / BF16 split tolerance."""
        if self.tilelang is None:
            self.skipTest("tilelang_sparse_fwd not importable")
        for n in [8, 64, 129, 2048]:
            with self.subTest(n=n):
                q, kv, ids = inputs(n, 2112, padded=True)
                hip, _ = self.ops.sparse_mla_fp8_fwd(q, kv, ids, 1 / 16)
                tl_out = self.tilelang(q=q, kv=kv.view(-1, 1, 512), indices=ids.unsqueeze(1), sm_scale=1 / 16, d_v=512)
                tl_out = tl_out.reshape(hip.shape)
                torch.cuda.synchronize()
                error = (hip.double() - tl_out.double()).square().mean(-1).sqrt()
                ref = tl_out.double().square().mean(-1).sqrt().clamp_min(1e-3)
                self.assertLessEqual(float((error / ref).max()), 2 * TOLERANCE)

    def test_bad_scales_and_metadata(self):
        q, kv, ids = inputs(8, 1)
        for argument in ("q_scale", "kv_scale"):
            for scale in (0.0, -1.0, math.inf, math.nan):
                with self.assertRaisesRegex(Exception, "scale"):
                    self.ops.sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, **{argument: scale})
        big_q = torch.empty((16385, 64, 512), device=DEV, dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, "16384"):
            self.ops.sparse_mla_fp8_fwd(big_q, kv, torch.empty((16385, 0), device=DEV, dtype=torch.int32), 1 / 16)
        with self.assertRaisesRegex(Exception, "dtype|FP8|float8"):
            self.ops.sparse_mla_fp8_fwd(q, kv.to(torch.bfloat16), ids, 1 / 16)
        with self.assertRaisesRegex(Exception, "dtype"):
            self.ops.sparse_mla_fp8_fwd(q, kv, ids.long(), 1 / 16)
        with self.assertRaisesRegex(Exception, "overlap"):
            self.ops.sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, out=q)
        unaligned = torch.empty((q.numel() + 1,), device=DEV, dtype=q.dtype)[1:].view(q.shape)
        with self.assertRaisesRegex(Exception, "align"):
            self.ops.sparse_mla_fp8_fwd(unaligned, kv, ids, 1 / 16)
        with self.assertRaisesRegex(Exception, "shape"):
            self.ops.sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, workspace=self.ops.workspace(1, DEV))
        with self.assertRaisesRegex(Exception, "2176"):
            self.ops.sparse_mla_fp8_fwd(q, kv, torch.empty((8, 2177), device=DEV, dtype=torch.int32), 1 / 16)
        empty, empty_lse = self.ops.sparse_mla_fp8_fwd(q[:0], kv, ids[:0], 1 / 16, return_lse=True)
        self.assertEqual(tuple(empty.shape), (0, 64, 512))
        self.assertEqual(tuple(empty_lse.shape), (0, 64))


if __name__ == "__main__":
    unittest.main()
