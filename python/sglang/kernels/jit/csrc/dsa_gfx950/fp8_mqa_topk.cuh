// GLM-5.3-Flash DSA k-pool indexer, prefill: FP8 MQA logits + the pooled top-k without fp32 logits (Step 0).
//
// The [rows, keys] fp32 logits are never stored. A 1/32 strided sample of each
// row's keys gives it a threshold tau (the sample's ~e + 5 sqrt(e) + 8-th largest
// logit, e = expected samples above the k-th largest), the scan -- the dense
// logits kernel's MFMA pipeline -- stores only the keys scoring >= tau, and a
// select ranks those candidates. A row whose candidates cannot prove its exact
// top-k (fewer than min(k, keys) of them, or a full candidate segment) is
// recomputed exactly by the fallback kernels, which are no-ops otherwise.
// Launches: sample scan, tau, scan, select and 3 fallback kernels; the caller
// slices the sample keys.
//
// fp8_mqa_topk/ is moreh-dev/rocm-aiter@1e9edc04a csrc/kernels/fp8_mqa_topk,
// kept byte for byte: its hand-placed vmcnt waits and inline asm were checked on
// the ISA built from that code with the flags the Python loader passes.
#pragma once

#ifndef USE_ROCM
#error "fp8_mqa_topk.cuh targets gfx950 (v_mfma_f32_32x32x64_f8f6f4, v_permlane32_swap)"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include "fp8_mqa_topk/fmt_scan.cuh"
#include "fp8_mqa_topk/fmt_select.cuh"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <queue>
#include <vector>

namespace sglang {

namespace dsa_gfx950::mqa_topk {

// The validated configuration; the Python gate refuses anything else.
inline constexpr int kTopK = 512;
inline constexpr int kMaxSplit = 16;
inline constexpr int kScanRows = 32;                // rows per scan workgroup
inline constexpr int kScanThreads = 512;            // 8 waves
inline constexpr int kRowThreads = 256;             // tau / select / fallback: one wave per row
inline constexpr int kFallbackRows = 2 * fmt::FBR;  // scratch rows: FBR batched + 64 one-wave rows
inline constexpr int kAssumedCus = 256;             // MI355X; the grid model below was fit there

// Makespan model of one scan launch, copied from the rocm-aiter launcher (fit on
// MI355X): a workgroup costs GA + GBM per masked stage + GBC per clean stage (us),
// workgroups are dispatched round-robin over 8 XCDs in blockIdx order and greedily
// onto each XCD's CUs, one per CU. It assumes one causal k-pool request (kpool 4);
// other layouts only get a worse grid, never a wrong result.
inline constexpr double kGA = 9.87, kGBM = 2.962, kGBC = 1.975;

inline double model_makespan(int M, int N, int split, int rev, int cus_per_xcd) {
  const int rb = FMT_CDIV(M, kScanRows);
  std::vector<std::priority_queue<double, std::vector<double>, std::greater<double>>> xcd(8);
  for (auto& q : xcd) {
    for (int i = 0; i < cus_per_xcd; ++i)
      q.push(0.0);
  }
  double makespan = 0.0;
  int idx = 0;
  for (int y = 0; y < split; ++y) {
    for (int x = 0; x < rb; ++x, ++idx) {
      const int bx = rev ? rb - 1 - x : x;
      const int r0 = bx * kScanRows, r1 = std::min(M, r0 + kScanRows);
      auto ke = [&](int m) { return std::min((4 * N - M + m + 1) / 4, N); };
      const int hi = ke(r1 - 1), mn = ke(r0);
      const int n = FMT_CDIV(hi, 64);
      const int tc1 = (r0 + kScanRows <= M) ? std::min(mn / 64, n) : 0;
      auto share = [&](int n_) {
        const int per = FMT_CDIV(n_, split);
        const int b = y * per;
        return std::max(0, std::min(n_, b + per) - b);
      };
      auto& q = xcd[idx % 8];
      const double t = q.top() + kGA + kGBM * share(n - tc1) + kGBC * share(tc1);
      q.pop();
      q.push(t);
      makespan = std::max(makespan, t);
    }
  }
  return makespan;
}

// (split, reverse row order) minimizing the modelled makespan, packed split | rev << 8.
inline int64_t choose_grid(int M, int N) {
  const int cus_per_xcd = std::max(1, kAssumedCus / 8);
  double best = 1e300;
  int split = 1, rev = 1;
  for (int s = 1; s <= kMaxSplit; ++s) {
    for (int r = 0; r < 2; ++r) {
      const double t = model_makespan(M, N, s, r, cus_per_xcd);
      if (t < best * 0.999) {
        best = t;
        split = s;
        rev = r;
      }
    }
  }
  return static_cast<int64_t>(split) | (static_cast<int64_t>(rev) << 8);
}

inline auto is_fp8_e4m3fn(DLDataType dtype) -> bool {
  return dtype.lanes == 1 && dtype.bits == 8 && dtype.code == kDLFloat8_e4m3fn;
}

using fp8p = const fmt::fp8*;

template <int MODE>
void launch_scan(
    const dim3 grid,
    const bool rev,
    const hipStream_t stream,
    fp8p q,
    fp8p k,
    const float* k_scale,
    const float* weights,
    const int* starts,
    const int* ends,
    const int rows,
    const int keys,
    float* dense_out,
    const int dense_cols,
    const float* tau,
    unsigned long long* candidates,
    int* candidate_counts,
    const int candidate_cap) {
  const int split = static_cast<int>(grid.y);
  // Two direct launches rather than one through a function pointer: the
  // instantiations are the ones the rocm-aiter launcher uses.
  if (rev) {
    fmt::scan_kernel<8, 2, 4, MODE, true><<<grid, kScanThreads, 0, stream>>>(
        q,
        k,
        k_scale,
        weights,
        starts,
        ends,
        rows,
        keys,
        split,
        dense_out,
        dense_cols,
        1,
        tau,
        candidates,
        candidate_counts,
        candidate_cap,
        nullptr,
        0,
        nullptr);
  } else {
    fmt::scan_kernel<8, 2, 4, MODE, false><<<grid, kScanThreads, 0, stream>>>(
        q,
        k,
        k_scale,
        weights,
        starts,
        ends,
        rows,
        keys,
        split,
        dense_out,
        dense_cols,
        1,
        tau,
        candidates,
        candidate_counts,
        candidate_cap,
        nullptr,
        0,
        nullptr);
  }
}

}  // namespace dsa_gfx950::mqa_topk

struct Fp8MqaTopkKernel {
  // Scan grid for `rows` x `keys`, packed split | reverse << 8.
  static int64_t plan(int64_t rows, int64_t keys) {
    return dsa_gfx950::mqa_topk::choose_grid(static_cast<int>(rows), static_cast<int>(keys));
  }

  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView k_scale,
      const tvm::ffi::TensorView weights,
      const tvm::ffi::TensorView starts,
      const tvm::ffi::TensorView ends,
      const tvm::ffi::TensorView sample_k,
      const tvm::ffi::TensorView sample_k_scale,
      const tvm::ffi::TensorView sample_starts,
      const tvm::ffi::TensorView sample_ends,
      const tvm::ffi::TensorView sample_logits,
      const tvm::ffi::TensorView tau,
      const tvm::ffi::TensorView candidates,
      const tvm::ffi::TensorView candidate_counts,
      const tvm::ffi::TensorView flagged_rows,
      const tvm::ffi::TensorView flagged_count,
      const tvm::ffi::TensorView fallback_scratch,
      const tvm::ffi::TensorView out,
      const int64_t sample_plan,
      const int64_t scan_plan,
      const int64_t sample_stride,
      const int64_t candidate_cap,
      const int64_t force_fallback,
      const tvm::ffi::Optional<tvm::ffi::TensorView> fallback_row_count) {
    using namespace host;
    namespace impl = dsa_gfx950::mqa_topk;

    auto M = SymbolicSize{"rows"};
    auto N = SymbolicSize{"keys"};
    auto NS = SymbolicSize{"sample_keys"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();
    // e4m3fn has no DLDataTypeTrait on ROCm; one symbolic dtype across q and the
    // keys is the "same fp8 dtype" check.
    auto fp8 = SymbolicDType{};

    // No with_strides: every tensor here must be contiguous.
    TensorMatcher({M, int64_t{fmt::NUM_HEADS}, int64_t{fmt::HEAD_SIZE}}).with_dtype(fp8).with_device(device).verify(q);
    TensorMatcher({N, int64_t{fmt::HEAD_SIZE}}).with_dtype(fp8).with_device(device).verify(k);
    TensorMatcher({NS, int64_t{fmt::HEAD_SIZE}}).with_dtype(fp8).with_device(device).verify(sample_k);
    RuntimeCheck(impl::is_fp8_e4m3fn(fp8.unwrap()), "q and the keys must be fp8 e4m3fn; they are read as raw bytes");
    TensorMatcher({N}).with_dtype<float>().with_device(device).verify(k_scale);
    TensorMatcher({NS}).with_dtype<float>().with_device(device).verify(sample_k_scale);
    TensorMatcher({M, int64_t{fmt::NUM_HEADS}}).with_dtype<float>().with_device(device).verify(weights);
    for (const auto& t : {starts, ends, sample_starts, sample_ends, flagged_rows}) {
      TensorMatcher({M}).with_dtype<int32_t>().with_device(device).verify(t);
    }
    TensorMatcher({M, NS}).with_dtype<float>().with_device(device).verify(sample_logits);
    TensorMatcher({M}).with_dtype<float>().with_device(device).verify(tau);
    TensorMatcher({-1}).with_dtype<int64_t>().with_device(device).verify(candidates);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(candidate_counts);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(flagged_count);
    TensorMatcher({-1}).with_dtype<float>().with_device(device).verify(fallback_scratch);
    TensorMatcher({M, int64_t{impl::kTopK}}).with_dtype<int32_t>().with_device(device).verify(out);
    if (fallback_row_count.has_value()) {
      TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(fallback_row_count.value());
    }

    const int rows = static_cast<int>(M.unwrap());
    const int keys = static_cast<int>(N.unwrap());
    const int sample_keys = static_cast<int>(NS.unwrap());
    const int split = static_cast<int>(scan_plan & 0xff), rev = static_cast<int>((scan_plan >> 8) & 1);
    const int sample_split = static_cast<int>(sample_plan & 0xff);
    const int sample_rev = static_cast<int>((sample_plan >> 8) & 1);
    const int cap = static_cast<int>(candidate_cap);
    RuntimeCheck(rows > 0 && keys > 0 && sample_keys > 0, "empty input: rows ", rows, ", keys ", keys);
    RuntimeCheck(split >= 1 && split <= impl::kMaxSplit, "scan split ", split, " outside [1, ", impl::kMaxSplit, "]");
    RuntimeCheck(sample_split >= 1 && sample_split <= impl::kMaxSplit, "sample split ", sample_split, " out of range");
    RuntimeCheck(sample_stride >= 1 && sample_keys == FMT_CDIV(keys, sample_stride), "sample_k must be keys[::stride]");
    // Buffer descriptors address the keys with 32-bit byte offsets.
    RuntimeCheck(static_cast<int64_t>(keys) * fmt::HEAD_SIZE < (int64_t{1} << 31), "too many keys: ", keys);
    const int64_t rows_padded = rows + (rows & 1);
    RuntimeCheck(
        candidates.size(0) >= split * rows_padded * cap,
        "candidates holds ",
        candidates.size(0),
        " slots, need ",
        split * rows_padded * cap);
    RuntimeCheck(candidate_counts.size(0) >= int64_t{split} * rows, "candidate_counts too small");
    RuntimeCheck(
        fallback_scratch.size(0) >= int64_t{impl::kFallbackRows} * keys,
        "fallback_scratch needs ",
        impl::kFallbackRows,
        " rows of keys");

    const auto stream = LaunchKernel::resolve_device(device.unwrap());
    const auto qp = static_cast<impl::fp8p>(q.data_ptr());
    const auto kp = static_cast<impl::fp8p>(k.data_ptr());
    const auto scale = static_cast<const float*>(k_scale.data_ptr());
    const auto w = static_cast<const float*>(weights.data_ptr());
    const auto ks = static_cast<const int*>(starts.data_ptr());
    const auto ke = static_cast<const int*>(ends.data_ptr());
    const auto tau_p = static_cast<float*>(tau.data_ptr());
    const auto cand = static_cast<unsigned long long*>(candidates.data_ptr());
    const auto cnt = static_cast<int*>(candidate_counts.data_ptr());
    const auto flag = static_cast<int*>(flagged_rows.data_ptr());
    const auto fbc = static_cast<int*>(flagged_count.data_ptr());
    const auto scratch = static_cast<float*>(fallback_scratch.data_ptr());
    const auto outp = static_cast<int*>(out.data_ptr());
    const auto nfb =
        fallback_row_count.has_value() ? static_cast<int*>(fallback_row_count.value().data_ptr()) : nullptr;
    const int row_blocks = FMT_CDIV(rows, impl::kScanRows);
    const int row_waves = FMT_CDIV(rows, 4);

    // Sample scan: dense fp32 logits of keys j * stride.
    impl::launch_scan<fmt::MODE_DENSE>(
        dim3(row_blocks, sample_split),
        sample_rev,
        stream,
        qp,
        static_cast<impl::fp8p>(sample_k.data_ptr()),
        static_cast<const float*>(sample_k_scale.data_ptr()),
        w,
        static_cast<const int*>(sample_starts.data_ptr()),
        static_cast<const int*>(sample_ends.data_ptr()),
        rows,
        sample_keys,
        static_cast<float*>(sample_logits.data_ptr()),
        sample_keys,
        nullptr,
        nullptr,
        nullptr,
        0);
    // tau per row; rows of at most kTopK keys take every key. Also resets *fbc.
    fmt::tau_kernel<<<row_waves, impl::kRowThreads, 0, stream>>>(
        static_cast<const float*>(sample_logits.data_ptr()),
        sample_keys,
        static_cast<int>(sample_stride),
        ks,
        ke,
        rows,
        keys,
        impl::kTopK,
        impl::kTopK,
        tau_p,
        fbc);
    if (force_fallback) {
      // Test hook: tau = +inf leaves no candidates, so every row takes the exact fallback.
      RuntimeCheck(
          hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(tau_p), static_cast<int>(0x7f800000), rows, stream) ==
              hipSuccess,
          "tau fill failed");
    }
    impl::launch_scan<fmt::MODE_EMIT>(
        dim3(row_blocks, split), rev, stream, qp, kp, scale, w, ks, ke, rows, keys, nullptr, 0, tau_p, cand, cnt, cap);
    fmt::select_kernel<32><<<row_waves, impl::kRowThreads, 0, stream>>>(
        cand, cnt, split, cap, ks, ke, rows, keys, impl::kTopK, outp, flag, fbc);
    fmt::fb_compute<<<64, impl::kRowThreads, 0, stream>>>(flag, fbc, ks, ke, rows, keys, scratch, qp, kp, scale, w);
    fmt::fb_select<<<fmt::FBR / 4, impl::kRowThreads, 0, stream>>>(
        flag, fbc, ks, ke, rows, keys, impl::kTopK, outp, scratch, nfb);
    fmt::fallback_kernel<<<16, impl::kRowThreads, 0, stream>>>(
        flag, fbc, ks, ke, rows, keys, impl::kTopK, outp, scratch, qp, kp, scale, w, nfb);
    const auto err = hipGetLastError();
    RuntimeCheck(err == hipSuccess, "fp8_mqa_topk launch failed: ", hipGetErrorString(err));
  }
};

}  // namespace sglang
