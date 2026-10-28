/// GLM-5.3-Flash sparse MLA attention core on gfx950: FP8 latent K/V pool,
/// BF16 query converted to FP8 inside the kernel, BF16 output (H64 / D512, no
/// appended RoPE). Same contract as aiter's sparse_mla_fp8_fwd:
///
///   O[q] = softmax(scale * Q[q] . KV[ids[q]]^T) . KV[ids[q]]
///
/// kv_buffer is the flat FP8 [slots, 512] pool, kv_indices [Q, K] holds
/// physical slot ids (negative / out-of-range masked, K <= 2176, Q <= 16384);
/// scores and values are decoded with q_scale / kv_scale; P is FP8 with scale
/// 1/256; accumulation, softmax and LSE are FP32; split partials and the
/// output are BF16. The kernels (sparse_mla_fp8/*.cu, verbatim from aiter's
/// csrc/kernels/mla/fp8) are four translation units: the loader builds this
/// header once per SGL_SPARSE_MLA_UNIT value (one .cu each) and once without
/// it for the tvm-ffi entry points below.
///
/// Loaded by kernels/ops/attention/dsa/hip_gfx950/sparse_mla_fp8.py.

#pragma once

#ifndef USE_ROCM
#error "sparse_mla_fp8.cuh targets gfx950"
#endif

#if defined(SGL_SPARSE_MLA_UNIT)
#if SGL_SPARSE_MLA_UNIT == 1
#include "sparse_mla_fp8/small.cu"
#elif SGL_SPARSE_MLA_UNIT == 2
#include "sparse_mla_fp8/prefetch.cu"
#elif SGL_SPARSE_MLA_UNIT == 3
#include "sparse_mla_fp8/rotating.cu"
#elif SGL_SPARSE_MLA_UNIT == 4
#include "sparse_mla_fp8/streaming.cu"
#else
#error "SGL_SPARSE_MLA_UNIT must be 1..4"
#endif
#else  // the entry-point unit

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <climits>
#include <cmath>
#include <cstdint>

#include "sparse_mla_fp8/params.cuh"

namespace sglang {
namespace dsa_gfx950::sparse_mla {

using host::RuntimeCheck;
using tvm::ffi::TensorView;

inline bool is_dtype(DLDataType t, uint8_t code, uint8_t bits) {
  return t.lanes == 1 && t.code == code && t.bits == bits;
}
inline bool is_bf16(DLDataType t) { return is_dtype(t, kDLBfloat, 16); }
inline bool is_fp8_e4m3fn(DLDataType t) { return is_dtype(t, kDLFloat8_e4m3fn, 8); }
inline bool is_f32(DLDataType t) { return is_dtype(t, kDLFloat, 32); }
inline bool is_i32(DLDataType t) { return is_dtype(t, kDLInt, 32); }

inline bool contiguous(const TensorView& t) {
  if (t.numel() == 0) return true;  // torch gives (1, 1) strides to [8, 0]
  int64_t expect = 1;
  for (int d = t.ndim() - 1; d >= 0; --d) {
    if (t.size(d) != 1 && t.stride(d) != expect) return false;
    expect *= t.size(d);
  }
  return true;
}

inline int64_t elem_bytes(DLDataType t) { return t.bits / 8; }

inline void check_aligned(const TensorView& t, const char* name, size_t alignment) {
  RuntimeCheck(!t.numel() || reinterpret_cast<uintptr_t>(t.data_ptr()) % alignment == 0, name,
               ": unaligned storage");
}

inline uintptr_t end_address(const TensorView& t) {
  if (!t.numel()) return 0;
  int64_t span = 1;
  for (int d = 0; d < t.ndim(); ++d) span += (t.size(d) - 1) * t.stride(d);
  return reinterpret_cast<uintptr_t>(t.data_ptr()) + span * elem_bytes(t.dtype());
}

inline void check_no_overlap(const TensorView& a, const TensorView& b) {
  if (!a.numel() || !b.numel()) return;
  RuntimeCheck(reinterpret_cast<uintptr_t>(a.data_ptr()) >= end_address(b) ||
                   reinterpret_cast<uintptr_t>(b.data_ptr()) >= end_address(a),
               "sparse_mla_fp8: output/workspace storage must not overlap");
}

inline void forward(const TensorView& q, const TensorView& kv_buffer, const TensorView& kv_indices,
                    const TensorView& out, const TensorView* lse, const TensorView& partial_o,
                    const TensorView& partial_lse, double softmax_scale, double q_scale, double kv_scale) {
  const DLDevice device = q.device();
  for (double scale : {softmax_scale, q_scale, kv_scale})
    RuntimeCheck(std::isfinite(scale) && scale > 0, "scales must be finite and positive");
  RuntimeCheck(is_bf16(q.dtype()) || is_fp8_e4m3fn(q.dtype()), "q must be BF16 or FP8 (float8_e4m3fn)");
  RuntimeCheck(q.ndim() == 3 && q.size(1) == 64 && q.size(2) == 512, "expected q=[Q,64,512]");
  const int64_t queries = q.size(0);
  RuntimeCheck(queries <= 16384, "Q must be <= 16384");
  RuntimeCheck(q.stride(2) == 1 && q.stride(1) >= 512 && q.stride(0) >= 63 * q.stride(1) + 512,
               "q must have non-overlapping, unit-inner-stride rows");
  const int64_t qb = elem_bytes(q.dtype());
  RuntimeCheck((q.stride(0) * qb) % 16 == 0 && (q.stride(1) * qb) % 16 == 0, "q rows must be aligned to 16 bytes");
  check_aligned(q, "q", 16);
  RuntimeCheck(is_fp8_e4m3fn(kv_buffer.dtype()), "kv_buffer must be float8_e4m3fn (dtype)");
  RuntimeCheck(kv_buffer.ndim() == 2 && kv_buffer.size(1) == 512 && kv_buffer.size(0) > 0 &&
                   kv_buffer.size(0) <= INT_MAX && contiguous(kv_buffer),
               "expected contiguous kv_buffer=[slots,512]");
  check_aligned(kv_buffer, "kv_buffer", 16);
  RuntimeCheck(is_i32(kv_indices.dtype()), "kv_indices must be int32 (dtype)");
  RuntimeCheck(kv_indices.ndim() == 2 && kv_indices.size(0) == queries && kv_indices.size(1) <= 2176 &&
                   contiguous(kv_indices),
               "expected contiguous kv_indices=[Q,K], K <= 2176");
  check_aligned(kv_indices, "kv_indices", 4);
  RuntimeCheck(is_bf16(out.dtype()) && out.ndim() == 3 && out.size(0) == queries && out.size(1) == 64 &&
                   out.size(2) == 512 && contiguous(out),
               "out must be a contiguous bf16 [Q,64,512] tensor (unexpected shape)");
  check_aligned(out, "out", 16);
  if (lse) {
    RuntimeCheck(is_f32(lse->dtype()) && lse->ndim() == 2 && lse->size(0) == queries && lse->size(1) == 64 &&
                     contiguous(*lse),
                 "lse must be a contiguous fp32 [Q,64] tensor (unexpected shape)");
    check_aligned(*lse, "lse", 4);
  }
  const int splits = sparse_mla_fp8::split_count(static_cast<int>(queries));
  const int64_t rows = splits > 1 ? queries * splits * 64 : 0;
  RuntimeCheck(is_bf16(partial_o.dtype()) && partial_o.ndim() == 1 && partial_o.size(0) == rows * 512 &&
                   is_f32(partial_lse.dtype()) && partial_lse.ndim() == 1 && partial_lse.size(0) == rows,
               "workspace has an unexpected shape: expected partial_o [", rows * 512, "] bf16 and partial_lse [",
               rows, "] fp32 for Q=", queries);
  RuntimeCheck(contiguous(partial_o) && contiguous(partial_lse), "workspace must be contiguous");
  check_aligned(partial_o, "partial_o", 16);
  check_aligned(partial_lse, "partial_lse", 4);
  {
    const TensorView* writes[4] = {&out, &partial_o, &partial_lse, lse};
    const int n_writes = lse ? 4 : 3;
    for (int i = 0; i < n_writes; ++i) {
      for (const TensorView* input : {&q, &kv_buffer, &kv_indices}) check_no_overlap(*writes[i], *input);
      for (int j = 0; j < i; ++j) check_no_overlap(*writes[i], *writes[j]);
    }
  }
  if (!queries) return;

  sparse_mla_fp8::Params p{};
  p.q = static_cast<const uint8_t*>(q.data_ptr());
  p.kv = static_cast<const uint8_t*>(kv_buffer.data_ptr());
  p.indices = static_cast<const int32_t*>(kv_indices.data_ptr());
  p.partial_o = static_cast<uint16_t*>(partial_o.data_ptr());
  p.partial_lse = static_cast<float*>(partial_lse.data_ptr());
  p.out = static_cast<uint8_t*>(out.data_ptr());
  p.lse = lse ? static_cast<float*>(lse->data_ptr()) : nullptr;
  p.queries = static_cast<int>(queries);
  p.slots = static_cast<int>(kv_buffer.size(0));
  p.capacity = static_cast<int>(kv_indices.size(1));
  p.q_scale = static_cast<float>(q_scale);
  p.kv_scale = static_cast<float>(kv_scale);
  p.attention_scale = static_cast<float>(softmax_scale);
  p.q_stride_row = q.stride(0);
  p.q_stride_head = q.stride(1);
  p.bf16_q = is_bf16(q.dtype());

  int prev_device = -1;
  (void)hipGetDevice(&prev_device);
  if (prev_device != device.device_id) (void)hipSetDevice(device.device_id);
  const hipStream_t stream = static_cast<hipStream_t>(::TVMFFIEnvGetStream(device.device_type, device.device_id));
  hipError_t status;
  if (queries <= 6)
    status = sparse_mla_fp8::launch_small(p, stream);
  else if (queries <= 12)
    status = sparse_mla_fp8::launch_prefetch(p, stream);
  else if (queries <= 64)
    status = sparse_mla_fp8::launch_rotating(p, stream);
  else
    status = sparse_mla_fp8::launch_streaming(p, stream);
  if (prev_device >= 0 && prev_device != device.device_id) (void)hipSetDevice(prev_device);
  RuntimeCheck(status == hipSuccess, "sparse_mla_fp8_fwd launch failed: ", hipGetErrorString(status));
}

}  // namespace dsa_gfx950::sparse_mla

struct SparseMlaFp8Kernel {
  // Writes `out`; partial_o / partial_lse is the opaque split workspace of
  // kernels/ops/attention/dsa/hip_gfx950/sparse_mla_fp8.py::workspace.
  static void fwd(const tvm::ffi::TensorView q, const tvm::ffi::TensorView kv_buffer,
                  const tvm::ffi::TensorView kv_indices, const tvm::ffi::TensorView out,
                  const tvm::ffi::TensorView partial_o, const tvm::ffi::TensorView partial_lse,
                  double softmax_scale, double q_scale, double kv_scale) {
    dsa_gfx950::sparse_mla::forward(q, kv_buffer, kv_indices, out, nullptr, partial_o, partial_lse, softmax_scale,
                                    q_scale, kv_scale);
  }
  // Same, also writing the natural-log LSE [Q, 64] fp32.
  static void fwd_lse(const tvm::ffi::TensorView q, const tvm::ffi::TensorView kv_buffer,
                      const tvm::ffi::TensorView kv_indices, const tvm::ffi::TensorView out,
                      const tvm::ffi::TensorView lse, const tvm::ffi::TensorView partial_o,
                      const tvm::ffi::TensorView partial_lse, double softmax_scale, double q_scale,
                      double kv_scale) {
    dsa_gfx950::sparse_mla::forward(q, kv_buffer, kv_indices, out, &lse, partial_o, partial_lse, softmax_scale,
                                    q_scale, kv_scale);
  }
};

}  // namespace sglang

#endif  // SGL_SPARSE_MLA_UNIT
