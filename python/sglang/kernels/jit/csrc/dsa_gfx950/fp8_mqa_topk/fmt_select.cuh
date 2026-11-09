// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Fused FP8 MQA logits + top-k (gfx950): threshold, select and exact fallback (port of Nam's fuse48.hip, MV-5041
// section 18) with two changes:
//   * tau is the exact target-th largest fp32 SAMPLE logit (Nam: lower edge of a 16-bit bucket), 32-bit keys;
//   * the fallback recompute is bit-identical to v35wes (plain MFMA, scalar even/odd fma chains), not to v35/v35a.
//
//   tau  (one wave per row): e = K * ns / nvalid samples expected above the K-th logit, target = ceil(e + 5 sqrt(e)) + 8,
//        tau = target-th largest sample; -inf when nvalid <= CAP or ns < target (every valid key is a candidate).
//   select (one wave per row): exact top-K over the fp32 candidates (<= 2048 in registers: bitwise binary search;
//        else 4 x 8-bit radix re-reading global); a row whose segment overflowed or that has fewer than
//        min(K, nvalid) candidates is appended to the fallback list.
//   fallback: flagged rows only; exact logits of the whole row recomputed into scratch, then the same radix select.
// Output = column - ks (vLLM convention), -1 padding.
#pragma once
#include "fmt_common.cuh"

namespace fmt {

// ------------------------------------------------------------------------------------------------- wave helpers
// wave-level: bin b of hist[256] with above(b) < target <= above(b) + h[b]
// inclusive prefix sum over the wave with DPP (row_shr 1/2/4/8 inside each 16-lane row, then row_bcast 15 / 31 across
// rows; gfx9). VALU only - replaces a chain of 6 dependent ds_bpermute (__shfl_up) round trips.
__device__ __forceinline__ int wave_incl_scan_dpp(int x) {
    x += __builtin_amdgcn_update_dpp(0, x, 0x111, 0xf, 0xf, false);   // row_shr:1
    x += __builtin_amdgcn_update_dpp(0, x, 0x112, 0xf, 0xf, false);   // row_shr:2
    x += __builtin_amdgcn_update_dpp(0, x, 0x114, 0xf, 0xf, false);   // row_shr:4
    x += __builtin_amdgcn_update_dpp(0, x, 0x118, 0xf, 0xf, false);   // row_shr:8
    x += __builtin_amdgcn_update_dpp(0, x, 0x142, 0xa, 0xf, false);   // row_bcast:15 into rows 1, 3
    x += __builtin_amdgcn_update_dpp(0, x, 0x143, 0xc, 0xf, false);   // row_bcast:31 into rows 2, 3
    return x;
}
// bin of the target-th largest key in a 256-bin histogram (lane l owns bins 4l..4l+3); fab = keys in higher bins,
// fcnt = keys in the bin. Two versions, chosen per caller:
//  DPP = false: shuffle scan (ds_bpermute) - the LDS pipe runs in parallel with the VALU, faster in the
//    high-occupancy standalone tau / select kernels (step 0 tau phase 70.5 vs 97.0 us on 576 x 250000);
//  DPP = true (default): DPP scan + readlane broadcasts + one ds_read_b128 - fewer dependent round trips, faster at the fused
//    kernel's 2 waves / SIMD (in-kernel tau, wave_tau_multi). hist 16-byte aligned.
template <bool DPP = true>
__device__ __forceinline__ void wave_find256(const uint32_t* hist, int target, int lane, int& fbin, int& fab, int& fcnt) {
    const int lo = lane * 4;
    if constexpr (!DPP) {
        int s4 = 0;
        for (int b = lo; b < lo + 4; ++b) s4 += hist[b];
        const int incl = wave_incl_scan(s4, lane);
        int cum = __shfl(incl, WARP - 1) - incl, fb = -1, fa = 0, fc = 0;
        for (int b = lo + 3; b >= lo; --b) { const int h = hist[b]; if (cum < target && cum + h >= target) { fb = b; fa = cum; fc = h; } cum += h; }
        const uint64_t bm = __ballot(fb >= 0);
        const int src = bm ? __builtin_ctzll(bm) : 0;
        fbin = __shfl(fb, src); fab = __shfl(fa, src); fcnt = __shfl(fc, src);
    } else {
        const u32x4 h4 = *reinterpret_cast<const u32x4*>(&hist[lo]);
        const int incl = wave_incl_scan_dpp((int)(h4[0] + h4[1] + h4[2] + h4[3]));
        int cum = __builtin_amdgcn_readlane(incl, WARP - 1) - incl, fb = -1, fa = 0, fc = 0;
        #pragma unroll
        for (int k = 3; k >= 0; --k) {
            const int h = (int)h4[k];
            if (cum < target && cum + h >= target) { fb = lo + k; fa = cum; fc = h; }
            cum += h;
        }
        const uint64_t bm = __ballot(fb >= 0);
        const int src = bm ? __builtin_ctzll(bm) : 0;
        fbin = __builtin_amdgcn_readlane(fb, src); fab = __builtin_amdgcn_readlane(fa, src); fcnt = __builtin_amdgcn_readlane(fc, src);
    }
}

// radix over 32-bit keys delivered by each(f) (f(key, col) called per element, non-uniformly) -> key T of rank K
template <int NRD, bool DPP = true, typename Each>
__device__ __forceinline__ void wave_radix(Each&& each, uint32_t* hist, int K, int lane, uint32_t& T, int& above, int& cnt_eq) {
    uint32_t prefix = 0; above = 0; cnt_eq = 0;
    #pragma unroll
    for (int r = 0; r < NRD; ++r) {
        const int sh = 8 * (NRD - 1 - r);
        for (int b = lane; b < 256; b += WARP) hist[b] = 0;
        mem_fence_wave();
        const uint32_t pf = prefix;
        each([&](uint32_t k, int) { if (r == 0 || (uint32_t)((uint64_t)k >> (sh + 8)) == pf) atomicAdd(&hist[(k >> sh) & 255], 1u); });
        mem_fence_wave();
        int fb, fa, fc;
        wave_find256<DPP>(hist, K - above, lane, fb, fa, fc);
        prefix = (prefix << 8) | (uint32_t)fb; above += fa; cnt_eq = fc;
        mem_fence_wave();
    }
    T = prefix;
}
// wave sum of a per-lane int with DPP (VALU only, lane 63 ends with the total; gfx9 row_bcast), as a scalar
__device__ __forceinline__ int wave_sum_dpp(int x) {
    x += __builtin_amdgcn_update_dpp(0, x, 0x111, 0xf, 0xf, false);   // row_shr:1
    x += __builtin_amdgcn_update_dpp(0, x, 0x112, 0xf, 0xf, false);   // row_shr:2
    x += __builtin_amdgcn_update_dpp(0, x, 0x114, 0xf, 0xf, false);   // row_shr:4
    x += __builtin_amdgcn_update_dpp(0, x, 0x118, 0xf, 0xf, false);   // row_shr:8  -> lane 15 of each row = row sum
    x += __builtin_amdgcn_update_dpp(0, x, 0x142, 0xa, 0xf, false);   // row_bcast:15 into rows 1, 3
    x += __builtin_amdgcn_update_dpp(0, x, 0x143, 0xc, 0xf, false);   // row_bcast:31 into rows 2, 3
    return __builtin_amdgcn_readlane(x, 63);
}
// count of keys >= x over the wave (keys in registers rk[0..R), 0 = empty slot never counted since x >= 1).
// Per-lane VALU count (v_cmp + v_addc on VCC: VALU->VALU forwarding) + one DPP reduction. A ballot + s_bcnt1 per
// register serializes every compare through VCC -> SALU transfers (measured ~10x slower in the tau search).
template <int R>
__device__ __forceinline__ int wave_count_ge(const uint32_t (&rk)[R], uint32_t x) {
    int c = 0;
    #pragma unroll
    for (int j = 0; j < R; ++j) c += (rk[j] >= x) ? 1 : 0;
    return wave_sum_dpp(c);
}
// T = largest key with count(>= T) >= target (bitwise binary search, 32 iterations of compare + wave reduce)
template <int R>
__device__ __forceinline__ void wave_bsearch32(const uint32_t (&rk)[R], int target, uint32_t& T, int& above, int& cnt_eq) {
    uint32_t t = 0;
    #pragma unroll 1
    for (int b = 31; b >= 0; --b) { const uint32_t c = t | (1u << b); if (wave_count_ge(rk, c) >= target) t = c; }
    T = t;
    const int ge = wave_count_ge(rk, t);
    above = (t == 0xFFFFFFFFu) ? 0 : wave_count_ge(rk, t + 1);
    cnt_eq = ge - above;
}
// prefix-only search: largest T with low LOWBIT bits zero and count(>= T) >= target (32 - LOWBIT iterations).
// The float of key T is <= the target-th largest key's float: a conservative threshold.
template <int R, int LOWBIT>
__device__ __forceinline__ uint32_t wave_bsearch_prefix(const uint32_t (&rk)[R], int target) {
    uint32_t t = 0;
    #pragma unroll 1
    for (int b = 31; b >= LOWBIT; --b) { const uint32_t c = t | (1u << b); if (wave_count_ge(rk, c) >= target) t = c; }
    return t;
}

// emission over a uniform stream: each_u(g) must call g(valid, key, col) with ALL lanes, same trip count
template <typename EachU>
__device__ __forceinline__ void wave_emit(EachU&& each_u, uint32_t T, int above, int K, int ks, int lane, int* __restrict__ orow) {
    const int need = K - above;
    int pdef = 0, ptie = 0;
    each_u([&](bool v, uint32_t k, int col) {
        const bool gt = v && k > T, eq = v && k == T;
        const uint64_t mg = __ballot(gt), me = __ballot(eq);
        const int og = pdef + (int)__builtin_amdgcn_mbcnt_hi((uint32_t)(mg >> 32), __builtin_amdgcn_mbcnt_lo((uint32_t)mg, 0));
        const int oe = ptie + (int)__builtin_amdgcn_mbcnt_hi((uint32_t)(me >> 32), __builtin_amdgcn_mbcnt_lo((uint32_t)me, 0));
        if (gt) orow[og] = col - ks;
        if (eq && oe < need) orow[above + oe] = col - ks;
        pdef += __popcll(mg); ptie += __popcll(me);
    });
}

// wave min / max of a per-lane u32 with DPP (gfx9 row_bcast), result as a scalar
__device__ __forceinline__ uint32_t wave_max_u32(uint32_t x) {
    x = max(x, (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x111, 0xf, 0xf, false));
    x = max(x, (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x112, 0xf, 0xf, false));
    x = max(x, (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x114, 0xf, 0xf, false));
    x = max(x, (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x118, 0xf, 0xf, false));
    x = max(x, (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x142, 0xa, 0xf, false));
    x = max(x, (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x143, 0xc, 0xf, false));
    return (uint32_t)__builtin_amdgcn_readlane((int)x, 63);
}
__device__ __forceinline__ uint32_t wave_min_u32(uint32_t x) { return ~wave_max_u32(~x); }

// Relative two-pass radix threshold of one row (one wave): keys of the valid samples in rk (0 = empty slot).
// Pass 1 histograms the 8 bits just below the common prefix of [kmin, kmax], pass 2 the next 8 bits inside the bin
// holding the target-th largest key, so the bucket resolution is ~(kmax - kmin) / 2^16 whatever the score offset.
// Returns the lowest float of the final bucket (<= the target-th largest sample): conservative.
template <int R, bool DPP = true>
__device__ __forceinline__ float wave_tau_relradix(const uint32_t (&rk)[R], int target, uint32_t* hist, int lane) {
    uint32_t vmax = 0u, vmin = 0xFFFFFFFFu;
    #pragma unroll
    for (int j = 0; j < R; ++j) { vmax = max(vmax, rk[j]); vmin = min(vmin, rk[j] ? rk[j] : 0xFFFFFFFFu); }
    const uint32_t kmax = wave_max_u32(vmax), kmin = wave_min_u32(vmin);
    const uint32_t diff = kmax ^ kmin;
    if (diff == 0u) return __builtin_bit_cast(float, okey32_inv(kmax));          // every sample equal
    const int hb = 31 - __builtin_clz(diff);                                      // highest differing bit
    const int sh1 = max(hb - 7, 0), sh2 = max(sh1 - 8, 0);
    const uint32_t pre1 = (sh1 + 8 >= 32) ? 0u : (kmax >> (sh1 + 8));             // common prefix above pass-1 bits
    // pass 1
    for (int b = lane; b < 256; b += WARP) hist[b] = 0u;
    mem_fence_wave();
    #pragma unroll
    for (int j = 0; j < R; ++j) if (rk[j]) atomicAdd(&hist[(rk[j] >> sh1) & 255u], 1u);
    mem_fence_wave();
    int b1, a1, c1;
    wave_find256<DPP>(hist, target, lane, b1, a1, c1);
    uint32_t T = (pre1 << 8 | (uint32_t)b1) << sh1;
    if (sh1 > 0) {
        // pass 2: keys whose bits above sh1 equal T's
        mem_fence_wave();
        for (int b = lane; b < 256; b += WARP) hist[b] = 0u;
        mem_fence_wave();
        const uint32_t hiT = T >> sh1;
        const int w2 = sh1 - sh2;                                                 // 1..8 bits
        #pragma unroll
        for (int j = 0; j < R; ++j)
            if (rk[j] && (rk[j] >> sh1) == hiT) atomicAdd(&hist[(rk[j] >> sh2) & ((1u << w2) - 1u)], 1u);
        mem_fence_wave();
        int b2, a2, c2;
        wave_find256<DPP>(hist, target - a1, lane, b2, a2, c2);
        T |= (uint32_t)b2 << sh2;
    }
    return __builtin_bit_cast(float, okey32_inv(T));
}

// tau of a row with many samples (ns > 32 * 64, any ns; one wave): every lane keeps the T largest keys of its samples
// i = lane + 64 j (branch-free insertion network, loads issued B at a time), then the relative radix runs on the 64 T
// kept keys. The kept keys are a SUBSET of the samples, so their target-th largest is <= the samples' target-th
// largest: tau stays a valid lower bound (it is lower - more candidates, still exact - only when one lane held more
// than T of the top-`target` samples). Replaces the radix path with one dependent global load per step (2.9 ms on a
// 576 x 250000 chunk at S = 32 inside the 1-WG/CU fused kernel). The caller checks target <= 16 T.
template <int T, int B, bool DPP = true, typename LD>
__device__ __forceinline__ float wave_tau_stream(LD& ld, int ns, int target, uint32_t* hist, int lane) {
    uint32_t top[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) top[t] = 0u;
    #pragma unroll 1
    for (int c0 = 0; c0 < ns; c0 += B * WARP) {   // wave-uniform trip count
        uint32_t v[B];
        #pragma unroll
        for (int j = 0; j < B; ++j) {               // unconditional loads (clamped index), masked after
            const int c = c0 + j * WARP + lane;
            const float x = ld(min(c, ns - 1));
            v[j] = c < ns ? okey32(__builtin_bit_cast(uint32_t, x)) : 0u;
        }
        #pragma unroll
        for (int j = 0; j < B; ++j) {
            uint32_t x = v[j];
            #pragma unroll
            for (int t = 0; t < T - 1; ++t) { const uint32_t h = max(top[t], x); x = min(top[t], x); top[t] = h; }
            top[T - 1] = max(top[T - 1], x);
        }
    }
    return wave_tau_relradix<T, DPP>(top, target, hist, lane);
}
constexpr int TAU_STREAM_T = 8, TAU_STREAM_B = 16;

// tau of up to NR rows of one wave in LOCKSTEP, so their load and LDS latency chains overlap (4 rows one after another
// were latency-bound: 29 us per 32-row block at 896 samples/row, 116 us at 8192). Row q: act[q] (false -> -inf),
// tgt[q] target rank (<= 16 T), samples in nseg segments: cntf(q, g) = wave-uniform count, ld(q, g, i) = value i.
// 1) every lane keeps its T largest keys per row; BAL: segment elements are dealt to lanes round-robin over the
//    concatenation (lane = (prefix + i) % 64), so lanes stay balanced and a row with <= 64 T samples keeps them all
//    (!BAL: lane = i % 64 in every segment - fewer registers; the kept-count check below still guards the result);
//    the kept keys are a subset of the samples, so the result is <= the exact target-th largest (valid lower bound);
// 2) the relative two-pass radix of wave_tau_relradix on the NR rows together (LDS: NR x 256 u32 at hist, 16 B aligned).
// A row whose kept-key count is below its target (cannot happen with balanced dealing) gets tau = -inf (safe).
template <int NR, int B, bool BAL = true, int ESTR = 4, typename CNT>
__device__ __forceinline__ void wave_tau_multi(const uint32_t* const (&rowp)[NR], int64_t seg_stride, CNT&& cntf, int nseg,
                                               const bool (&act)[NR], const int (&tgt)[NR], uint32_t* hist, int lane,
                                               float (&tau)[NR], int* dbg_kept = nullptr, long long* tsp = nullptr) {
    // tsp (probe builds): s_memrealtime after streaming [0], after radix pass 1 [1], at the end [2] (lane 0)
    // rowp[q]: wave-uniform base of row q's samples in segment 0 (value i's fp32 bits at rowp[q][ESTR i]: ESTR 2 =
    // pair-interleaved 4-byte value slots, 4 = pair-interleaved u64 records); segment g starts seg_stride u32 later.
    // Loads past a row's count are masked, not clamped (uniform base + 32-bit lane offset + immediate: one VGPR per row
    // and round), so the sample buffer needs B * 64 samples of slack at its end. Inactive rows still need a valid rowp.
    constexpr int T = TAU_STREAM_T;
    // per-lane T largest samples per row, kept as FLOATS in descending order (finite logits: float order = key order)
    // and inserted with one v_med3_f32 per slot - b0 = max(a0, x), bt = med3(a(t-1), at, x) - instead of a
    // 2T-1-step max/min chain on order keys plus a key conversion per sample (~20 -> ~10 VALU per sample, no chain).
    // med3 / max return one of their inputs, so the kept values stay a sub-multiset of the samples.
    float topf[NR][T];
    #pragma unroll
    for (int q = 0; q < NR; ++q)
        #pragma unroll
        for (int t = 0; t < T; ++t) topf[q][t] = -__builtin_huge_valf();
    int pref[NR];
    #pragma unroll
    for (int q = 0; q < NR; ++q) pref[q] = 0;
    #pragma unroll 1
    for (int g = 0; g < nseg; ++g) {
        int n[NR], off[NR], nmax = 0;
        const uint32_t* sp[NR];
        #pragma unroll
        for (int q = 0; q < NR; ++q) {
            n[q] = act[q] ? cntf(q, g) : 0;
            off[q] = BAL ? ((lane - pref[q]) & (WARP - 1)) : lane;   // this lane's first element of the segment
            nmax = max(nmax, n[q]);
            sp[q] = rowp[q] + (int64_t)g * seg_stride;
        }
        #pragma unroll 1
        for (int c0 = 0; c0 < nmax; c0 += B * WARP) {         // wave-uniform trip count
            uint32_t v[NR][B];
            #pragma unroll
            for (int q = 0; q < NR; ++q) {
                const uint32_t* lp = sp[q] + ESTR * (c0 + off[q]);   // value i at [ESTR i]: 2 = value slots, 4 = u64 records
                #pragma unroll
                for (int j = 0; j < B; ++j) v[q][j] = lp[ESTR * j * WARP];
            }
            __builtin_amdgcn_sched_barrier(0);                 // keep the round's loads together (in flight at once)
            #pragma unroll
            for (int q = 0; q < NR; ++q)
                #pragma unroll
                for (int j = 0; j < B; ++j) {
                    const float x = (c0 + j * WARP + off[q] < n[q]) ? __builtin_bit_cast(float, v[q][j]) : -__builtin_huge_valf();
                    float nb[T];
                    nb[0] = __builtin_fmaxf(topf[q][0], x);
                    #pragma unroll
                    for (int t = 1; t < T; ++t) nb[t] = __builtin_amdgcn_fmed3f(topf[q][t - 1], topf[q][t], x);
                    #pragma unroll
                    for (int t = 0; t < T; ++t) topf[q][t] = nb[t];
                }
        }
        #pragma unroll
        for (int q = 0; q < NR; ++q) pref[q] += n[q];
    }
    if (tsp && lane == 0) tsp[0] = (long long)__builtin_amdgcn_s_memrealtime();
    uint32_t top[NR][T];   // order keys of the kept samples (0 = empty slot)
    #pragma unroll
    for (int q = 0; q < NR; ++q)
        #pragma unroll
        for (int t = 0; t < T; ++t)
            top[q][t] = topf[q][t] == -__builtin_huge_valf() ? 0u : okey32(__builtin_bit_cast(uint32_t, topf[q][t]));
    // ---- relative two-pass radix, NR rows together ----
    uint32_t kmax[NR], kmin[NR];
    int kept[NR];
    #pragma unroll
    for (int q = 0; q < NR; ++q) {
        uint32_t vmax = 0u, vmin = 0xFFFFFFFFu; int nk = 0;
        #pragma unroll
        for (int t = 0; t < T; ++t) { const uint32_t k = top[q][t]; vmax = max(vmax, k); vmin = min(vmin, k ? k : 0xFFFFFFFFu); nk += k != 0u; }
        kmax[q] = wave_max_u32(vmax); kmin[q] = wave_min_u32(vmin); kept[q] = wave_sum_dpp(nk);
    }
    int sh1[NR], sh2[NR], a1[NR];
    uint32_t pre1[NR], Tk[NR];
    #pragma unroll
    for (int q = 0; q < NR; ++q) {
        const uint32_t diff = kmax[q] ^ kmin[q];
        const int hb = diff ? 31 - __builtin_clz(diff) : 0;
        sh1[q] = max(hb - 7, 0); sh2[q] = max(sh1[q] - 8, 0);
        pre1[q] = (sh1[q] + 8 >= 32) ? 0u : (kmax[q] >> (sh1[q] + 8));
    }
    auto zero_hist = [&]() {
        #pragma unroll
        for (int b = lane * 4; b < NR * 256; b += WARP * 4) *reinterpret_cast<u32x4*>(&hist[b]) = u32x4{0u, 0u, 0u, 0u};
    };
    zero_hist();
    mem_fence_wave();
    #pragma unroll
    for (int q = 0; q < NR; ++q)
        #pragma unroll
        for (int t = 0; t < T; ++t) { const uint32_t k = top[q][t]; if (k) atomicAdd(&hist[q * 256 + ((k >> sh1[q]) & 255u)], 1u); }
    mem_fence_wave();
    #pragma unroll
    for (int q = 0; q < NR; ++q) {
        int b1, c1;
        wave_find256<true>(hist + q * 256, tgt[q], lane, b1, a1[q], c1);
        Tk[q] = (pre1[q] << 8 | (uint32_t)(b1 & 255)) << sh1[q];
    }
    if (tsp && lane == 0) tsp[1] = (long long)__builtin_amdgcn_s_memrealtime();
    mem_fence_wave();
    zero_hist();
    mem_fence_wave();
    #pragma unroll
    for (int q = 0; q < NR; ++q) {
        const uint32_t hiT = Tk[q] >> sh1[q];
        const uint32_t wmask = (1u << (sh1[q] - sh2[q])) - 1u;
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const uint32_t k = top[q][t];
            if (sh1[q] > 0 && k && (k >> sh1[q]) == hiT) atomicAdd(&hist[q * 256 + ((k >> sh2[q]) & wmask)], 1u);
        }
    }
    mem_fence_wave();
    #pragma unroll
    for (int q = 0; q < NR; ++q) {
        int b2, a2, c2;
        wave_find256<true>(hist + q * 256, tgt[q] - a1[q], lane, b2, a2, c2);
        const uint32_t Tf = sh1[q] > 0 ? (Tk[q] | ((uint32_t)(b2 & 255) << sh2[q])) : Tk[q];
        const bool flat = kmax[q] == kmin[q];
        const uint32_t key = flat ? kmax[q] : Tf;
        tau[q] = (act[q] && kept[q] >= tgt[q]) ? __builtin_bit_cast(float, okey32_inv(key)) : -__builtin_huge_valf();
        if (dbg_kept) dbg_kept[q] = kept[q];
    }
    mem_fence_wave();
    if (tsp && lane == 0) tsp[2] = (long long)__builtin_amdgcn_s_memrealtime();
}
// target rank of a row's tau from its sample count ns and valid-key count nvalid (steps 2-4 rule); 0 -> tau = -inf
__device__ __forceinline__ int tau_target(int ns, int nvalid, int K) {
    const float e = (float)K * (float)ns / (float)max(nvalid, 1);
    const int target = (int)ceilf(e + 5.f * sqrtf(e)) + 8;
    return (nvalid <= K || ns < target) ? 0 : target;
}

// Exact K-th largest of the keys rk (0 = empty slot) by a RELATIVE radix: 8-bit LDS-histogram passes starting just
// below the common prefix of the row's min / max key; stops as soon as the boundary bin holds exactly the keys still
// needed (then every key >= the bin's lower edge is selected). Returns T / above / cnt_eq in wave_emit's convention:
// emit keys > T, then keys == T until K. Typically 2-3 passes instead of a 32-step bitwise search.
template <int R, bool DPP = true>
__device__ __forceinline__ void wave_select_relradix(const uint32_t (&rk)[R], int K, uint32_t* hist, int lane,
                                                     uint32_t& T, int& above, int& cnt_eq) {
    uint32_t vmax = 0u, vmin = 0xFFFFFFFFu;
    #pragma unroll
    for (int j = 0; j < R; ++j) { vmax = max(vmax, rk[j]); vmin = min(vmin, rk[j] ? rk[j] : 0xFFFFFFFFu); }
    const uint32_t kmax = wave_max_u32(vmax), kmin = wave_min_u32(vmin);
    above = 0;
    if (kmax == kmin) { T = kmax; cnt_eq = wave_count_ge(rk, kmax); return; }
    // pass covers bits [sh, top); bits >= top are fixed (common prefix, then previous passes) and must match
    int top = 32 - __builtin_clz(kmax ^ kmin);                 // highest differing bit + 1
    int sh = max(top - 8, 0);
    uint32_t pref = (top >= 32) ? 0u : ((kmax >> top) << top);
    int need = K;
    #pragma unroll 1
    while (true) {
        for (int b = lane; b < 256; b += WARP) hist[b] = 0u;
        mem_fence_wave();
        const uint32_t hp = (top >= 32) ? 0u : (pref >> top);
        const uint32_t wmask = (1u << (top - sh)) - 1u;
        #pragma unroll
        for (int j = 0; j < R; ++j) {
            const uint32_t k = rk[j];
            const bool match = (k != 0u) & (((top >= 32) ? 0u : (k >> top)) == hp);
            if (match) atomicAdd(&hist[(k >> sh) & wmask], 1u);
        }
        mem_fence_wave();
        int b, ab, cb;
        wave_find256<DPP>(hist, need, lane, b, ab, cb);
        pref |= (uint32_t)b << sh;
        above += ab; need -= ab;
        mem_fence_wave();
        if (cb == need && pref != 0u) {   // the whole boundary bin is selected: keys >= pref (low bits zero)
            T = pref - 1u; cnt_eq = 0; above += need; return;
        }
        if (sh == 0) { T = pref; cnt_eq = cb; return; }
        top = sh;
        sh = max(sh - 8, 0);
    }
}

// ------------------------------------------------------------------------------------------------- tau
// Threshold of one row from fp32 sample logits srow[a, b) (sample j = key j*S). One wave. Returns -inf when every
// valid key must be a candidate. `hist` = 256 u32 of LDS owned by this wave. The register count follows the sample
// count (wave-uniform branch), and only the top 20 bits of the order key are searched (tau = lowest float of that
// 2^-11-relative bucket, still <= the target-th largest sample).
constexpr int TAU_LOWBIT = 12;
template <int R, bool COHERENT, bool DPP = true>
__device__ __forceinline__ float wave_tau_regs(const float* __restrict__ srow, int a, int b, int target, uint32_t* hist, int lane) {
    uint32_t rk[R];
    if constexpr (COHERENT) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    #pragma unroll
    for (int j = 0; j < R; ++j) {               // unconditional loads (clamped index), masked after: no branch + wait per load
        const int c = a + j * WARP + lane;
        const float v = srow[min(c, b - 1)];
        rk[j] = c < b ? okey32(__builtin_bit_cast(uint32_t, v)) : 0u;
    }
    return wave_tau_relradix<R, DPP>(rk, target, hist, lane);
}
template <bool COHERENT, bool DPP = true>
__device__ __forceinline__ float wave_tau(const float* __restrict__ srow, int a, int b, int nvalid, int K, int CAP,
                                          uint32_t* hist, int lane) {
    const int ns = max(b - a, 0);
    const float e = (float)K * (float)ns / (float)max(nvalid, 1);
    const int target = (int)ceilf(e + 5.f * sqrtf(e)) + 8;
    if (nvalid <= CAP || ns < target) return -__builtin_huge_valf();
    if (ns <= 8 * WARP) return wave_tau_regs<8, COHERENT, DPP>(srow, a, b, target, hist, lane);
    if (ns <= 16 * WARP) return wave_tau_regs<16, COHERENT, DPP>(srow, a, b, target, hist, lane);
    if (ns <= 32 * WARP) return wave_tau_regs<32, COHERENT, DPP>(srow, a, b, target, hist, lane);
    if constexpr (COHERENT) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    if (target <= 16 * TAU_STREAM_T) {
        auto ld = [&](int i) -> float { return srow[a + i]; };
        return wave_tau_stream<TAU_STREAM_T, TAU_STREAM_B, DPP>(ld, ns, target, hist, lane);
    }
    constexpr int RT = 32;                          // first RT * 64 samples held in VGPRs (one read), rest re-read
    uint32_t rk[RT];
    #pragma unroll
    for (int j = 0; j < RT; ++j) { const int c = a + j * WARP + lane; const float v = srow[min(c, b - 1)]; rk[j] = c < b ? okey32(__builtin_bit_cast(uint32_t, v)) : 0u; }
    auto each = [&](auto&& f) {
        #pragma unroll
        for (int j = 0; j < RT; ++j) if (rk[j]) f(rk[j], 0);
        for (int c0 = a + RT * WARP; c0 < b; c0 += WARP) {     // uniform trip count, unconditional load
            const int c = c0 + lane; const float v = srow[min(c, b - 1)];
            if (c < b) f(okey32(__builtin_bit_cast(uint32_t, v)), c);
        }
    };
    uint32_t T; int above, ce;
    wave_radix<4, DPP>(each, hist, target, lane, T, above, ce);
    return __builtin_bit_cast(float, okey32_inv(T));
}

// Threshold of NR rows at once (one wave): the NR prefix searches run in lockstep, so every step issues NR
// independent compare/popcount chains (the single-row search is a serial chain and latency-bound at low occupancy).
// Row i: samples srow_i[a_i, b_i), wanted rank target_i; act_i false -> tau_i = -inf. All rows must have
// b_i - a_i <= R * 64. The caller handles rows needing more samples with wave_tau().
template <int R, int NR, bool COHERENT>
__device__ __forceinline__ void wave_tau_rows(const float* const (&srow)[NR], const int (&a)[NR], const int (&b)[NR],
                                              const int (&target)[NR], const bool (&act)[NR], float (&tau)[NR], int lane) {
    uint32_t rk[NR][R];
    if constexpr (COHERENT) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    #pragma unroll
    for (int i = 0; i < NR; ++i) {
        const int last = act[i] ? b[i] - 1 : 0;   // inactive rows read element 0 of a valid row (value unused)
        #pragma unroll
        for (int j = 0; j < R; ++j) {
            const int c = a[i] + j * WARP + lane;
            const float v = srow[i][act[i] ? min(c, last) : 0];
            rk[i][j] = (act[i] && c < b[i]) ? okey32(__builtin_bit_cast(uint32_t, v)) : 0u;
        }
    }
    uint32_t t[NR];
    #pragma unroll
    for (int i = 0; i < NR; ++i) t[i] = 0;
    #pragma unroll 1
    for (int bit = 31; bit >= TAU_LOWBIT; --bit) {
        #pragma unroll
        for (int i = 0; i < NR; ++i) {
            const uint32_t c = t[i] | (1u << bit);
            if (wave_count_ge(rk[i], c) >= target[i]) t[i] = c;
        }
    }
    #pragma unroll
    for (int i = 0; i < NR; ++i) tau[i] = act[i] ? __builtin_bit_cast(float, okey32_inv(t[i])) : -__builtin_huge_valf();
}

// tau of the 4 rows m0..m0+3 of one wave from samples ls[m][*] (row stride ls_cols; sample j = key j*S).
template <bool COHERENT>
__device__ __forceinline__ void wave_tau4(const float* __restrict__ ls, int ls_cols, int Ns, int S, int m0, int M, int N,
                                          const int* __restrict__ ks_p, const int* __restrict__ ke_p, int K, int CAP,
                                          uint32_t* hist, int lane, float (&tau)[4]) {
    const float* srow[4]; int a[4], b[4], target[4], nv[4]; bool act[4];
    int nsmax = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = m0 + i;
        const bool ok = m < M;
        const int ksm = ok ? max(ks_p[m], 0) : 0, kem = ok ? min(ke_p[m], N) : 0;
        nv[i] = kem - ksm;
        a[i] = FMT_CDIV(ksm, S);
        b[i] = kem > 0 ? min(FMT_CDIV(kem, S), Ns) : 0;
        const int ns = max(b[i] - a[i], 0);
        const float e = (float)K * (float)ns / (float)max(nv[i], 1);
        target[i] = (int)ceilf(e + 5.f * sqrtf(e)) + 8;
        act[i] = ok && nv[i] > CAP && ns >= target[i];
        srow[i] = ls + (int64_t)(ok ? m : 0) * ls_cols;
        if (act[i]) nsmax = max(nsmax, ns);
    }
    (void)nsmax;
    #pragma unroll 1
    for (int i = 0; i < 4; ++i)
        tau[i] = act[i] ? wave_tau<COHERENT>(srow[i], a[i], b[i], nv[i], K, CAP, hist, lane) : -__builtin_huge_valf();
}

// Generic-loader tau (steps 2-4): samples v_i = ld(i), i in [0, ns). tau = -inf only when the row has at most K valid
// keys (all selected anyway) or too few samples for the target rank.
template <int R, typename LD>
__device__ __forceinline__ float wave_tau_ld_regs(LD& ld, int ns, int target, uint32_t* hist, int lane) {
    uint32_t rk[R];
    #pragma unroll
    for (int j = 0; j < R; ++j) {
        const int c = j * WARP + lane;
        const float v = ld(min(c, ns - 1));
        rk[j] = c < ns ? okey32(__builtin_bit_cast(uint32_t, v)) : 0u;
    }
    return wave_tau_relradix<R>(rk, target, hist, lane);
}
template <typename LD>
__device__ __forceinline__ float wave_tau_ld(LD&& ld, int ns, int nvalid, int K, uint32_t* hist, int lane) {
    const float e = (float)K * (float)ns / (float)max(nvalid, 1);
    const int target = (int)ceilf(e + 5.f * sqrtf(e)) + 8;
    if (nvalid <= K || ns < target) return -__builtin_huge_valf();
    if (ns <= 8 * WARP) return wave_tau_ld_regs<8>(ld, ns, target, hist, lane);
    if (ns <= 16 * WARP) return wave_tau_ld_regs<16>(ld, ns, target, hist, lane);
    if (ns <= 32 * WARP) return wave_tau_ld_regs<32>(ld, ns, target, hist, lane);
#ifndef FMT_NO_TAU_STREAM
    if (target <= 16 * TAU_STREAM_T) return wave_tau_stream<TAU_STREAM_T, TAU_STREAM_B>(ld, ns, target, hist, lane);
#endif
    constexpr int RT = 32;
    uint32_t rk[RT];
    #pragma unroll
    for (int j = 0; j < RT; ++j) { const int c = j * WARP + lane; rk[j] = okey32(__builtin_bit_cast(uint32_t, ld(c))); }
    auto each = [&](auto&& f) {
        #pragma unroll
        for (int j = 0; j < RT; ++j) f(rk[j], 0);
        for (int c0 = RT * WARP; c0 < ns; c0 += WARP) {
            const int c = c0 + lane; const float v = ld(min(c, ns - 1));
            if (c < ns) f(okey32(__builtin_bit_cast(uint32_t, v)), c);
        }
    };
    uint32_t T; int above, ce;
    wave_radix<4>(each, hist, target, lane, T, above, ce);
    return __builtin_bit_cast(float, okey32_inv(T));
}

// standalone tau kernel (steps 0): fp32 sample logits ls[M, Ns] of keys j*S; 4 rows (waves) per block.
// Also resets the fallback counter of this call (select appends, fallback kernels read).
__global__ __launch_bounds__(256) void tau_kernel(const float* __restrict__ ls, int Ns, int S, const int* __restrict__ ks_p,
                                                  const int* __restrict__ ke_p, int M, int N, int K, int CAP,
                                                  float* __restrict__ tau, int* __restrict__ fbc) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *fbc = 0;
    __shared__ uint32_t hist_all[4][256];
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, m = blockIdx.x * 4 + w;
    if (m >= M) return;
    const int ksm = max(ks_p[m], 0), kem = min(ke_p[m], N);
    const int nvalid = kem - ksm;
    const int a = FMT_CDIV(ksm, S), b = kem > 0 ? min(FMT_CDIV(kem, S), Ns) : 0;
    const float t = wave_tau<false, false>(ls + (int64_t)m * Ns, a, b, nvalid, K, CAP, hist_all[w], lane);   // shuffle find256: high occupancy
    if (lane == 0) tau[m] = t;
}

// ------------------------------------------------------------------------------------------------- select
// One row: candidates i < cnt[s * M + m] of split s at cand[cand_index(s, m, i)]. Returns false if the row must go to
// the fallback (overflow, or fewer than min(K, nvalid) candidates).
// need == nullptr: row-wide tau (steps 0-1), pass iff no split overflowed and C >= min(K, nvalid).
// need != nullptr: per-split tau (steps 2-4), pass iff every split s has cnt_s <= CAP and cnt_s >= need_s
// (need_s = min(K, nvalid_s), written by the scan itself).
// taus != nullptr (with need): exact pre-filter - every split with need_s == K holds >= K keys >= tau_s, so the row's
// K-th largest is >= max of those tau_s; candidates below it are dropped (keys 0) before the search.
template <int RC, bool COHERENT, bool DPP = true>
__device__ __forceinline__ bool wave_select_row(const unsigned long long* __restrict__ cand, const int* __restrict__ cnt,
                                                int nsplit, int CAP, int M, int m, int ks, int nvalid, int K,
                                                int* __restrict__ orow, uint32_t* hist, int lane,
                                                const int* __restrict__ need = nullptr, const float* __restrict__ taus = nullptr,
                                                uint32_t* __restrict__ surv = nullptr) {
    const int Mp = M + (M & 1);
    if (nvalid <= 0) { for (int i = lane; i < K; i += WARP) orow[i] = -1; return true; }
    if constexpr (COHERENT) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    const int64_t so = (int64_t)min(lane, nsplit - 1) * M + m;
    int c_l = cnt[so];
    int n_l = need ? need[so] : 0;
    if (lane >= nsplit) { c_l = 0; n_l = 0; }
    const bool bad_l = (c_l > CAP) | (c_l < n_l);
    const bool fail_split = __ballot(bad_l) != 0;
    uint32_t kthr = 0u;                              // keys < kthr cannot be in the top-K
    if (taus) {
        const float t_l = (lane < nsplit && n_l == K) ? taus[so] : -__builtin_huge_valf();
        kthr = wave_max_u32(okey32(__builtin_bit_cast(uint32_t, t_l)));
    }
    c_l = min(c_l, CAP);
    const int c_incl = wave_incl_scan(c_l, lane);
    int C = __shfl(c_incl, WARP - 1);
    if (fail_split || (!need && C < min(K, nvalid))) return false;
    int pe_s[16];                                   // segment prefix ends, read with all lanes active (uniform)
    #pragma unroll
    for (int t = 0; t < 16; ++t) pe_s[t] = __shfl(c_incl, t);
    auto addr = [&](int i) -> int64_t {
        int s = 0, base = 0;
        #pragma unroll
        for (int t = 0; t < 16; ++t) if (t < nsplit && i >= pe_s[t]) { s = t + 1; base = pe_s[t]; }
        return cand_index(s, m, i - base, Mp, CAP);
    };
    // unconditional loads of a clamped index (masked after): no exec branch + vmcnt(0) per load
    auto ldc = [&](int i) -> unsigned long long {
        const int ic = min(i, C - 1);
        const int64_t o = nsplit == 1 ? cand_index(0, m, ic, Mp, CAP) : addr(ic);
        return cand[o];
    };
    auto keyf = [&](unsigned long long d) -> uint32_t { const uint32_t k = okey32((uint32_t)d); return k >= kthr ? k : 0u; };
    uint32_t rk[RC]; int rc[RC];
    // Survivor compaction (multi-split rows): stream every candidate, keep keys >= kthr, compacted in this wave's LDS
    // area surv[0, 2*SURV) (keys, then cols) by ballot/mbcnt; if they fit in the registers the search runs on them only.
    constexpr int SURV = RC * WARP;
    bool compacted = false;
    if (surv && C > RC * WARP) {
        int nsv = 0;
        for (int i0 = 0; i0 < C; i0 += WARP) {
            const int i = i0 + lane;
            const unsigned long long d = ldc(i);
            const uint32_t k = i < C ? keyf(d) : 0u;
            const uint64_t mk = __ballot(k != 0u);
            const int pos = nsv + (int)__builtin_amdgcn_mbcnt_hi((uint32_t)(mk >> 32), __builtin_amdgcn_mbcnt_lo((uint32_t)mk, 0u));
            if (k != 0u && pos < SURV) { surv[pos] = k; surv[SURV + pos] = (int)(d >> 32); }
            nsv += __popcll(mk);
        }
        if (nsv <= SURV) {
            mem_fence_wave();
            #pragma unroll
            for (int j = 0; j < RC; ++j) {
                const int i = j * WARP + lane;
                rk[j] = i < nsv ? surv[min(i, SURV - 1)] : 0u; rc[j] = (int)surv[SURV + min(i, SURV - 1)];
            }
            C = nsv;
            compacted = true;
        }
    }
    if (!compacted) {
        #pragma unroll
        for (int j = 0; j < RC; ++j) {
            const int i = j * WARP + lane;
            const unsigned long long d = ldc(i);
            rk[j] = i < C ? keyf(d) : 0u; rc[j] = (int)(d >> 32);
        }
    }
    if (C <= K) {     // every candidate is selected (rows with < K valid keys): emit all, pad -1
        #pragma unroll
        for (int j = 0; j < RC; ++j) if (j * WARP + lane < C) orow[j * WARP + lane] = rc[j] - ks;
        for (int i0 = RC * WARP; i0 < C; i0 += WARP) { const int i = i0 + lane; const unsigned long long d = ldc(i); if (i < C) orow[i] = (int)(d >> 32) - ks; }
        for (int i = C + lane; i < K; i += WARP) orow[i] = -1;
        return true;
    }
    auto each = [&](auto&& f) {
        #pragma unroll
        for (int j = 0; j < RC; ++j) if (j * WARP + lane < C && rk[j]) f(rk[j], rc[j]);
        for (int i0 = RC * WARP; i0 < C; i0 += WARP) { const int i = i0 + lane; const unsigned long long d = ldc(i); const uint32_t k = keyf(d); if (i < C && k) f(k, (int)(d >> 32)); }
    };
    uint32_t T; int above, ce;
    if (C <= RC * WARP) wave_select_relradix<RC, DPP>(rk, K, hist, lane, T, above, ce);
    else wave_radix<4, DPP>(each, hist, K, lane, T, above, ce);
    wave_emit([&](auto&& g) {
        #pragma unroll
        for (int j = 0; j < RC; ++j) if (j * WARP < C) g(j * WARP + lane < C, rk[j], rc[j]);
        for (int i0 = RC * WARP; i0 < C; i0 += WARP) {
            const int i = i0 + lane; const bool v = i < C;
            const unsigned long long d = ldc(i);
            g(v, keyf(d), (int)(d >> 32));
        }
    }, T, above, K, ks, lane, orow);
    return true;
}

// RC from the candidate count of the row (wave-uniform): fewer registers to compare in every search step
template <bool COHERENT, bool DPP = true>
__device__ __forceinline__ bool wave_select_row_auto(const unsigned long long* __restrict__ cand, const int* __restrict__ cnt,
                                                     int nsplit, int CAP, int M, int m, int ks, int nvalid, int K,
                                                     int* __restrict__ orow, uint32_t* hist, int lane,
                                                     const int* __restrict__ need = nullptr,
                                                     const float* __restrict__ taus = nullptr,
                                                     uint32_t* __restrict__ surv = nullptr) {
    if constexpr (COHERENT) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    int c_l = cnt[(int64_t)min(lane, nsplit - 1) * M + m];
    if (lane >= nsplit) c_l = 0;
    int C = min(c_l, CAP);
    #pragma unroll
    for (int d = 32; d >= 1; d >>= 1) C += __shfl_xor(C, d);
    if (C <= 16 * WARP) return wave_select_row<16, COHERENT, DPP>(cand, cnt, nsplit, CAP, M, m, ks, nvalid, K, orow, hist, lane, need, taus);
    if (C <= 24 * WARP) return wave_select_row<24, COHERENT, DPP>(cand, cnt, nsplit, CAP, M, m, ks, nvalid, K, orow, hist, lane, need, taus);
    if (C <= 32 * WARP) return wave_select_row<32, COHERENT, DPP>(cand, cnt, nsplit, CAP, M, m, ks, nvalid, K, orow, hist, lane, need, taus, surv);
    return wave_select_row<48, COHERENT, DPP>(cand, cnt, nsplit, CAP, M, m, ks, nvalid, K, orow, hist, lane, need, taus);
}

template <int RC, bool SURVIVORS = false>
__global__ __launch_bounds__(256) void select_kernel(const unsigned long long* __restrict__ cand, const int* __restrict__ cnt, int nsplit, int CAP,
                                                     const int* __restrict__ ks_p, const int* __restrict__ ke_p, int M, int N, int K,
                                                     int* __restrict__ out, int* __restrict__ flag, int* __restrict__ fbc,
                                                     const int* __restrict__ need = nullptr, const float* __restrict__ taus = nullptr) {
    __shared__ uint32_t hist_all[4][256];
    __shared__ uint32_t surv_all[SURVIVORS ? 4 : 1][SURVIVORS ? 2 * 32 * WARP : 1];
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, m = blockIdx.x * 4 + w;
    if (m >= M) return;
    const int ks = max(ks_p[m], 0), nvalid = min(ke_p[m], N) - ks;
    if (!wave_select_row_auto<false, false>(cand, cnt, nsplit, CAP, M, m, ks, nvalid, K, out + (int64_t)m * K, hist_all[w], lane, need, taus,
                                     SURVIVORS ? surv_all[w] : nullptr))
        if (lane == 0) flag[atomicAdd(fbc, 1)] = m;
}

// ------------------------------------------------------------------------------------------------- exact recompute
// Exact v35wes logit of (row m, token tok) for the 32 tokens held by lanes l % 32 (all 64 lanes take part); result
// valid in lanes 0..31. Same operand layout, plain MFMA, scalar even/odd fma chains, permlane pairing (hset 0 half +
// hset 1 half) and kv_scale multiply as mqa_v35wes::kernel -> bit-identical.
struct QW { f8x32 q0, q1; f32x2 w2[8]; };
__device__ __forceinline__ QW load_qw(int m, const fp8* __restrict__ Q, const float* __restrict__ W) {
    const int l = threadIdx.x % WARP, nA = l % 32, dhalf = l / 32, hset = l / 32;
    const int64_t qb = ((int64_t)m * NUM_HEADS + nA) * HEAD_SIZE;
    QW r;
    r.q0 = pack(*reinterpret_cast<const f8x16*>(&Q[qb + dhalf * 16]), *reinterpret_cast<const f8x16*>(&Q[qb + (2 + dhalf) * 16]));
    r.q1 = pack(*reinterpret_cast<const f8x16*>(&Q[qb + (4 + dhalf) * 16]), *reinterpret_cast<const f8x16*>(&Q[qb + (6 + dhalf) * 16]));
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const f32x4 wv = *reinterpret_cast<const f32x4*>(&W[(int64_t)m * NUM_HEADS + 8 * j + 4 * hset]);
        r.w2[2 * j] = f32x2{wv[0], wv[1]};
        r.w2[2 * j + 1] = f32x2{wv[2], wv[3]};
    }
    return r;
}
__device__ __forceinline__ float recompute_wes(const QW& qw, int tok, const fp8* __restrict__ Kc, const float* __restrict__ kv_scale) {
    const int l = threadIdx.x % WARP, dhalf = l / 32;
    const int64_t kb = (int64_t)tok * HEAD_SIZE;
    const f8x32 b0 = pack(*reinterpret_cast<const f8x16*>(&Kc[kb + dhalf * 16]), *reinterpret_cast<const f8x16*>(&Kc[kb + (2 + dhalf) * 16]));
    const f8x32 b1 = pack(*reinterpret_cast<const f8x16*>(&Kc[kb + (4 + dhalf) * 16]), *reinterpret_cast<const f8x16*>(&Kc[kb + (6 + dhalf) * 16]));
    const float scale = kv_scale[tok];
    VecOut z = {};
    VecOut c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(qw.q0, b0, z, 0, 0, 0, 0, 0, 0);
    c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(qw.q1, b1, c, 0, 0, 0, 0, 0, 0);
    float ae = 0.f, ao = 0.f;
    #pragma unroll
    for (int v = 0; v < 8; ++v) {
        const float ye = __builtin_fmaxf(c[2 * v], 0.f), yo = __builtin_fmaxf(c[2 * v + 1], 0.f);
        if (v == 0) { ae = ye * qw.w2[0][0]; ao = yo * qw.w2[0][1]; }
        else { ae = __builtin_fmaf(ye, qw.w2[v][0], ae); ao = __builtin_fmaf(yo, qw.w2[v][1], ao); }
    }
    float p0 = ae + ao, p1 = p0;
    swap32_reduce2<2>(p0, p1);      // lanes 0-31: (own hset-0 half, hset-1 half of lane + 32)
    return (p0 + p1) * scale;
}

// exact select of one row from a dense fp32 scratch row srow[ks, ke) (one wave)
template <bool COHERENT>
__device__ __forceinline__ void wave_select_dense(const float* __restrict__ srow, int ks, int ke, int K, int* __restrict__ orow,
                                                  uint32_t* hist, int lane) {
    if (ke - ks <= K) { for (int i = lane; i < K; i += WARP) orow[i] = i < ke - ks ? i : -1; return; }
    if constexpr (COHERENT) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    auto ld = [&](int c) -> uint32_t { return okey32(__builtin_bit_cast(uint32_t, srow[min(c, ke - 1)])); };
    auto each = [&](auto&& f) { for (int c0 = ks; c0 < ke; c0 += WARP) { const int c = c0 + lane; const uint32_t k = ld(c); if (c < ke) f(k, c); } };
    uint32_t T; int above, ce;
    wave_radix<4>(each, hist, K, lane, T, above, ce);
    wave_emit([&](auto&& g) {
        for (int c0 = ks; c0 < ke; c0 += WARP) { const int c = c0 + lane; const bool v = c < ke; const uint32_t k = ld(c); g(v, v ? k : 0u, c); }
    }, T, above, K, ks, lane, orow);
}

// ------------------------------------------------------------------------------------------------- fallback (step 0: 3 kernels)
// The first FBR flagged rows (in list order) are recomputed by every wave of the grid in parallel (fb_compute) and
// selected one wave per row (fb_select); further flagged rows fall back to one wave per row (fallback_kernel).
constexpr int FBR = 64;
__global__ __launch_bounds__(256) void fb_compute(const int* __restrict__ flag, const int* __restrict__ fbc, const int* __restrict__ ks_p, const int* __restrict__ ke_p,
                                                  int M, int N, float* __restrict__ scratch, const fp8* __restrict__ Q, const fp8* __restrict__ Kc,
                                                  const float* __restrict__ kv_scale, const float* __restrict__ W) {
    const int lane = threadIdx.x % WARP, gw = blockIdx.x * 4 + threadIdx.x / WARP, G = gridDim.x * 4;
    const int nf = min(*fbc, FBR);
    for (int ord = 0; ord < nf; ++ord) {
        const int m = flag[ord];
        const int ks = max(ks_p[m], 0), ke = min(ke_p[m], N);
        const int nb = (ke - ks + 31) / 32;
        if (gw >= nb) continue;
        const QW qw = load_qw(m, Q, W);
        for (int b = gw; b < nb; b += G) {
            const int c0 = ks + b * 32, tok = min(c0 + lane % 32, ke - 1);
            const float v = recompute_wes(qw, tok, Kc, kv_scale);
            if (lane < 32 && c0 + lane < ke) scratch[(int64_t)ord * N + c0 + lane] = v;
        }
    }
}
__global__ __launch_bounds__(256) void fb_select(const int* __restrict__ flag, const int* __restrict__ fbc, const int* __restrict__ ks_p, const int* __restrict__ ke_p,
                                                 int M, int N, int K, int* __restrict__ out, const float* __restrict__ scratch, int* __restrict__ nfb) {
    __shared__ uint32_t hist_all[4][256];
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, ord = blockIdx.x * 4 + w;
    if (ord >= min(*fbc, FBR)) return;
    const int m = flag[ord];
    if (nfb && lane == 0) atomicAdd(nfb, 1);
    const int ks = max(ks_p[m], 0), ke = min(ke_p[m], N);
    wave_select_dense<false>(scratch + (int64_t)ord * N, ks, ke, K, out + (int64_t)m * K, hist_all[w], lane);
}
// beyond FBR flagged rows: one wave per row, own scratch row
__global__ __launch_bounds__(256) void fallback_kernel(const int* __restrict__ flag, const int* __restrict__ fbc, const int* __restrict__ ks_p,
                                                       const int* __restrict__ ke_p, int M, int N, int K, int* __restrict__ out, float* __restrict__ scratch,
                                                       const fp8* __restrict__ Q, const fp8* __restrict__ Kc, const float* __restrict__ kv_scale,
                                                       const float* __restrict__ W, int* __restrict__ nfb) {
    __shared__ uint32_t hist_all[4][256];
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, gw = blockIdx.x * 4 + w;
    uint32_t* hist = hist_all[w];
    float* srow = scratch + (int64_t)(FBR + gw) * N;
    const int nf = *fbc;
    for (int ord = FBR + gw; ord < nf; ord += gridDim.x * 4) {
        const int m = flag[ord];
        if (nfb && lane == 0) atomicAdd(nfb, 1);
        const int ks = max(ks_p[m], 0), ke = min(ke_p[m], N);
        const QW qw = load_qw(m, Q, W);
        for (int c0 = ks; c0 < ke; c0 += 32) {
            const int tok = min(c0 + lane % 32, ke - 1);
            const float v = recompute_wes(qw, tok, Kc, kv_scale);
            if (lane < 32 && c0 + lane < ke) srow[c0 + lane] = v;
        }
        __threadfence_block();
        wave_select_dense<true>(srow, ks, ke, K, out + (int64_t)m * K, hist, lane);
    }
}

// ------------------------------------------------------------------------------------------------- fallback (step 1+: 1 kernel)
// The first FBR flagged rows: every wave of the grid recomputes 32-token blocks of each row (strided over the grid)
// into scratch[ord][*]; a per-row completion counter done[ord] (acq_rel at agent scope - the slices of one row may be
// computed on different XCDs, whose L2s are not coherent) lets the wave that completes the row's last block run the
// exact select; that wave resets done[ord] = 0 for the next call (done[] is zeroed once at allocation). Flagged rows
// beyond FBR: one wave per row with its own scratch row FBR + wave (first 64 waves). *fbc is reset by the next call's sample pass.
__global__ __launch_bounds__(256) void fallback1_kernel(const int* __restrict__ flag, const int* __restrict__ fbc, int* __restrict__ done,
                                                        const int* __restrict__ ks_p, const int* __restrict__ ke_p, int M, int N, int K,
                                                        int* __restrict__ out, float* __restrict__ scratch, const fp8* __restrict__ Q,
                                                        const fp8* __restrict__ Kc, const float* __restrict__ kv_scale,
                                                        const float* __restrict__ W, int* __restrict__ nfb,
                                                        int* __restrict__ fb_exit = nullptr) {
    __shared__ uint32_t hist_all[4][256];
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, gw = blockIdx.x * 4 + w, G = gridDim.x * 4;
    uint32_t* hist = hist_all[w];
    const int nf = *fbc;
    const int nfa = min(nf, FBR);
    for (int ord = 0; ord < nfa; ++ord) {
        const int m = flag[ord];
        const int ks = max(ks_p[m], 0), ke = min(ke_p[m], N);
        const int nb = (ke - ks + 31) / 32;
        if (gw >= nb) continue;
        int mine = 0;
        {
            const QW qw = load_qw(m, Q, W);
            for (int b = gw; b < nb; b += G, ++mine) {
                const int c0 = ks + b * 32, tok = min(c0 + lane % 32, ke - 1);
                const float v = recompute_wes(qw, tok, Kc, kv_scale);
                if (lane < 32 && c0 + lane < ke) scratch[(int64_t)ord * N + c0 + lane] = v;
            }
        }
        int old = 0;
        if (lane == 0) { FMT_RELEASE_AGENT(); old = __hip_atomic_fetch_add(&done[ord], mine, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT); }
        old = __shfl(old, 0);
        if (old + mine == nb) {                 // this wave completed the row: every slice is visible (acquire)
            if (nfb && lane == 0) atomicAdd(nfb, 1);
            wave_select_dense<true>(scratch + (int64_t)ord * N, ks, ke, K, out + (int64_t)m * K, hist, lane);
            if (lane == 0) done[ord] = 0;
        }
    }
    // rows beyond FBR (rare): the first FBR2 waves only, so the scratch stays (FBR + FBR2) rows
    constexpr int FBR2 = 64;
    if (gw < FBR2) {
        float* srow = scratch + (int64_t)(FBR + gw) * N;
        for (int ord = FBR + gw; ord < nf; ord += FBR2) {
            const int m = flag[ord];
            if (nfb && lane == 0) atomicAdd(nfb, 1);
            const int ks = max(ks_p[m], 0), ke = min(ke_p[m], N);
            const QW qw = load_qw(m, Q, W);
            for (int c0 = ks; c0 < ke; c0 += 32) {
                const int tok = min(c0 + lane % 32, ke - 1);
                const float v = recompute_wes(qw, tok, Kc, kv_scale);
                if (lane < 32 && c0 + lane < ke) srow[c0 + lane] = v;
            }
            __threadfence_block();
            wave_select_dense<true>(srow, ks, ke, K, out + (int64_t)m * K, hist, lane);
        }
    }
    // fb_exit != nullptr: this kernel is the only consumer of the list (step 3): the last WG out resets *fbc
    // (every WG read *fbc at its start, before its exit increment).
    if (fb_exit) {
        __syncthreads();
        if (threadIdx.x == 0) {
            FMT_RELEASE_AGENT();
            const int old = __hip_atomic_fetch_add(fb_exit, 1, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
            if (old == (int)gridDim.x - 1) { *const_cast<int*>(fbc) = 0; *fb_exit = 0; }
        }
    }
}

}  // namespace fmt
