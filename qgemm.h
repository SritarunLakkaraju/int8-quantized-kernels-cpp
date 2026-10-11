// qgemm.h - int8 x int8 -> int32 matrix multiply.
//
//   C (MxN, int32) = A (MxK, int8) * B (KxN, int8), row-major.
//
// Products of two int8 values fit in int16 and are accumulated in int32, so
// nothing can overflow for any realistic K (|sum| <= K * 2^14).
//
//   naive    scalar triple loop
//   widen    NEON without the dot-product extension: widen int8 -> int16,
//            then widening multiply-accumulate into int32 (vmlal)
//   packed   BLIS-style packing + an SDOT micro-kernel (ARMv8.2 dot product):
//            one instruction performs 16 multiplies and adds them into
//            4 int32 lanes
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace qgemm {

inline void naive(int M, int N, int K, const int8_t* A, int lda, const int8_t* B, int ldb,
                  int32_t* C, int ldc) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) acc += (int32_t)A[i * lda + k] * (int32_t)B[k * ldb + j];
            C[i * ldc + j] = acc;
        }
}

// ---------------------------------------------------------------------------
// Baseline SIMD, no dot-product instruction: (i, k, j) order, 16 columns per
// step. Each int8 vector is widened to two int16 vectors, then four widening
// multiply-accumulates update 16 int32 results.
// ---------------------------------------------------------------------------
inline void widen(int M, int N, int K, const int8_t* A, int lda, const int8_t* B, int ldb,
                  int32_t* C, int ldc) {
    for (int i = 0; i < M; ++i) std::memset(C + (size_t)i * ldc, 0, sizeof(int32_t) * N);
    constexpr int TK = 64, TJ = 1024;  // B tile: 64 KB of int8, L1 resident
    for (int kk = 0; kk < K; kk += TK) {
        const int kend = std::min(kk + TK, K);
        for (int jj = 0; jj < N; jj += TJ) {
            const int jend = std::min(jj + TJ, N);
            for (int i = 0; i < M; ++i) {
                int32_t* c = C + (size_t)i * ldc;
                for (int k = kk; k < kend; ++k) {
                    const int16_t a = A[(size_t)i * lda + k];
                    const int8_t* b = B + (size_t)k * ldb;
                    int j = jj;
#if defined(__ARM_NEON)
                    for (; j + 16 <= jend; j += 16) {
                        const int8x16_t bv = vld1q_s8(b + j);
                        const int16x8_t lo = vmovl_s8(vget_low_s8(bv)), hi = vmovl_high_s8(bv);
                        vst1q_s32(c + j, vmlal_n_s16(vld1q_s32(c + j), vget_low_s16(lo), a));
                        vst1q_s32(c + j + 4, vmlal_high_n_s16(vld1q_s32(c + j + 4), lo, a));
                        vst1q_s32(c + j + 8, vmlal_n_s16(vld1q_s32(c + j + 8), vget_low_s16(hi), a));
                        vst1q_s32(c + j + 12, vmlal_high_n_s16(vld1q_s32(c + j + 12), hi, a));
                    }
#endif
                    for (; j < jend; ++j) c[j] += (int32_t)a * b[j];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Packed SDOT kernel.
//
// K is processed four values at a time ("k4 groups"). For one group the
// packed operands hold
//   A panel: MR rows x 4 bytes  -> lane r of the A vector = A[r][k..k+3]
//   B panel: NR cols x 4 bytes  -> lane c of a B vector   = B[k..k+3][c]
// vdotq_laneq_s32(acc, b, a, r) then adds, into each of 4 int32 lanes, the
// 4-term dot product of one column of B with row r of A: 16 MACs per
// instruction, 384 per k4 group for the 8x12 block.
// ---------------------------------------------------------------------------
constexpr int MR = 8, NR = 12;
constexpr int KC4 = 256;   // k4 groups per block  (KC = 1024 values)
constexpr int MC = 480, NC = 4080;

static inline void micro_kernel(int k4, const int8_t* a, const int8_t* b, int32_t* acc) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t c00 = vdupq_n_s32(0), c01 = c00, c02 = c00, c10 = c00, c11 = c00, c12 = c00;
    int32x4_t c20 = c00, c21 = c00, c22 = c00, c30 = c00, c31 = c00, c32 = c00;
    int32x4_t c40 = c00, c41 = c00, c42 = c00, c50 = c00, c51 = c00, c52 = c00;
    int32x4_t c60 = c00, c61 = c00, c62 = c00, c70 = c00, c71 = c00, c72 = c00;
#define QGEMM_ROW(r, av, lane)                        \
    c##r##0 = vdotq_laneq_s32(c##r##0, b0, av, lane); \
    c##r##1 = vdotq_laneq_s32(c##r##1, b1, av, lane); \
    c##r##2 = vdotq_laneq_s32(c##r##2, b2, av, lane);
    for (int p = 0; p < k4; ++p) {
        const int8x16_t a0 = vld1q_s8(a), a1 = vld1q_s8(a + 16);
        const int8x16_t b0 = vld1q_s8(b), b1 = vld1q_s8(b + 16), b2 = vld1q_s8(b + 32);
        QGEMM_ROW(0, a0, 0) QGEMM_ROW(1, a0, 1) QGEMM_ROW(2, a0, 2) QGEMM_ROW(3, a0, 3)
        QGEMM_ROW(4, a1, 0) QGEMM_ROW(5, a1, 1) QGEMM_ROW(6, a1, 2) QGEMM_ROW(7, a1, 3)
        a += 4 * MR;
        b += 4 * NR;
    }
#undef QGEMM_ROW
#define QGEMM_STORE(r)                    \
    vst1q_s32(acc + r * NR, c##r##0);     \
    vst1q_s32(acc + r * NR + 4, c##r##1); \
    vst1q_s32(acc + r * NR + 8, c##r##2);
    QGEMM_STORE(0) QGEMM_STORE(1) QGEMM_STORE(2) QGEMM_STORE(3)
    QGEMM_STORE(4) QGEMM_STORE(5) QGEMM_STORE(6) QGEMM_STORE(7)
#undef QGEMM_STORE
#else
    for (int i = 0; i < MR * NR; ++i) acc[i] = 0;
    for (int p = 0; p < k4; ++p, a += 4 * MR, b += 4 * NR)
        for (int i = 0; i < MR; ++i)
            for (int j = 0; j < NR; ++j)
                for (int t = 0; t < 4; ++t) acc[i * NR + j] += (int32_t)a[i * 4 + t] * (int32_t)b[j * 4 + t];
#endif
}

// kc values of K (zero-padded up to a multiple of 4), mc rows (zero-padded to MR).
static inline void pack_A(int mc, int kc, const int8_t* A, int lda, int8_t* dst) {
    const int k4 = (kc + 3) / 4;
    for (int ir = 0; ir < mc; ir += MR)
        for (int g = 0; g < k4; ++g)
            for (int i = 0; i < MR; ++i)
                for (int t = 0; t < 4; ++t) {
                    const int row = ir + i, k = g * 4 + t;
                    *dst++ = (row < mc && k < kc) ? A[(size_t)row * lda + k] : (int8_t)0;
                }
}

static inline void pack_B(int kc, int nc, const int8_t* B, int ldb, int8_t* dst) {
    const int k4 = (kc + 3) / 4;
    for (int jr = 0; jr < nc; jr += NR)
        for (int g = 0; g < k4; ++g)
            for (int j = 0; j < NR; ++j)
                for (int t = 0; t < 4; ++t) {
                    const int col = jr + j, k = g * 4 + t;
                    *dst++ = (col < nc && k < kc) ? B[(size_t)k * ldb + col] : (int8_t)0;
                }
}

static inline int8_t* alloc_bytes(size_t n) {
    void* p = nullptr;
    if (posix_memalign(&p, 64, std::max<size_t>(n, 64)) != 0) std::abort();
    return static_cast<int8_t*>(p);
}

inline void packed(int M, int N, int K, const int8_t* A, int lda, const int8_t* B, int ldb,
                   int32_t* C, int ldc, bool parallel = false) {
    for (int i = 0; i < M; ++i) std::memset(C + (size_t)i * ldc, 0, sizeof(int32_t) * N);
    if (M == 0 || N == 0 || K == 0) return;
    constexpr int KC = KC4 * 4;

    int mc_step = MC;
#ifdef _OPENMP
    if ((double)M * N * K < 4e6) parallel = false;
    if (parallel) {
        const int nt = omp_get_max_threads();
        int want = (M + 2 * nt - 1) / (2 * nt);
        want = ((want + MR - 1) / MR) * MR;
        mc_step = std::max(2 * MR, std::min(MC, want));
    }
#else
    parallel = false;
#endif
    const size_t kc_bytes = (size_t)((std::min(KC, K) + 3) / 4) * 4;
    int8_t* Bp = alloc_bytes(kc_bytes * (((size_t)std::min(NC, N) + NR - 1) / NR) * NR);

    for (int jc = 0; jc < N; jc += NC) {
        const int nc = std::min(NC, N - jc);
        for (int pc = 0; pc < K; pc += KC) {
            const int kc = std::min(KC, K - pc), k4 = (kc + 3) / 4;
            pack_B(kc, nc, B + (size_t)pc * ldb + jc, ldb, Bp);
#pragma omp parallel if (parallel)
            {
                int8_t* Ap = alloc_bytes(kc_bytes * (((size_t)std::min(mc_step, M) + MR - 1) / MR) * MR);
                int32_t acc[MR * NR] __attribute__((aligned(64)));
#pragma omp for schedule(dynamic, 1)
                for (int ic = 0; ic < M; ic += mc_step) {
                    const int mc = std::min(mc_step, M - ic);
                    pack_A(mc, kc, A + (size_t)ic * lda + pc, lda, Ap);
                    for (int jr = 0; jr < nc; jr += NR) {
                        const int nr = std::min(NR, nc - jr);
                        const int8_t* bp = Bp + (size_t)(jr / NR) * k4 * 4 * NR;
                        for (int ir = 0; ir < mc; ir += MR) {
                            const int mr = std::min(MR, mc - ir);
                            micro_kernel(k4, Ap + (size_t)(ir / MR) * k4 * 4 * MR, bp, acc);
                            int32_t* c = C + (size_t)(ic + ir) * ldc + jc + jr;
                            for (int i = 0; i < mr; ++i)
                                for (int j = 0; j < nr; ++j) c[(size_t)i * ldc + j] += acc[i * NR + j];
                        }
                    }
                }
                std::free(Ap);
            }
        }
    }
    std::free(Bp);
}

// y (M, int32) = A (MxK, int8) * x (K, int8): the batch-1 fully connected case.
inline void gemv(int M, int K, const int8_t* A, int lda, const int8_t* x, int32_t* y) {
    for (int i = 0; i < M; ++i) {
        const int8_t* a = A + (size_t)i * lda;
        int k = 0;
        int32_t acc = 0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        int32x4_t s0 = vdupq_n_s32(0), s1 = s0, s2 = s0, s3 = s0;
        for (; k + 64 <= K; k += 64) {
            s0 = vdotq_s32(s0, vld1q_s8(a + k), vld1q_s8(x + k));
            s1 = vdotq_s32(s1, vld1q_s8(a + k + 16), vld1q_s8(x + k + 16));
            s2 = vdotq_s32(s2, vld1q_s8(a + k + 32), vld1q_s8(x + k + 32));
            s3 = vdotq_s32(s3, vld1q_s8(a + k + 48), vld1q_s8(x + k + 48));
        }
        for (; k + 16 <= K; k += 16) s0 = vdotq_s32(s0, vld1q_s8(a + k), vld1q_s8(x + k));
        acc = vaddvq_s32(vaddq_s32(vaddq_s32(s0, s1), vaddq_s32(s2, s3)));
#endif
        for (; k < K; ++k) acc += (int32_t)a[k] * (int32_t)x[k];
        y[i] = acc;
    }
}

}  // namespace qgemm
