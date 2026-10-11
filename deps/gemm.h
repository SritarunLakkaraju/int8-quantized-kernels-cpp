// gemm.h - single-precision matrix multiply, from naive to BLIS-style.
//
// Every routine computes  C (MxN) = A (MxK) * B (KxN)  on row-major data.
// lda/ldb/ldc are the row strides (leading dimensions) of each matrix, so a
// routine can operate on a sub-matrix of a larger buffer.
//
// Header-only so the other projects (conv, inference engine, attention) can
// reuse the kernels with a single #include.
#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gemm {

// ---------------------------------------------------------------------------
// Step 0: naive triple loop (i, j, k).
// The inner loop walks B down a column: stride ldb, one cache miss per
// element once B no longer fits in cache.
// ---------------------------------------------------------------------------
inline void naive(int M, int N, int K, const float* A, int lda, const float* B, int ldb,
                  float* C, int ldc) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += A[i * lda + k] * B[k * ldb + j];
            C[i * ldc + j] = acc;
        }
}

inline void zero(int M, int N, float* C, int ldc) {
    for (int i = 0; i < M; ++i) std::memset(C + (size_t)i * ldc, 0, sizeof(float) * N);
}

// c[0..n) += a * b[0..n), deliberately scalar: auto-vectorisation is switched
// off (and the function kept out of line) so the scalar steps measure the
// memory-access pattern alone and share exactly the same inner-loop code.
__attribute__((noinline)) inline void axpy_scalar(int n, float a, const float* b, float* c) {
#pragma clang loop vectorize(disable)
    for (int j = 0; j < n; ++j) c[j] += a * b[j];
}

// ---------------------------------------------------------------------------
// Step 1: loop interchange (i, k, j).
// Same arithmetic, but the inner loop is now a unit-stride AXPY over a row of
// B and a row of C.
// ---------------------------------------------------------------------------
inline void reordered(int M, int N, int K, const float* A, int lda, const float* B, int ldb,
                      float* C, int ldc) {
    zero(M, N, C, ldc);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            axpy_scalar(N, A[i * lda + k], B + (size_t)k * ldb, C + (size_t)i * ldc);
}

// ---------------------------------------------------------------------------
// Step 2: loop tiling (cache blocking), still scalar.
// A TILE_K x TILE_J tile of B (64 KB) is reused for every row of A before
// moving on, so it stays resident in L1 instead of being re-fetched M times.
// ---------------------------------------------------------------------------
constexpr int TILE_K = 64;
constexpr int TILE_J = 256;

inline void tiled(int M, int N, int K, const float* A, int lda, const float* B, int ldb,
                  float* C, int ldc) {
    zero(M, N, C, ldc);
    for (int kk = 0; kk < K; kk += TILE_K) {
        const int kend = std::min(kk + TILE_K, K);
        for (int jj = 0; jj < N; jj += TILE_J) {
            const int jend = std::min(jj + TILE_J, N);
            for (int i = 0; i < M; ++i) {
                float* c = C + (size_t)i * ldc;
                for (int k = kk; k < kend; ++k)
                    axpy_scalar(jend - jj, A[i * lda + k], B + (size_t)k * ldb + jj, c + jj);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Step 3: SIMD intrinsics, with the tile sizes as parameters so the benchmark
// can run the same vector code with and without tiling.
// Four k-iterations are fused so each vector of C is loaded and stored once
// per four FMAs instead of once per FMA.
// ---------------------------------------------------------------------------
inline void simd_tiles(int M, int N, int K, const float* A, int lda, const float* B, int ldb,
                       float* C, int ldc, int tile_k, int tile_j) {
    zero(M, N, C, ldc);
    for (int kk = 0; kk < K; kk += tile_k) {
        const int kend = std::min(kk + tile_k, K);
        for (int jj = 0; jj < N; jj += tile_j) {
            const int jend = std::min(jj + tile_j, N);
            for (int i = 0; i < M; ++i) {
                float* c = C + (size_t)i * ldc;
                const float* a = A + (size_t)i * lda;
                int k = kk;
                for (; k + 4 <= kend; k += 4) {
                    const float* b0 = B + (size_t)k * ldb;
                    const float* b1 = b0 + ldb;
                    const float* b2 = b1 + ldb;
                    const float* b3 = b2 + ldb;
                    int j = jj;
#if defined(__ARM_NEON)
                    const float32x4_t a0 = vdupq_n_f32(a[k]), a1 = vdupq_n_f32(a[k + 1]);
                    const float32x4_t a2 = vdupq_n_f32(a[k + 2]), a3 = vdupq_n_f32(a[k + 3]);
                    for (; j + 8 <= jend; j += 8) {
                        float32x4_t c0 = vld1q_f32(c + j), c1 = vld1q_f32(c + j + 4);
                        c0 = vfmaq_f32(c0, a0, vld1q_f32(b0 + j));
                        c1 = vfmaq_f32(c1, a0, vld1q_f32(b0 + j + 4));
                        c0 = vfmaq_f32(c0, a1, vld1q_f32(b1 + j));
                        c1 = vfmaq_f32(c1, a1, vld1q_f32(b1 + j + 4));
                        c0 = vfmaq_f32(c0, a2, vld1q_f32(b2 + j));
                        c1 = vfmaq_f32(c1, a2, vld1q_f32(b2 + j + 4));
                        c0 = vfmaq_f32(c0, a3, vld1q_f32(b3 + j));
                        c1 = vfmaq_f32(c1, a3, vld1q_f32(b3 + j + 4));
                        vst1q_f32(c + j, c0);
                        vst1q_f32(c + j + 4, c1);
                    }
#elif defined(__AVX2__) && defined(__FMA__)
                    const __m256 a0 = _mm256_set1_ps(a[k]), a1 = _mm256_set1_ps(a[k + 1]);
                    const __m256 a2 = _mm256_set1_ps(a[k + 2]), a3 = _mm256_set1_ps(a[k + 3]);
                    for (; j + 8 <= jend; j += 8) {
                        __m256 c0 = _mm256_loadu_ps(c + j);
                        c0 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b0 + j), c0);
                        c0 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b1 + j), c0);
                        c0 = _mm256_fmadd_ps(a2, _mm256_loadu_ps(b2 + j), c0);
                        c0 = _mm256_fmadd_ps(a3, _mm256_loadu_ps(b3 + j), c0);
                        _mm256_storeu_ps(c + j, c0);
                    }
#endif
                    for (; j < jend; ++j)
                        c[j] += a[k] * b0[j] + a[k + 1] * b1[j] + a[k + 2] * b2[j] + a[k + 3] * b3[j];
                }
                for (; k < kend; ++k) {
                    const float ak = a[k];
                    const float* b = B + (size_t)k * ldb;
                    for (int j = jj; j < jend; ++j) c[j] += ak * b[j];
                }
            }
        }
    }
}

inline void simd_untiled(int M, int N, int K, const float* A, int lda, const float* B, int ldb,
                         float* C, int ldc) {
    simd_tiles(M, N, K, A, lda, B, ldb, C, ldc, K, N);
}

inline void simd(int M, int N, int K, const float* A, int lda, const float* B, int ldb,
                 float* C, int ldc) {
    simd_tiles(M, N, K, A, lda, B, ldb, C, ldc, TILE_K, TILE_J);
}

// ---------------------------------------------------------------------------
// Step 4: packing + register-blocked micro-kernel (the GotoBLAS/BLIS design).
//
//   for jc in N step NC            B slab      -> shared
//     for pc in K step KC          pack B      -> L2/L3 resident
//       for ic in M step MC        pack A      -> L2 resident   (parallel loop)
//         for jr in nc step NR     B micro-panel (KC x NR) -> L1 resident
//           for ir in mc step MR   micro-kernel: MR x NR block of C in registers
//
// The micro-kernel keeps the whole MR x NR block of C in vector registers for
// the full k-loop and reads both packed operands strictly sequentially.
// ---------------------------------------------------------------------------
#if defined(__ARM_NEON)
constexpr int MR = 8, NR = 12;   // 24 accumulators + 2 A + 3 B = 29 of 32 registers
#elif defined(__AVX2__) && defined(__FMA__)
constexpr int MR = 6, NR = 16;   // 12 accumulators + 2 B + 1 broadcast = 15 of 16
#else
constexpr int MR = 4, NR = 4;
#endif

#ifndef GEMM_KC
#define GEMM_KC 512
#endif
#ifndef GEMM_MC
#define GEMM_MC 480
#endif
#ifndef GEMM_NC
#define GEMM_NC 4080
#endif
// Tuned on an Apple M3 Pro (128 KB L1d, 16 MB L2 per performance cluster):
// a B micro-panel is KC*NR*4 = 24 KB (L1), a packed A block is MC*KC*4 = 1 MB (L2).
constexpr int KC = GEMM_KC, MC = GEMM_MC, NC = GEMM_NC;

// acc (MR x NR, row-major) = sum_p a[p*MR + i] * b[p*NR + j]
static inline void micro_kernel(int kc, const float* a, const float* b, float* acc) {
#if defined(__ARM_NEON)
    float32x4_t c00 = vdupq_n_f32(0), c01 = c00, c02 = c00, c10 = c00, c11 = c00, c12 = c00;
    float32x4_t c20 = c00, c21 = c00, c22 = c00, c30 = c00, c31 = c00, c32 = c00;
    float32x4_t c40 = c00, c41 = c00, c42 = c00, c50 = c00, c51 = c00, c52 = c00;
    float32x4_t c60 = c00, c61 = c00, c62 = c00, c70 = c00, c71 = c00, c72 = c00;
#define GEMM_ROW(r, av, lane)                      \
    c##r##0 = vfmaq_laneq_f32(c##r##0, b0, av, lane); \
    c##r##1 = vfmaq_laneq_f32(c##r##1, b1, av, lane); \
    c##r##2 = vfmaq_laneq_f32(c##r##2, b2, av, lane);
    for (int p = 0; p < kc; ++p) {
        const float32x4_t a0 = vld1q_f32(a), a1 = vld1q_f32(a + 4);
        const float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4), b2 = vld1q_f32(b + 8);
        GEMM_ROW(0, a0, 0) GEMM_ROW(1, a0, 1) GEMM_ROW(2, a0, 2) GEMM_ROW(3, a0, 3)
        GEMM_ROW(4, a1, 0) GEMM_ROW(5, a1, 1) GEMM_ROW(6, a1, 2) GEMM_ROW(7, a1, 3)
        a += MR;
        b += NR;
    }
#undef GEMM_ROW
#define GEMM_STORE(r)                    \
    vst1q_f32(acc + r * NR, c##r##0);     \
    vst1q_f32(acc + r * NR + 4, c##r##1); \
    vst1q_f32(acc + r * NR + 8, c##r##2);
    GEMM_STORE(0) GEMM_STORE(1) GEMM_STORE(2) GEMM_STORE(3)
    GEMM_STORE(4) GEMM_STORE(5) GEMM_STORE(6) GEMM_STORE(7)
#undef GEMM_STORE
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 c[MR][2];
    for (int i = 0; i < MR; ++i) c[i][0] = c[i][1] = _mm256_setzero_ps();
    for (int p = 0; p < kc; ++p) {
        const __m256 b0 = _mm256_loadu_ps(b), b1 = _mm256_loadu_ps(b + 8);
        for (int i = 0; i < MR; ++i) {
            const __m256 ai = _mm256_broadcast_ss(a + i);
            c[i][0] = _mm256_fmadd_ps(ai, b0, c[i][0]);
            c[i][1] = _mm256_fmadd_ps(ai, b1, c[i][1]);
        }
        a += MR;
        b += NR;
    }
    for (int i = 0; i < MR; ++i) {
        _mm256_storeu_ps(acc + i * NR, c[i][0]);
        _mm256_storeu_ps(acc + i * NR + 8, c[i][1]);
    }
#else
    for (int i = 0; i < MR * NR; ++i) acc[i] = 0.0f;
    for (int p = 0; p < kc; ++p, a += MR, b += NR)
        for (int i = 0; i < MR; ++i)
            for (int j = 0; j < NR; ++j) acc[i * NR + j] += a[i] * b[j];
#endif
}

// Pack an mc x kc block of A into MR-row panels: panel-major, then k, then row.
// Short panels are zero-padded so the micro-kernel never needs an edge case.
static inline void pack_A(int mc, int kc, const float* A, int lda, float* dst) {
    for (int ir = 0; ir < mc; ir += MR) {
        const int mr = std::min(MR, mc - ir);
        for (int p = 0; p < kc; ++p) {
            for (int i = 0; i < mr; ++i) dst[i] = A[(size_t)(ir + i) * lda + p];
            for (int i = mr; i < MR; ++i) dst[i] = 0.0f;
            dst += MR;
        }
    }
}

// Pack a kc x nc block of B into NR-column panels: panel-major, then k, then col.
static inline void pack_B(int kc, int nc, const float* B, int ldb, float* dst, bool parallel) {
    (void)parallel;  // only read by the OpenMP pragma
    const int npanels = (nc + NR - 1) / NR;
#pragma omp parallel for if (parallel) schedule(static)
    for (int pj = 0; pj < npanels; ++pj) {
        const int jr = pj * NR;
        const int nr = std::min(NR, nc - jr);
        float* d = dst + (size_t)pj * kc * NR;
        for (int p = 0; p < kc; ++p) {
            const float* src = B + (size_t)p * ldb + jr;
            for (int j = 0; j < nr; ++j) d[j] = src[j];
            for (int j = nr; j < NR; ++j) d[j] = 0.0f;
            d += NR;
        }
    }
}

static inline float* alloc_floats(size_t n) {
    void* p = nullptr;
    if (posix_memalign(&p, 64, std::max<size_t>(n, 16) * sizeof(float)) != 0) std::abort();
    return static_cast<float*>(p);
}

inline void packed_impl(int M, int N, int K, const float* A, int lda, const float* B, int ldb,
                        float* C, int ldc, bool parallel) {
    zero(M, N, C, ldc);
    if (M == 0 || N == 0 || K == 0) return;

    int mc_step = MC;
#ifdef _OPENMP
    // Below roughly 160^3 the fork/join cost outweighs the work being shared.
    if ((double)M * N * K < 4e6) parallel = false;
    if (parallel) {
        // Smaller row blocks give the dynamic scheduler enough chunks to balance
        // fast and slow cores (performance vs. efficiency cores).
        const int nt = omp_get_max_threads();
        int want = (M + 2 * nt - 1) / (2 * nt);
        want = ((want + MR - 1) / MR) * MR;
        mc_step = std::max(2 * MR, std::min(MC, want));
    }
#else
    parallel = false;
#endif

    const int nc_max = std::min(NC, N), kc_max = std::min(KC, K), mc_max = std::min(mc_step, M);
    float* Bp = alloc_floats((size_t)kc_max * (((size_t)nc_max + NR - 1) / NR) * NR);

    for (int jc = 0; jc < N; jc += NC) {
        const int nc = std::min(NC, N - jc);
        for (int pc = 0; pc < K; pc += KC) {
            const int kc = std::min(KC, K - pc);
            pack_B(kc, nc, B + (size_t)pc * ldb + jc, ldb, Bp, parallel);

#pragma omp parallel if (parallel)
            {
                float* Ap = alloc_floats((size_t)kc_max * (((size_t)mc_max + MR - 1) / MR) * MR);
                float acc[MR * NR] __attribute__((aligned(64)));
#pragma omp for schedule(dynamic, 1)
                for (int ic = 0; ic < M; ic += mc_step) {
                    const int mc = std::min(mc_step, M - ic);
                    pack_A(mc, kc, A + (size_t)ic * lda + pc, lda, Ap);
                    for (int jr = 0; jr < nc; jr += NR) {
                        const int nr = std::min(NR, nc - jr);
                        const float* bp = Bp + (size_t)(jr / NR) * kc * NR;
                        for (int ir = 0; ir < mc; ir += MR) {
                            const int mr = std::min(MR, mc - ir);
                            micro_kernel(kc, Ap + (size_t)(ir / MR) * kc * MR, bp, acc);
                            float* c = C + (size_t)(ic + ir) * ldc + jc + jr;
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

inline void packed(int M, int N, int K, const float* A, int lda, const float* B, int ldb,
                   float* C, int ldc) {
    packed_impl(M, N, K, A, lda, B, ldb, C, ldc, false);
}

// Step 5: the packed kernel with the ic loop spread over OpenMP threads.
inline void packed_omp(int M, int N, int K, const float* A, int lda, const float* B, int ldb,
                       float* C, int ldc) {
    packed_impl(M, N, K, A, lda, B, ldb, C, ldc, true);
}

// Convenience entry point used by the other projects: packing only pays for
// itself once the problem is big enough to amortise the copies.
inline void sgemm(int M, int N, int K, const float* A, int lda, const float* B, int ldb,
                  float* C, int ldc, bool parallel = false) {
    if ((double)M * N * K < 32.0 * 32.0 * 32.0 || M < 4)
        simd(M, N, K, A, lda, B, ldb, C, ldc);
    else
        packed_impl(M, N, K, A, lda, B, ldb, C, ldc, parallel);
}

}  // namespace gemm
