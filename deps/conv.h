// conv.h - 2D convolution kernels on NCHW float32 data (one image at a time).
//
//   direct        naive seven-deep loop nest, any kernel/stride/padding
//   direct3x3     NEON direct 3x3 kernel, optional fused bias + ReLU
//   im2col_gemm   lower the convolution to one matrix multiply (project 1)
//   winograd3x3   Winograd F(2x2, 3x3): 16 multiplies per 2x2 output tile
//                 instead of 36
//
// Weights are OIHW: w[oc][ic][ky][kx].
#pragma once

#include <algorithm>
#include <cstring>
#include <vector>

#include "gemm.h"
#include "util.h"

namespace conv {

struct Shape {
    int IC, H, W;        // input channels and spatial size
    int OC, KH, KW;      // output channels and kernel size
    int stride, pad;
    int OH() const { return (H + 2 * pad - KH) / stride + 1; }
    int OW() const { return (W + 2 * pad - KW) / stride + 1; }
    // multiply-adds a direct convolution performs, counted as 2 flops each
    double flops() const { return 2.0 * OC * OH() * OW() * IC * KH * KW; }
};

// ---------------------------------------------------------------------------
// Reference: direct convolution, written for clarity.
// ---------------------------------------------------------------------------
inline void direct(const Shape& s, const float* in, const float* w, const float* bias, float* out) {
    const int OH = s.OH(), OW = s.OW();
    for (int oc = 0; oc < s.OC; ++oc)
        for (int oh = 0; oh < OH; ++oh)
            for (int ow = 0; ow < OW; ++ow) {
                float acc = bias ? bias[oc] : 0.0f;
                for (int ic = 0; ic < s.IC; ++ic)
                    for (int ky = 0; ky < s.KH; ++ky) {
                        const int ih = oh * s.stride - s.pad + ky;
                        if (ih < 0 || ih >= s.H) continue;
                        for (int kx = 0; kx < s.KW; ++kx) {
                            const int iw = ow * s.stride - s.pad + kx;
                            if (iw < 0 || iw >= s.W) continue;
                            acc += in[((size_t)ic * s.H + ih) * s.W + iw] *
                                   w[(((size_t)oc * s.IC + ic) * s.KH + ky) * s.KW + kx];
                        }
                    }
                out[((size_t)oc * OH + oh) * OW + ow] = acc;
            }
}

// ---------------------------------------------------------------------------
// Separate bias and ReLU passes (the "unfused" pipeline), and the fused pass.
// ---------------------------------------------------------------------------
inline void add_bias(int OC, int hw, const float* bias, float* out) {
    for (int oc = 0; oc < OC; ++oc) {
        float* o = out + (size_t)oc * hw;
        const float b = bias[oc];
        for (int i = 0; i < hw; ++i) o[i] += b;
    }
}

inline void relu(size_t n, float* x) {
    for (size_t i = 0; i < n; ++i) x[i] = std::max(x[i], 0.0f);
}

inline void bias_relu(int OC, int hw, const float* bias, float* out) {
    for (int oc = 0; oc < OC; ++oc) {
        float* o = out + (size_t)oc * hw;
        const float b = bias[oc];
        for (int i = 0; i < hw; ++i) o[i] = std::max(o[i] + b, 0.0f);
    }
}

// Copy the input into a zero-bordered buffer so the hot loops need no bounds checks.
inline void pad_input(int IC, int H, int W, int pad, const float* in, int Hp, int Wp, float* dst) {
    std::memset(dst, 0, sizeof(float) * (size_t)IC * Hp * Wp);
    for (int ic = 0; ic < IC; ++ic)
        for (int h = 0; h < H; ++h)
            std::memcpy(dst + ((size_t)ic * Hp + h + pad) * Wp + pad, in + ((size_t)ic * H + h) * W,
                        sizeof(float) * W);
}

// ---------------------------------------------------------------------------
// Direct 3x3, stride 1, vectorised across the output row.
//
// NV vectors (4*NV outputs) are accumulated in registers across all input
// channels and taps. With `fused`, the accumulators start at the bias and are
// clamped by ReLU before the single store: bias and activation cost no extra
// pass over memory.
// ---------------------------------------------------------------------------
#if defined(__ARM_NEON)
template <int NV>
static inline void direct3x3_block(const float* in0, size_t plane, int Wp, int IC, const float* w,
                                   float bias, bool do_relu, float* out) {
    float32x4_t acc[NV];
    for (int v = 0; v < NV; ++v) acc[v] = vdupq_n_f32(bias);
    for (int ic = 0; ic < IC; ++ic) {
        const float* r = in0 + ic * plane;
        for (int ky = 0; ky < 3; ++ky, r += Wp)
            for (int kx = 0; kx < 3; ++kx) {
                const float32x4_t wv = vdupq_n_f32(*w++);
                for (int v = 0; v < NV; ++v) acc[v] = vfmaq_f32(acc[v], wv, vld1q_f32(r + kx + 4 * v));
            }
    }
    if (do_relu)
        for (int v = 0; v < NV; ++v) acc[v] = vmaxq_f32(acc[v], vdupq_n_f32(0.0f));
    for (int v = 0; v < NV; ++v) vst1q_f32(out + 4 * v, acc[v]);
}
#endif

// `bias` may be null. `scratch` is resized to hold the padded input.
inline void direct3x3(const Shape& s, const float* in, const float* w, const float* bias,
                      bool do_relu, float* out, util::avec<float>& scratch) {
    const int OH = s.OH(), OW = s.OW();
    const int Hp = s.H + 2 * s.pad, Wp = s.W + 2 * s.pad;
    const size_t plane = (size_t)Hp * Wp;
    scratch.resize((size_t)s.IC * plane + 16);
    pad_input(s.IC, s.H, s.W, s.pad, in, Hp, Wp, scratch.data());
    const float* x = scratch.data();

    // Row-outer order: the 3 input rows needed for one output row (all channels)
    // stay in L1 while every output channel consumes them.
    for (int oh = 0; oh < OH; ++oh) {
        const float* in0 = x + (size_t)oh * Wp;
        for (int oc = 0; oc < s.OC; ++oc) {
            const float* wk = w + (size_t)oc * s.IC * 9;
            const float b = bias ? bias[oc] : 0.0f;
            float* o = out + ((size_t)oc * OH + oh) * OW;
            int ow = 0;
#if defined(__ARM_NEON)
            for (; ow + 16 <= OW; ow += 16) direct3x3_block<4>(in0 + ow, plane, Wp, s.IC, wk, b, do_relu, o + ow);
            for (; ow + 8 <= OW; ow += 8) direct3x3_block<2>(in0 + ow, plane, Wp, s.IC, wk, b, do_relu, o + ow);
            for (; ow + 4 <= OW; ow += 4) direct3x3_block<1>(in0 + ow, plane, Wp, s.IC, wk, b, do_relu, o + ow);
#endif
            for (; ow < OW; ++ow) {
                float acc = b;
                const float* wp = wk;
                for (int ic = 0; ic < s.IC; ++ic) {
                    const float* r = in0 + ic * plane + ow;
                    for (int ky = 0; ky < 3; ++ky, r += Wp)
                        for (int kx = 0; kx < 3; ++kx) acc += *wp++ * r[kx];
                }
                o[ow] = do_relu ? std::max(acc, 0.0f) : acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// im2col: unfold every receptive field into a column so that
//   out (OC x OH*OW) = W (OC x IC*KH*KW) * col (IC*KH*KW x OH*OW)
// ---------------------------------------------------------------------------
inline void im2col(const Shape& s, const float* in, float* col) {
    const int OH = s.OH(), OW = s.OW();
    for (int ic = 0; ic < s.IC; ++ic)
        for (int ky = 0; ky < s.KH; ++ky)
            for (int kx = 0; kx < s.KW; ++kx) {
                float* dst = col + ((size_t)(ic * s.KH + ky) * s.KW + kx) * OH * OW;
                for (int oh = 0; oh < OH; ++oh, dst += OW) {
                    const int ih = oh * s.stride - s.pad + ky;
                    if (ih < 0 || ih >= s.H) {
                        std::memset(dst, 0, sizeof(float) * OW);
                        continue;
                    }
                    const float* src = in + ((size_t)ic * s.H + ih) * s.W;
                    if (s.stride == 1) {
                        // valid ow range is where 0 <= ow - pad + kx < W: one memcpy per row
                        const int lo = std::max(0, s.pad - kx), hi = std::min(OW, s.W + s.pad - kx);
                        for (int ow = 0; ow < lo; ++ow) dst[ow] = 0.0f;
                        if (hi > lo) std::memcpy(dst + lo, src + lo - s.pad + kx, sizeof(float) * (hi - lo));
                        for (int ow = std::max(hi, lo); ow < OW; ++ow) dst[ow] = 0.0f;
                    } else {
                        for (int ow = 0; ow < OW; ++ow) {
                            const int iw = ow * s.stride - s.pad + kx;
                            dst[ow] = (iw >= 0 && iw < s.W) ? src[iw] : 0.0f;
                        }
                    }
                }
            }
}

inline void im2col_gemm(const Shape& s, const float* in, const float* w, float* out,
                        util::avec<float>& scratch, bool parallel = false) {
    const int hw = s.OH() * s.OW(), ckk = s.IC * s.KH * s.KW;
    if (s.KH == 1 && s.KW == 1 && s.stride == 1 && s.pad == 0) {  // 1x1: the input already is the column matrix
        gemm::sgemm(s.OC, hw, ckk, w, ckk, in, hw, out, hw, parallel);
        return;
    }
    scratch.resize((size_t)ckk * hw);
    im2col(s, in, scratch.data());
    gemm::sgemm(s.OC, hw, ckk, w, ckk, scratch.data(), hw, out, hw, parallel);
}

// ---------------------------------------------------------------------------
// Winograd F(2x2, 3x3)  (Lavin & Gray, 2016), 3x3 kernels with stride 1.
//
//   Y = A^T [ (G g G^T) .* (B^T d B) ] A
//
// d is a 4x4 input tile (tiles overlap by 2), g the 3x3 kernel, Y the 2x2
// output tile. The elementwise product needs 16 multiplies where the direct
// method needs 36. Summing over input channels happens in the transformed
// domain, so for each of the 16 positions it is one GEMM:
//   M[xi] (OC x T) = U[xi] (OC x IC) * V[xi] (IC x T),   T = number of tiles.
// ---------------------------------------------------------------------------

// U = G g G^T for every (oc, ic); done once, offline. U is 16 matrices of OC x IC.
inline void winograd_transform_kernel(const Shape& s, const float* w, std::vector<float>& U) {
    static const float G[4][3] = {{1, 0, 0}, {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0, 0, 1}};
    const size_t n = (size_t)s.OC * s.IC;
    U.assign(16 * n, 0.0f);
    for (size_t e = 0; e < n; ++e) {
        const float* g = w + e * 9;
        float t[4][3];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 3; ++j) t[i][j] = G[i][0] * g[j] + G[i][1] * g[3 + j] + G[i][2] * g[6 + j];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                U[(size_t)(i * 4 + j) * n + e] = t[i][0] * G[j][0] + t[i][1] * G[j][1] + t[i][2] * G[j][2];
    }
}

// B^T d B. Written on a generic value type: instantiated with float for the
// scalar tail and with float32x4_t to transform four tiles at once.
template <class V>
static inline void winograd_in(const V d[4][4], V v[16]) {
    V t[4][4];
    for (int j = 0; j < 4; ++j) {  // B^T d : combine rows
        t[0][j] = d[0][j] - d[2][j];
        t[1][j] = d[1][j] + d[2][j];
        t[2][j] = d[2][j] - d[1][j];
        t[3][j] = d[1][j] - d[3][j];
    }
    for (int i = 0; i < 4; ++i) {  // (.) B : combine columns
        v[i * 4 + 0] = t[i][0] - t[i][2];
        v[i * 4 + 1] = t[i][1] + t[i][2];
        v[i * 4 + 2] = t[i][2] - t[i][1];
        v[i * 4 + 3] = t[i][1] - t[i][3];
    }
}

// A^T m A : 4x4 -> 2x2.
template <class V>
static inline void winograd_out(const V m[16], V y[4]) {
    V t[2][4];
    for (int j = 0; j < 4; ++j) {
        t[0][j] = m[0 + j] + m[4 + j] + m[8 + j];
        t[1][j] = m[4 + j] - m[8 + j] - m[12 + j];
    }
    for (int i = 0; i < 2; ++i) {
        y[i * 2 + 0] = t[i][0] + t[i][1] + t[i][2];
        y[i * 2 + 1] = t[i][1] - t[i][2] - t[i][3];
    }
}

struct WinogradScratch {
    util::avec<float> padded, V, M;
};

inline void winograd3x3(const Shape& s, const float* in, const float* U, float* out,
                        WinogradScratch& ws, bool parallel = false) {
    const int OH = s.OH(), OW = s.OW();
    const int TH = (OH + 1) / 2, TW = (OW + 1) / 2, T = TH * TW;
    const int TWv = (TW + 3) & ~3;                     // tile columns rounded up to the vector width
    const int Hp = 2 * TH + 2, Wp = 2 * TWv + 2;       // padded input covers every (vectorised) tile
    const size_t plane = (size_t)Hp * Wp;
    const size_t vstride = (size_t)s.IC * T + 4, mstride = (size_t)s.OC * T + 4;  // +4: vector over-run slack

    ws.padded.resize((size_t)s.IC * plane);
    ws.V.resize(16 * vstride);
    ws.M.resize(16 * mstride);
    pad_input(s.IC, s.H, s.W, s.pad, in, Hp, Wp, ws.padded.data());

    // 1. Input transform: V[xi][ic][tile] = (B^T d B)[xi]
    for (int ic = 0; ic < s.IC; ++ic)
        for (int th = 0; th < TH; ++th) {
            const float* r = ws.padded.data() + (size_t)ic * plane + (size_t)(2 * th) * Wp;
            const size_t base = (size_t)ic * T + (size_t)th * TW;
            int tw = 0;
#if defined(__ARM_NEON)
            // vld2q de-interleaves 8 floats into even/odd columns: exactly the
            // stride-2 tile layout, so four neighbouring tiles load at once.
            // The last vector of a row may run past TW; those lanes are
            // rewritten by the next row (or land in the slack).
            for (; tw < TW; tw += 4) {
                float32x4_t d[4][4], v[16];
                for (int i = 0; i < 4; ++i) {
                    const float32x4x2_t a = vld2q_f32(r + (size_t)i * Wp + 2 * tw);
                    const float32x4x2_t b = vld2q_f32(r + (size_t)i * Wp + 2 * tw + 2);
                    d[i][0] = a.val[0];
                    d[i][1] = a.val[1];
                    d[i][2] = b.val[0];
                    d[i][3] = b.val[1];
                }
                winograd_in(d, v);
                for (int xi = 0; xi < 16; ++xi) vst1q_f32(ws.V.data() + xi * vstride + base + tw, v[xi]);
            }
#endif
            for (; tw < TW; ++tw) {
                float d[4][4], v[16];
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j) d[i][j] = r[(size_t)i * Wp + 2 * tw + j];
                winograd_in(d, v);
                for (int xi = 0; xi < 16; ++xi) ws.V[xi * vstride + base + tw] = v[xi];
            }
        }

    // 2. Sixteen GEMMs: the only place multiplications by weights happen.
    const size_t un = (size_t)s.OC * s.IC;
    for (int xi = 0; xi < 16; ++xi)
        gemm::sgemm(s.OC, T, s.IC, U + xi * un, s.IC, ws.V.data() + xi * vstride, T,
                    ws.M.data() + xi * mstride, T, parallel);

    // 3. Output transform: Y = A^T M A, scattered back as 2x2 tiles.
    for (int oc = 0; oc < s.OC; ++oc)
        for (int th = 0; th < TH; ++th) {
            float* o0 = out + ((size_t)oc * OH + 2 * th) * OW;
            const bool row1 = 2 * th + 1 < OH;
            const size_t base = (size_t)oc * T + (size_t)th * TW;
            int tw = 0;
#if defined(__ARM_NEON)
            for (; tw + 4 <= TW && 2 * tw + 8 <= OW; tw += 4) {
                float32x4_t m[16], y[4];
                for (int xi = 0; xi < 16; ++xi) m[xi] = vld1q_f32(ws.M.data() + xi * mstride + base + tw);
                winograd_out(m, y);
                const float32x4x2_t top = {{y[0], y[1]}}, bottom = {{y[2], y[3]}};
                vst2q_f32(o0 + 2 * tw, top);  // re-interleave even/odd columns
                if (row1) vst2q_f32(o0 + OW + 2 * tw, bottom);
            }
#endif
            for (; tw < TW; ++tw) {
                float m[16], y[4];
                for (int xi = 0; xi < 16; ++xi) m[xi] = ws.M[xi * mstride + base + tw];
                winograd_out(m, y);
                const bool col1 = 2 * tw + 1 < OW;
                o0[2 * tw] = y[0];
                if (col1) o0[2 * tw + 1] = y[1];
                if (row1) {
                    o0[OW + 2 * tw] = y[2];
                    if (col1) o0[OW + 2 * tw + 1] = y[3];
                }
            }
        }
}

// ---------------------------------------------------------------------------
// Max pooling 2x2 stride 2 (used by the inference engine).
// ---------------------------------------------------------------------------
inline void maxpool2x2(int C, int H, int W, const float* in, float* out) {
    const int OH = H / 2, OW = W / 2;
    for (int c = 0; c < C; ++c)
        for (int oh = 0; oh < OH; ++oh) {
            const float* r0 = in + ((size_t)c * H + 2 * oh) * W;
            const float* r1 = r0 + W;
            float* o = out + ((size_t)c * OH + oh) * OW;
            for (int ow = 0; ow < OW; ++ow)
                o[ow] = std::max(std::max(r0[2 * ow], r0[2 * ow + 1]), std::max(r1[2 * ow], r1[2 * ow + 1]));
        }
}

}  // namespace conv
