// quant_mnist.cpp - post-training INT8 quantisation of the MNIST CNN from
// project 3, executed with the int8 GEMM kernels, and compared with FP32 for
// accuracy and latency.
//
// Scheme (the usual affine one, as in TFLite / QNNPACK):
//   real = scale * (q - zero_point)
//   weights      int8, symmetric (zero_point = 0), one scale per tensor
//   activations  int8, asymmetric, range from min/max calibration
//   bias         int32 at scale s_w * s_in
// For one layer:
//   sum_k w_q[k] * (x_q[k] - zp_in) = int8 GEMM  -  zp_in * sum_k w_q[k]
// The second term depends only on the weights, so it is folded into the bias
// offline and the hot loop stays a pure int8 x int8 -> int32 GEMM.
//
// usage: quant_mnist [model_dir] [results.json]
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "deps/engine.h"
#include "qgemm.h"

struct QParams {
    float scale = 1.0f;
    int zp = 0;
};

static QParams choose_qparams(float lo, float hi) {
    lo = std::min(lo, 0.0f), hi = std::max(hi, 0.0f);  // real 0 must be exactly representable
    QParams q;
    q.scale = (hi - lo) / 255.0f;
    q.zp = (int)std::lround(-128.0f - lo / q.scale);
    q.zp = std::max(-128, std::min(127, q.zp));
    return q;
}

static inline int8_t sat8(long v, int lo = -128) { return (int8_t)std::max<long>(lo, std::min<long>(127, v)); }

// Round to nearest; compiles to a single convert instruction, unlike lround.
static inline long rnd(float v) { return std::lrintf(v); }

struct QLayer {
    engine::Layer::Kind kind;
    std::string name;
    bool relu = false, last = false;
    conv::Shape cs{};
    int in_features = 0, out_features = 0;
    std::vector<int8_t> wq;      // (out x K)
    std::vector<int32_t> bq;     // bias with the zero-point correction folded in
    float w_scale = 1.0f;
    float mult = 1.0f;           // s_w * s_in / s_out : int32 accumulator -> int8 output
    float acc_scale = 1.0f;      // s_w * s_in         : int32 accumulator -> real value
    QParams in, out;
    double sqnr_db = 0.0;        // weight quantisation signal-to-noise ratio
};

// int8 im2col for 3x3 / stride 1 / pad 1; padding holds the zero point (real 0).
static void im2col_q(const conv::Shape& s, const int8_t* in, int8_t pad_value, int8_t* col) {
    const int OH = s.OH(), OW = s.OW();
    for (int ic = 0; ic < s.IC; ++ic)
        for (int ky = 0; ky < 3; ++ky)
            for (int kx = 0; kx < 3; ++kx) {
                int8_t* dst = col + ((size_t)(ic * 3 + ky) * 3 + kx) * OH * OW;
                // valid columns are those with 0 <= ow - pad + kx < W: one memcpy per row
                const int lo = std::max(0, s.pad - kx), hi = std::min(OW, s.W + s.pad - kx);
                for (int oh = 0; oh < OH; ++oh, dst += OW) {
                    const int ih = oh - s.pad + ky;
                    if (ih < 0 || ih >= s.H) {
                        std::memset(dst, pad_value, OW);
                        continue;
                    }
                    std::memset(dst, pad_value, OW);
                    std::memcpy(dst + lo, in + ((size_t)ic * s.H + ih) * s.W + lo - s.pad + kx, hi - lo);
                }
            }
}

struct QWorkspace {
    std::vector<int8_t> a, b, col;
    std::vector<int32_t> acc;
};

struct QModel {
    std::vector<QLayer> layers;
    QParams input;
    size_t in_size = 0, max_act = 0;
    size_t weight_bytes = 0, fp32_weight_bytes = 0;

    void forward(const float* x, float* logits, QWorkspace& ws) const {
        ws.a.resize(max_act);
        ws.b.resize(max_act);
        int8_t* cur = ws.a.data();
        int8_t* next = ws.b.data();
        const float inv_in = 1.0f / input.scale;
        for (size_t i = 0; i < in_size; ++i) cur[i] = sat8(rnd(x[i] * inv_in) + input.zp);

        for (const QLayer& L : layers) {
            switch (L.kind) {
                case engine::Layer::Conv: {
                    const int hw = L.cs.OH() * L.cs.OW(), K = L.cs.IC * 9;
                    ws.col.resize((size_t)K * hw);
                    ws.acc.resize((size_t)L.cs.OC * hw);
                    im2col_q(L.cs, cur, (int8_t)L.in.zp, ws.col.data());
                    qgemm::packed(L.cs.OC, hw, K, L.wq.data(), K, ws.col.data(), hw, ws.acc.data(), hw);
                    const int lo = L.relu ? L.out.zp : -128;  // ReLU is a clamp at the zero point
                    for (int oc = 0; oc < L.cs.OC; ++oc)
                        for (int p = 0; p < hw; ++p) {
                            const int32_t v = ws.acc[(size_t)oc * hw + p] + L.bq[oc];
                            next[(size_t)oc * hw + p] = sat8(rnd(v * L.mult) + L.out.zp, lo);
                        }
                    std::swap(cur, next);
                    break;
                }
                case engine::Layer::MaxPool: {  // max commutes with the monotonic quantisation map
                    const int C = L.cs.IC, H = L.cs.H, W = L.cs.W, OH = H / 2, OW = W / 2;
                    for (int c = 0; c < C; ++c)
                        for (int oh = 0; oh < OH; ++oh)
                            for (int ow = 0; ow < OW; ++ow) {
                                const int8_t* r = cur + ((size_t)c * H + 2 * oh) * W + 2 * ow;
                                next[((size_t)c * OH + oh) * OW + ow] = std::max(std::max(r[0], r[1]), std::max(r[W], r[W + 1]));
                            }
                    std::swap(cur, next);
                    break;
                }
                case engine::Layer::Flatten:
                    break;
                case engine::Layer::Linear: {
                    ws.acc.resize(L.out_features);
                    qgemm::gemv(L.out_features, L.in_features, L.wq.data(), L.in_features, cur, ws.acc.data());
                    if (L.last) {
                        for (int o = 0; o < L.out_features; ++o) logits[o] = (ws.acc[o] + L.bq[o]) * L.acc_scale;
                    } else {
                        const int lo = L.relu ? L.out.zp : -128;
                        for (int o = 0; o < L.out_features; ++o)
                            next[o] = sat8(rnd((ws.acc[o] + L.bq[o]) * L.mult) + L.out.zp, lo);
                        std::swap(cur, next);
                    }
                    break;
                }
            }
        }
    }
};

// FP32 forward that records the activation range after every layer.
static void calibrate(const engine::Model& m, const float* images, int count, std::vector<float>& lo,
                      std::vector<float>& hi, float& in_lo, float& in_hi) {
    const auto& layers = m.layers();
    lo.assign(layers.size(), 0.0f);
    hi.assign(layers.size(), 0.0f);
    in_lo = in_hi = 0.0f;
    util::avec<float> a, b, scratch;
    for (int n = 0; n < count; ++n) {
        const float* x = images + (size_t)n * m.input_size();
        for (size_t i = 0; i < m.input_size(); ++i) in_lo = std::min(in_lo, x[i]), in_hi = std::max(in_hi, x[i]);
        a.assign(x, x + m.input_size());
        for (size_t li = 0; li < layers.size(); ++li) {
            const engine::Layer& L = layers[li];
            b.resize(L.out_size());
            if (L.kind == engine::Layer::Conv) {
                conv::direct3x3(L.cs, a.data(), L.w, L.b, L.relu, b.data(), scratch);
            } else if (L.kind == engine::Layer::MaxPool) {
                conv::maxpool2x2(L.C, L.H * 2, L.W * 2, a.data(), b.data());
            } else if (L.kind == engine::Layer::Linear) {
                gemm::sgemm(1, L.C, L.in_features, a.data(), L.in_features, L.wt.data(), L.C, b.data(), L.C);
                L.relu ? conv::bias_relu(L.C, 1, L.b, b.data()) : conv::add_bias(L.C, 1, L.b, b.data());
            } else {
                b = a;
            }
            for (float v : b) lo[li] = std::min(lo[li], v), hi[li] = std::max(hi[li], v);
            a.swap(b);
        }
    }
}

static QModel quantize(const engine::Model& m, const std::vector<float>& lo, const std::vector<float>& hi, float in_lo,
                       float in_hi) {
    QModel q;
    q.in_size = m.input_size();
    q.max_act = m.input_size();
    q.input = choose_qparams(in_lo, in_hi);
    QParams cur = q.input;
    const auto& layers = m.layers();
    int C = 1, H = 28, W = 28;
    for (size_t li = 0; li < layers.size(); ++li) {
        const engine::Layer& L = layers[li];
        QLayer ql;
        ql.kind = L.kind;
        ql.name = L.name;
        ql.relu = L.relu;
        ql.last = li + 1 == layers.size();
        ql.in = cur;
        if (L.kind == engine::Layer::Conv || L.kind == engine::Layer::Linear) {
            const bool is_conv = L.kind == engine::Layer::Conv;
            const int out = L.C, K = is_conv ? L.cs.IC * 9 : L.in_features;
            ql.cs = L.cs;
            ql.in_features = K;
            ql.out_features = out;
            // Linear weights were transposed by the engine to (in x out); read them back as (out x in).
            auto weight = [&](int o, int k) { return is_conv ? L.w[(size_t)o * K + k] : L.wt[(size_t)k * out + o]; };
            float wmax = 0.0f;
            for (int o = 0; o < out; ++o)
                for (int k = 0; k < K; ++k) wmax = std::max(wmax, std::fabs(weight(o, k)));
            ql.w_scale = wmax / 127.0f;
            ql.wq.resize((size_t)out * K);
            ql.bq.resize(out);
            ql.out = choose_qparams(lo[li], hi[li]);
            ql.acc_scale = ql.w_scale * cur.scale;
            ql.mult = ql.acc_scale / ql.out.scale;
            double sig = 0.0, noise = 0.0;
            for (int o = 0; o < out; ++o) {
                long rowsum = 0;
                for (int k = 0; k < K; ++k) {
                    const float w = weight(o, k);
                    const int8_t wq = sat8(std::lround(w / ql.w_scale), -127);
                    ql.wq[(size_t)o * K + k] = wq;
                    rowsum += wq;
                    sig += (double)w * w;
                    noise += (double)(w - wq * ql.w_scale) * (w - wq * ql.w_scale);
                }
                ql.bq[o] = (int32_t)(std::lround(L.b[o] / ql.acc_scale) - cur.zp * rowsum);
            }
            ql.sqnr_db = 10.0 * std::log10(sig / noise);
            q.weight_bytes += (size_t)out * K + (size_t)out * 4;
            q.fp32_weight_bytes += ((size_t)out * K + out) * 4;
            cur = ql.out;
        } else if (L.kind == engine::Layer::MaxPool) {
            ql.cs.IC = C, ql.cs.H = H, ql.cs.W = W;  // input shape of the pool
        }
        C = L.C, H = L.H, W = L.W;
        q.max_act = std::max(q.max_act, L.out_size());
        q.layers.push_back(std::move(ql));
    }
    return q;
}

template <class F>
static double median_latency_us(int first, int last, F&& run) {
    std::vector<double> lat;
    for (int i = first; i < last; ++i) {
        double best = 1e30;
        for (int r = 0; r < 3; ++r) {
            const double t0 = util::now();
            run(i);
            best = std::min(best, util::now() - t0);
        }
        lat.push_back(best);
    }
    std::sort(lat.begin(), lat.end());
    return lat[lat.size() / 2] * 1e6;
}

int main(int argc, char** argv) try {
    const std::string dir = argc > 1 ? argv[1] : "model";
    const char* json_path = argc > 2 ? argv[2] : nullptr;
    const auto images = engine::read_file<float>(dir + "/test_images.bin");
    const auto labels = engine::read_file<uint8_t>(dir + "/test_labels.bin");
    const int N = (int)labels.size();
    const int CALIB = 1000;  // calibration images; accuracy is measured on the other 9000

    engine::Model fp_gemm, fp_auto;
    fp_gemm.load(dir, engine::ConvAlgo::Im2col);
    fp_auto.load(dir, engine::ConvAlgo::Auto);
    const size_t in_n = fp_gemm.input_size();

    std::vector<float> lo, hi;
    float in_lo, in_hi;
    calibrate(fp_gemm, images.data(), CALIB, lo, hi, in_lo, in_hi);
    const QModel qm = quantize(fp_gemm, lo, hi, in_lo, in_hi);

    engine::Workspace ws;
    QWorkspace qws;
    int fp_correct = 0, q_correct = 0, agree = 0;
    double sum_abs = 0.0, max_abs = 0.0;
    float fl[10], ql[10];
    for (int i = CALIB; i < N; ++i) {
        fp_gemm.forward(images.data() + i * in_n, fl, ws);
        qm.forward(images.data() + i * in_n, ql, qws);
        const int fa = (int)(std::max_element(fl, fl + 10) - fl), qa = (int)(std::max_element(ql, ql + 10) - ql);
        fp_correct += fa == labels[i];
        q_correct += qa == labels[i];
        agree += fa == qa;
        for (int c = 0; c < 10; ++c) {
            const double d = std::fabs(fl[c] - ql[c]);
            sum_abs += d;
            max_abs = std::max(max_abs, d);
        }
    }
    const int E = N - CALIB;

    float out[10];
    const double lat_q = median_latency_us(CALIB, N, [&](int i) { qm.forward(images.data() + i * in_n, out, qws); });
    const double lat_f = median_latency_us(CALIB, N, [&](int i) { fp_gemm.forward(images.data() + i * in_n, out, ws); });
    const double lat_a = median_latency_us(CALIB, N, [&](int i) { fp_auto.forward(images.data() + i * in_n, out, ws); });

    std::printf("Post-training INT8 quantisation of the MNIST CNN (%d calibration, %d evaluation images)\n", CALIB, E);
    std::printf("  input range [%.3f, %.3f] -> scale %.5f, zero point %d\n", in_lo, in_hi, qm.input.scale, qm.input.zp);
    for (const QLayer& L : qm.layers)
        if (!L.wq.empty())
            std::printf("  %-6s weight scale %.6f (SQNR %.1f dB), output scale %.5f, zero point %d\n", L.name.c_str(),
                        L.w_scale, L.sqnr_db, L.out.scale, L.out.zp);
    std::printf("  accuracy  FP32 %.2f%%   INT8 %.2f%%   (change %+.2f points)\n", 100.0 * fp_correct / E,
                100.0 * q_correct / E, 100.0 * (q_correct - fp_correct) / E);
    std::printf("  top-1 agreement INT8 vs FP32: %.2f%%   mean |logit error| %.4f   max %.3f\n", 100.0 * agree / E,
                sum_abs / (E * 10.0), max_abs);
    std::printf("  weights: FP32 %.1f KB -> INT8 %.1f KB (%.2fx smaller)\n", qm.fp32_weight_bytes / 1024.0,
                qm.weight_bytes / 1024.0, (double)qm.fp32_weight_bytes / qm.weight_bytes);
    std::printf("  latency (batch 1, 1 thread, median): INT8 %.1f us | FP32 im2col+GEMM %.1f us | FP32 engine default %.1f us\n",
                lat_q, lat_f, lat_a);

    if (json_path) {
        FILE* js = std::fopen(json_path, "w");
        std::fprintf(js,
                     "{\n  \"calibration_images\": %d,\n  \"eval_images\": %d,\n  \"fp32_accuracy\": %.4f,\n"
                     "  \"int8_accuracy\": %.4f,\n  \"agreement\": %.4f,\n  \"mean_abs_logit_error\": %.5f,\n"
                     "  \"max_abs_logit_error\": %.4f,\n  \"fp32_weight_kb\": %.1f,\n  \"int8_weight_kb\": %.1f,\n"
                     "  \"latency_us_int8\": %.2f,\n  \"latency_us_fp32_im2col\": %.2f,\n  \"latency_us_fp32_auto\": %.2f,\n"
                     "  \"layers\": {",
                     CALIB, E, (double)fp_correct / E, (double)q_correct / E, (double)agree / E, sum_abs / (E * 10.0),
                     max_abs, qm.fp32_weight_bytes / 1024.0, qm.weight_bytes / 1024.0, lat_q, lat_f, lat_a);
        bool first = true;
        for (const QLayer& L : qm.layers)
            if (!L.wq.empty()) {
                std::fprintf(js, "%s\"%s\": {\"w_scale\": %.6f, \"sqnr_db\": %.2f, \"out_scale\": %.5f, \"out_zp\": %d}",
                             first ? "" : ", ", L.name.c_str(), L.w_scale, L.sqnr_db, L.out.scale, L.out.zp);
                first = false;
            }
        std::fprintf(js, "}\n}\n");
        std::fclose(js);
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
