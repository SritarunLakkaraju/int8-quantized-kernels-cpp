// engine.h - a tiny inference engine: loads a layer list plus raw weights
// exported from PyTorch and runs the network on the kernels from projects 1-2.
//
// Supported ops: conv (3x3, stride 1, pad 1, optional fused ReLU), maxpool
// (2x2), flatten, linear (optional ReLU). Enough for small CNNs and MLPs.
#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "conv.h"

namespace engine {

struct Tensor {
    std::vector<int> shape;
    util::avec<float> data;
};

// weights.bin: "TINY", u32 count, then per tensor:
//   u32 name_len, name bytes, u32 ndim, u32 dims[ndim], float32 data[prod(dims)]
inline std::map<std::string, Tensor> load_weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[4];
    uint32_t count = 0;
    f.read(magic, 4);
    f.read(reinterpret_cast<char*>(&count), 4);
    if (!f || std::string(magic, 4) != "TINY") throw std::runtime_error("bad weights file " + path);
    std::map<std::string, Tensor> tensors;
    for (uint32_t t = 0; t < count; ++t) {
        uint32_t len = 0, ndim = 0;
        f.read(reinterpret_cast<char*>(&len), 4);
        std::string name(len, '\0');
        f.read(&name[0], len);
        f.read(reinterpret_cast<char*>(&ndim), 4);
        Tensor ten;
        size_t n = 1;
        for (uint32_t d = 0; d < ndim; ++d) {
            uint32_t dim = 0;
            f.read(reinterpret_cast<char*>(&dim), 4);
            ten.shape.push_back((int)dim);
            n *= dim;
        }
        ten.data.resize(n);
        f.read(reinterpret_cast<char*>(ten.data.data()), (std::streamsize)(n * sizeof(float)));
        if (!f) throw std::runtime_error("truncated tensor " + name);
        tensors[name] = std::move(ten);
    }
    return tensors;
}

enum class ConvAlgo { Auto, Direct, Im2col, Winograd };

struct Layer {
    enum Kind { Conv, MaxPool, Flatten, Linear } kind;
    std::string name;
    bool relu = false;
    int C = 0, H = 0, W = 0;          // output shape (Linear/Flatten: C = features, H = W = 1)
    conv::Shape cs{};                 // Conv
    ConvAlgo algo = ConvAlgo::Auto;   // Conv, resolved at load time
    const float* w = nullptr;
    const float* b = nullptr;
    std::vector<float> U;             // Conv: Winograd-transformed kernel
    util::avec<float> wt;             // Linear: weight transposed to (in x out)
    int in_features = 0;
    size_t out_size() const { return (size_t)C * H * W; }
};

// Per-thread scratch memory, so one Model can serve many threads.
struct Workspace {
    util::avec<float> a, b, scratch;
    conv::WinogradScratch wino;
};

class Model {
public:
    void load(const std::string& dir, ConvAlgo algo = ConvAlgo::Auto) {
        tensors_ = load_weights(dir + "/weights.bin");
        std::ifstream f(dir + "/model.txt");
        if (!f) throw std::runtime_error("cannot open " + dir + "/model.txt");
        layers_.clear();
        int C = 0, H = 0, W = 0;
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            std::string op, name, act;
            if (!(ss >> op)) continue;
            Layer L;
            if (op == "input") {
                ss >> C >> H >> W;
                inC_ = C, inH_ = H, inW_ = W;
                continue;
            } else if (op == "conv") {
                ss >> name >> act;
                const Tensor& w = tensor(name + ".weight");
                if (w.shape.size() != 4 || w.shape[1] != C || w.shape[2] != 3 || w.shape[3] != 3)
                    throw std::runtime_error("conv " + name + ": only 3x3 kernels matching the input are supported");
                L.kind = Layer::Conv;
                L.cs = {C, H, W, w.shape[0], 3, 3, 1, 1};
                L.w = w.data.data();
                L.b = tensor(name + ".bias").data.data();
                // Few input channels: the direct kernel wins (nothing to gain from
                // lowering to GEMM, and bias + ReLU fuse into its store for free).
                // Otherwise Winograd, the fastest 3x3 path measured in project 2.
                L.algo = algo != ConvAlgo::Auto ? algo : (C < 8 ? ConvAlgo::Direct : ConvAlgo::Winograd);
                if (L.algo == ConvAlgo::Winograd) conv::winograd_transform_kernel(L.cs, L.w, L.U);
                C = w.shape[0];
            } else if (op == "maxpool") {
                L.kind = Layer::MaxPool;
                H /= 2, W /= 2;
            } else if (op == "flatten") {
                L.kind = Layer::Flatten;
                C = C * H * W, H = W = 1;
            } else if (op == "linear") {
                ss >> name >> act;
                const Tensor& w = tensor(name + ".weight");  // PyTorch stores (out x in)
                if (w.shape.size() != 2 || w.shape[1] != C) throw std::runtime_error("linear " + name + ": shape mismatch");
                L.kind = Layer::Linear;
                L.in_features = C;
                const int out = w.shape[0];
                L.wt.resize((size_t)C * out);  // transpose once so y = x * Wt is a plain GEMM
                for (int o = 0; o < out; ++o)
                    for (int i = 0; i < C; ++i) L.wt[(size_t)i * out + o] = w.data[(size_t)o * C + i];
                L.b = tensor(name + ".bias").data.data();
                C = out;
            } else {
                throw std::runtime_error("unknown op " + op);
            }
            L.name = name.empty() ? op : name;
            L.relu = act == "relu";
            L.C = C, L.H = H, L.W = W;
            max_act_ = std::max(max_act_, L.out_size());
            layers_.push_back(std::move(L));
        }
        max_act_ = std::max(max_act_, input_size());
    }

    size_t input_size() const { return (size_t)inC_ * inH_ * inW_; }
    size_t output_size() const { return layers_.back().out_size(); }
    const std::vector<Layer>& layers() const { return layers_; }

    // Runs one sample. If `layer_seconds` is given it accumulates time per layer.
    void forward(const float* x, float* out, Workspace& ws, double* layer_seconds = nullptr) const {
        ws.a.resize(max_act_);
        ws.b.resize(max_act_);
        const float* cur = x;
        float* next = ws.a.data();
        for (size_t li = 0; li < layers_.size(); ++li) {
            const Layer& L = layers_[li];
            const double t0 = layer_seconds ? util::now() : 0.0;
            float* dst = li + 1 == layers_.size() ? out : next;
            switch (L.kind) {
                case Layer::Conv: {
                    const int hw = L.H * L.W;
                    if (L.algo == ConvAlgo::Direct) {
                        conv::direct3x3(L.cs, cur, L.w, L.b, L.relu, dst, ws.scratch);
                    } else {
                        if (L.algo == ConvAlgo::Winograd)
                            conv::winograd3x3(L.cs, cur, L.U.data(), dst, ws.wino);
                        else
                            conv::im2col_gemm(L.cs, cur, L.w, dst, ws.scratch);
                        if (L.relu)
                            conv::bias_relu(L.C, hw, L.b, dst);
                        else
                            conv::add_bias(L.C, hw, L.b, dst);
                    }
                    break;
                }
                case Layer::MaxPool:
                    conv::maxpool2x2(L.C, L.H * 2, L.W * 2, cur, dst);
                    break;
                case Layer::Flatten:
                    continue;  // NCHW is already contiguous: a no-op, keep `cur`
                case Layer::Linear:
                    gemm::sgemm(1, L.C, L.in_features, cur, L.in_features, L.wt.data(), L.C, dst, L.C);
                    if (L.relu)
                        conv::bias_relu(L.C, 1, L.b, dst);
                    else
                        conv::add_bias(L.C, 1, L.b, dst);
                    break;
            }
            if (layer_seconds) layer_seconds[li] += util::now() - t0;
            cur = dst;
            next = dst == ws.a.data() ? ws.b.data() : ws.a.data();
        }
    }

private:
    const Tensor& tensor(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) throw std::runtime_error("missing tensor " + name);
        return it->second;
    }

    std::map<std::string, Tensor> tensors_;
    std::vector<Layer> layers_;
    int inC_ = 0, inH_ = 0, inW_ = 0;
    size_t max_act_ = 0;
};

template <class T>
std::vector<T> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const std::streamsize bytes = f.tellg();
    f.seekg(0);
    std::vector<T> v((size_t)bytes / sizeof(T));
    f.read(reinterpret_cast<char*>(v.data()), bytes);
    return v;
}

}  // namespace engine
