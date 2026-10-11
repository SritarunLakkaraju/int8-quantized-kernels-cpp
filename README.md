# INT8 quantised kernels in C++

An int8 x int8 -> int32 matrix multiply built on the ARM dot-product instruction, and post-training
quantisation of a small CNN, measured against FP32 for both speed and accuracy.

## Kernel speed

Square n x n multiply, in billions of multiply-add operations per second (same counting as GFLOPS):

| Kernel | n = 256 | n = 1024 | n = 2048 |
|---|---:|---:|---:|
| FP32 packed NEON, 1 thread | 100.3 | 109.8 | 110.5 |
| INT8 naive, 1 thread | 7.1 | 3.8 | - |
| INT8 widening NEON (no dot product), 1 thread | 39.5 | 40.2 | 39.5 |
| **INT8 SDOT packed, 1 thread** | 266.8 | 398.4 | **405.9** |
| FP32 packed, 11 threads | 240.7 | 626.2 | 663.6 |
| INT8 SDOT packed, 11 threads | 396.9 | 1792.7 | 2142.7 |

The SDOT kernel is 3.7x faster than FP32 at n = 2048. Int8 is not automatically fast: without a
dot-product instruction, the widening kernel is 2.8x *slower* than FP32. Integer arithmetic is
exact, so the int8 kernels are checked against each other bit for bit.

## Accuracy

The MNIST CNN from [tiny-inference-engine-cpp](https://github.com/SritarunLakkaraju/tiny-inference-engine-cpp) with all four weight layers quantised, calibrated on 1,000 images and
evaluated on the other 9,000:

| Metric | FP32 | INT8 |
|---|---:|---:|
| Accuracy | 98.73% | 98.73% |
| Predictions agreeing with FP32 | - | 99.90% |
| Mean absolute logit error | - | 0.028 |
| Weight storage | 808 KB | 203 KB |
| Latency per image (im2col + GEMM path) | 70.5 us | 58.6 us |

No measurable accuracy loss, but MNIST is an easy problem with wide margins, so this should not be
extrapolated to harder models. End to end the tiny network is only 1.20x faster in INT8: its
matrices are so small that packing and requantisation dominate, not the multiply.

Measured on an Apple M3 Pro (5 performance + 6 efficiency cores), Apple clang 21, `-O3 -mcpu=native`, no `-ffast-math`. Raw output is in `results/`.

## How it works

Affine quantisation, `real = scale * (q - zero_point)`: weights are symmetric int8 with one scale
per tensor, activations asymmetric int8 from min/max calibration, biases int32.

```
sum_k w_q[k] * (x_q[k] - zp)  =  sum_k w_q[k] * x_q[k]  -  zp * sum_k w_q[k]
                                 plain int8 GEMM            folded into the bias offline
```

The hot loop is therefore a pure int8 GEMM. One `SDOT` instruction performs 16 multiply-accumulates
into four int32 lanes; the micro-kernel keeps an 8x12 block of int32 results in 24 registers.

## Requirements

- macOS on Apple Silicon (the SIMD code is ARM NEON) with Apple clang
- Homebrew `libomp`

```sh
brew install libomp
```

The fast path needs the ARMv8.2 dot-product extension (all Apple Silicon has it).

## Build and run

```sh
make          # build bench_qgemm and quant_mnist
make run      # kernel benchmark, then quantise and evaluate the model -> results/
```

## Files

| File | Contents |
|---|---|
| `qgemm.h` | int8 `naive`, `widen` and `packed` (SDOT) GEMM; SDOT `gemv` |
| `bench_qgemm.cpp` | INT8 against FP32 kernel throughput |
| `quant_mnist.cpp` | Calibration, quantisation and int8 execution of the CNN |
| `model/` | The trained FP32 model and test set from [tiny-inference-engine-cpp](https://github.com/SritarunLakkaraju/tiny-inference-engine-cpp) |
| `deps/` | FP32 kernels and engine from the earlier projects |

## Limitations

- Requantisation multiplies by a float scale; an integer-only deployment would use a fixed-point multiplier.
- One scale per weight tensor and min/max calibration only.
- Calibration images come from the test set (the first 1,000) and are excluded from the accuracy figure.
- No x86 VNNI version.

## Part of a series

Five projects that build on each other:

1. [fast-gemm-cpp](https://github.com/SritarunLakkaraju/fast-gemm-cpp) - matrix multiply, the kernel everything else uses
2. [conv2d-kernels-cpp](https://github.com/SritarunLakkaraju/conv2d-kernels-cpp) - convolution on top of that GEMM
3. [tiny-inference-engine-cpp](https://github.com/SritarunLakkaraju/tiny-inference-engine-cpp) - a small inference engine on top of both
4. [int8-quantized-kernels-cpp](https://github.com/SritarunLakkaraju/int8-quantized-kernels-cpp) - the same ideas in 8-bit integers
5. [transformer-kernels-cpp](https://github.com/SritarunLakkaraju/transformer-kernels-cpp) - softmax, LayerNorm and attention

Each repository is self-contained: the headers it needs from the earlier projects are copied into `deps/`.
