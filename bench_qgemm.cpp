// bench_qgemm.cpp - int8 GEMM kernels against the FP32 GEMM from project 1.
// "GOPS" counts a multiply and an add per term (2*n^3), the same convention as
// GFLOPS, so the integer and float numbers are directly comparable.
//
// usage: bench_qgemm [out.csv]
#include <omp.h>

#include <cstdio>
#include <string>
#include <vector>

#include "deps/gemm.h"
#include "deps/util.h"
#include "qgemm.h"

int main(int argc, char** argv) {
    FILE* csv = argc > 1 ? std::fopen(argv[1], "w") : nullptr;
    if (csv) std::fprintf(csv, "kernel,n,threads,seconds,gops\n");
    const std::vector<int> sizes = {128, 256, 512, 1024, 2048};
    const int max_threads = omp_get_max_threads();

    struct Row {
        std::string name;
        int threads, max_n;
        std::vector<double> gops;
    };
    std::vector<Row> rows = {{"fp32_packed", 1, 4096, {}},   {"int8_naive", 1, 1024, {}},
                             {"int8_widen", 1, 4096, {}},    {"int8_sdot_packed", 1, 4096, {}},
                             {"fp32_packed_omp", max_threads, 4096, {}}, {"int8_sdot_packed_omp", max_threads, 4096, {}}};

    util::Rng rng(3);
    for (int n : sizes) {
        const size_t nn = (size_t)n * n;
        util::avec<float> fa(nn), fb(nn), fc(nn);
        util::avec<int8_t> qa(nn), qb(nn);
        util::avec<int32_t> qc(nn), qref(nn);
        util::fill_uniform(fa.data(), nn, rng);
        util::fill_uniform(fb.data(), nn, rng);
        for (size_t i = 0; i < nn; ++i) {
            qa[i] = (int8_t)(rng.next() >> 56);
            qb[i] = (int8_t)(rng.next() >> 56);
        }
        omp_set_num_threads(max_threads);
        qgemm::packed(n, n, n, qa.data(), n, qb.data(), n, qref.data(), n, true);

        for (Row& r : rows) {
            if (n > r.max_n) {
                r.gops.push_back(-1);
                continue;
            }
            omp_set_num_threads(r.threads);
            double t = 0;
            bool is_int = true;
            if (r.name == "fp32_packed") {
                is_int = false;
                t = util::time_best([&] { gemm::packed(n, n, n, fa.data(), n, fb.data(), n, fc.data(), n); });
            } else if (r.name == "fp32_packed_omp") {
                is_int = false;
                t = util::time_best([&] { gemm::packed_omp(n, n, n, fa.data(), n, fb.data(), n, fc.data(), n); });
            } else if (r.name == "int8_naive") {
                t = util::time_best([&] { qgemm::naive(n, n, n, qa.data(), n, qb.data(), n, qc.data(), n); });
            } else if (r.name == "int8_widen") {
                t = util::time_best([&] { qgemm::widen(n, n, n, qa.data(), n, qb.data(), n, qc.data(), n); });
            } else if (r.name == "int8_sdot_packed") {
                t = util::time_best([&] { qgemm::packed(n, n, n, qa.data(), n, qb.data(), n, qc.data(), n, false); });
            } else {
                t = util::time_best([&] { qgemm::packed(n, n, n, qa.data(), n, qb.data(), n, qc.data(), n, true); });
            }
            // Integer arithmetic is exact: every int8 kernel must match bit for bit.
            if (is_int && std::memcmp(qc.data(), qref.data(), nn * sizeof(int32_t)) != 0)
                std::printf("MISMATCH: %s n=%d\n", r.name.c_str(), n);
            const double gops = 2.0 * n * n * n / t * 1e-9;
            r.gops.push_back(gops);
            if (csv) std::fprintf(csv, "%s,%d,%d,%.6e,%.2f\n", r.name.c_str(), n, r.threads, t, gops);
        }
    }

    std::printf("int8 vs fp32 GEMM, square n x n (G-ops/s, %d threads for *_omp)\n%-22s", max_threads, "kernel");
    for (int n : sizes) std::printf("%9d", n);
    std::printf("\n");
    for (const Row& r : rows) {
        std::printf("%-22s", r.name.c_str());
        for (double g : r.gops) g < 0 ? std::printf("%9s", "-") : std::printf("%9.1f", g);
        std::printf("\n");
    }
    if (csv) std::fclose(csv);
    return 0;
}
