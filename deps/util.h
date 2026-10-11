// util.h - timing, aligned buffers and deterministic random data shared by all projects.
#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

namespace util {

inline double now() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// Best-of-N wall time: repeat until `min_total` seconds have elapsed (and at
// least `min_reps` runs), return the fastest run. The minimum is the least
// noisy estimator for a deterministic kernel.
template <class F>
double time_best(F&& f, double min_total = 0.25, int min_reps = 3, int max_reps = 1000) {
    double best = 1e30, total = 0.0;
    for (int r = 0; r < max_reps && (r < min_reps || total < min_total); ++r) {
        const double t0 = now();
        f();
        const double dt = now() - t0;
        best = std::min(best, dt);
        total += dt;
    }
    return best;
}

template <class T>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() = default;
    template <class U>
    AlignedAllocator(const AlignedAllocator<U>&) {}
    T* allocate(size_t n) {
        void* p = nullptr;
        if (posix_memalign(&p, 64, std::max<size_t>(n, 1) * sizeof(T)) != 0) std::abort();
        return static_cast<T*>(p);
    }
    void deallocate(T* p, size_t) { std::free(p); }
    template <class U>
    bool operator==(const AlignedAllocator<U>&) const { return true; }
    template <class U>
    bool operator!=(const AlignedAllocator<U>&) const { return false; }
};

template <class T>
using avec = std::vector<T, AlignedAllocator<T>>;

// xorshift64*: fast, reproducible, good enough for benchmark inputs.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed = 0x9E3779B97F4A7C15ull) : s(seed) {}
    uint64_t next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 0x2545F4914F6CDD1Dull;
    }
    float uniform(float lo = -1.0f, float hi = 1.0f) {
        return lo + (hi - lo) * (float)((next() >> 40) * (1.0 / 16777216.0));
    }
};

inline void fill_uniform(float* p, size_t n, Rng& rng, float lo = -1.0f, float hi = 1.0f) {
    for (size_t i = 0; i < n; ++i) p[i] = rng.uniform(lo, hi);
}

inline double max_abs_diff(const float* a, const float* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

}  // namespace util
