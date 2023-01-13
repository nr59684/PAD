#include <benchmark/benchmark.h>  // google benchmark
#include <umesimd/UMESimd.h>
#include <algorithm>
#include <execution>
#include <numeric>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <array>
#include "range/v3/view.hpp"



using IndexType = ssize_t;
using ValueType = float;
using ContainerType = std::vector<ValueType>;

static void Args(benchmark::internal::Benchmark* b) {
  const auto lowerLimit = 15;
  const auto upperLimit = 30;

  for (auto x = lowerLimit; x <= upperLimit; ++x) {
    b->Args({1 << x});
  }
}

void setCustomCounter(benchmark::State& state, std::string name) {
  state.counters["Elements"] = state.range(0);
  state.counters["Bytes"] = 3 * state.range(0) * sizeof(ValueType);
  state.SetLabel(name);
}

// Ex 1.1.1
static void benchTransformIterator(benchmark::State& state) {
  ValueType a = -1;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);

  for (auto _ : state) {
    for (auto x = X.begin(), y = Y.begin(); x != X.end(); ++x, ++y) {
      *y = a * (*x) + *y;
    }
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "Iterator");
}

static void benchTransformIteratorInnerLoop(benchmark::State& state) {
  constexpr IndexType simd_width = 8;
  ValueType a = -1;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);

  for (auto _ : state) {
    for (auto x = X.begin(), y = Y.begin(); x != X.end();
         x += simd_width, y += simd_width) {
      for (IndexType i = 0; i < simd_width; ++i) {
        *(y + i) = a * (*(x + i)) + *(y + i);
      }
    }
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "IteratorInnerLoop");
}

static void benchTransformIteratorInnerLoop2(benchmark::State& state) {
  constexpr IndexType simd_width = 8;
  ValueType a = -1;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);

  for (auto _ : state) {
// TODO
// like IteratorInnerLoop, but X.size() is not necessarily a multiple of simd_width
// do not put any if-statements into loops
    }
    benchmark::ClobberMemory();
    setCustomCounter(state, "IteratorInnerLoop2");
  }
static void benchTransformRange(benchmark::State& state) {
  ValueType a = -1;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);

  for (auto _ : state) {
    for (auto i : ranges::iota_view(0, state.range(0))) {
      Y[i] = a * X[i] + Y[i];
    }
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "Range");
}

static void benchTransformRangeInnerLoop(benchmark::State& state) {
  constexpr IndexType simd_width = 8;
  ValueType a = -1;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);
  for (auto _ : state) {
    for (auto u : ranges::iota_view(0, state.range(0) / simd_width)) {
	  auto U = u * simd_width;
      for (IndexType i = 0; i < simd_width; ++i) {
        Y[U + i] = a * X[U + i] + Y[U + i];
      }
    }
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "RangeInnerLoop");
}

static void benchTransformRangeInnerLoop2(benchmark::State& state) {
  constexpr IndexType simd_width = 8;
  ValueType a = -1;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);
  for (auto _ : state) {
// TODO
// like RangeInnerLoop, but X.size() is not necessarily a multiple of simd_width
// do not put any if-statements into loops
    }
    benchmark::ClobberMemory();
    setCustomCounter(state, "RangeInnerLoop2");
  }

// Ex 1.1.2
static void benchTransformStl(benchmark::State& state) {
  ValueType a = -1;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);

  for (auto _ : state) {
    std::transform(X.begin(), X.end(), Y.begin(), Y.begin(),
                   [a](const auto x, const auto y) { return a * x + y; });
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "Stl");
}

// Ex 1.2.1
static void benchTransformSimdStl(benchmark::State& state) {
  ValueType a = -1;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);

  for (auto _ : state) {
    std::transform(std::execution::unseq, X.begin(), X.end(), Y.begin(),
                   Y.begin(),
                   [a](const auto x, const auto y) { return a * x + y; });
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "SimdStl");
}

static void benchUmeSimdTransform(benchmark::State& state) {
  constexpr IndexType simd_width = 8;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);

  for (auto _ : state) {
    UME::SIMD::SIMDVec<ValueType, simd_width> a_vec(-1);
    UME::SIMD::SIMDVec<ValueType, simd_width> x_vec, y_vec;
    for (auto x = X.begin(), y = Y.begin(); x != X.end();
         x += simd_width, y += simd_width) {
      x_vec.load(&*x);
      y_vec.load(&*y);
      y_vec = a_vec * x_vec + y_vec;
      y_vec.store(&*y);
    }
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "Ume");
}

static void benchUmeSimdTransform2(benchmark::State& state) {
  constexpr IndexType simd_width = 8;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);

  for (auto _ : state) {
// TODO
// like benchUmeSimdTransform, but X.size() is not necessarily a multiple of simd_width
// do not put any if-statements into loops
    }
    benchmark::ClobberMemory();
    setCustomCounter(state, "Ume2");
  }

// Ex 1.2.2
static void benchOmpSimdTransformIterator(benchmark::State& state) {
  ValueType a = -1;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);

  for (auto _ : state) {
    auto y = Y.begin();
#pragma omp simd
    for (auto x = X.begin(); x != X.end(); ++x) {
      *y = a * (*x) + *y;
      ++y;
    }
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "OmpIterator");
}
static void benchOmpSimdTransformIteratorInnerLoop(benchmark::State& state) {
  constexpr IndexType simd_width = 8;
  ValueType a = -1;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);

  for (auto _ : state) {
    auto y = Y.begin();
    for (auto x = X.begin(); x != X.end(); x += simd_width) {
#pragma omp simd
      for (IndexType i = 0; i < simd_width; ++i) {
        *(y + i) = a * (*(x + i)) + *(y + i);
      }
      y += simd_width;
    }
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "OmpIteratorInnerLoop");
}

static void benchOmpSimdTransformRange(benchmark::State& state) {
  ValueType a = -1;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);

  for (auto _ : state) {
#pragma omp simd
    for (auto i : ranges::iota_view<int64_t, int64_t>(0, state.range(0))) {
      Y[i] = a * X[i] + Y[i];
    }
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "OmpRange");
}

static void benchOmpSimdTransformRangeInnerLoop(benchmark::State& state) {
  constexpr IndexType simd_width = 8;
  ValueType a = -1;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);

  for (auto _ : state) {
    for (auto u : ranges::iota_view<int64_t, int64_t>(0, state.range(0) / simd_width)) {
      auto U = u * simd_width;
#pragma omp simd
      for (IndexType i = 0; i < simd_width; ++i) {
        Y[U + i] = a * X[U + i] + Y[U + i];
      }
    }
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "OmpRangeInnerLoop");
}

static void benchOmpSimdTransformRangeInnerLoop2(benchmark::State& state) {
  constexpr IndexType simd_width = 8;
  ValueType a = -1;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);

  for (auto _ : state) {
// TODO
// like OmpRangeInnerLoop, but X.size() is not necessarily a multiple of simd_width
// do not put any if-statements into loops
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "OmpRangeInnerLoop2");
}


// TODO add the *2 versions to the benchmarking
BENCHMARK(benchTransformIterator)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformIteratorInnerLoop)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformRange)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformRangeInnerLoop)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformStl)->Apply(Args)->UseRealTime();
BENCHMARK(benchOmpSimdTransformIterator)->Apply(Args)->UseRealTime();
BENCHMARK(benchOmpSimdTransformIteratorInnerLoop)->Apply(Args)->UseRealTime();
BENCHMARK(benchOmpSimdTransformRange)->Apply(Args)->UseRealTime();
BENCHMARK(benchOmpSimdTransformRangeInnerLoop)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformSimdStl)->Apply(Args)->UseRealTime();
BENCHMARK(benchUmeSimdTransform)->Apply(Args)->UseRealTime();
BENCHMARK_MAIN();
