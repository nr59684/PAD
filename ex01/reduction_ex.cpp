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
#include "omp.h"
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

void setCustomCounter(benchmark::State& state, std::string name ) {
	state.counters["Elements"] = state.range(0);
	state.counters["Bytes"] = state.range(0) * sizeof(ValueType);
	state.SetLabel(name);
}

// Ex 1.1.1
static void benchReduceIterator(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  ValueType sum;
  for (auto _ : state) {
    sum = 0;
    for (auto x = X.begin(); x != X.end(); ++x) {
      sum += *x;
    }
    benchmark::DoNotOptimize(X.data());
	benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state,"Iterator");
}

static void benchReduceRange(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  ValueType sum;
  for (auto _ : state) {
    sum = 0;
    for (auto i : ranges::iota_view(0, state.range(0))) {
      sum += X[i];
    }
    benchmark::DoNotOptimize(X.data());
	benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state,"Range");
}

static void benchReduceRangeFor(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  ValueType sum;
  for (auto _ : state) {
    sum = 0;
    for (const auto& c : X) {
      sum += c;
    }
    benchmark::DoNotOptimize(X.data());
	benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state,"RangeFor");
}

// Ex 1.1.2
static void benchReduceStl(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  ValueType sum;
  for (auto _ : state) {
    sum = std::reduce(X.begin(), X.end());
    benchmark::DoNotOptimize(X.data());
	benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state,"Stl");
}

static void benchReduceSimdStl(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  ValueType sum;
  for (auto _ : state) {
    sum = std::reduce(std::execution::unseq, X.begin(), X.end());
    benchmark::DoNotOptimize(X.data());
	benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state,"SimdStl");
}

// Ex 1.2.1
static void benchReduceSimdUmeH(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  constexpr IndexType simd_width = 8;
  ValueType sum;
  for (auto _ : state) {
    sum = 0;
    UME::SIMD::SIMDVec<ValueType, simd_width> simd_vec;
    for (auto x = X.begin(); x != X.end(); x += simd_width) {
      simd_vec.load(&*x);
      sum += simd_vec.hadd();
    }
	benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state,"UmeH");
}

static void benchReduceSimdUmeH2(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  constexpr IndexType simd_width = 8;
  ValueType sum;
  for (auto _ : state) {
// TODO
// like UmeH, but X.size() is not necessarily a multiple of simd_width
// do not put any if-statements into loops
	benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state,"UmeH2");
}

static void benchReduceSimdUmeV(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  constexpr IndexType simd_width = 8;
  ValueType sum;
  UME::SIMD::SIMDVec<ValueType, simd_width> simd_sum;
  for (auto _ : state) {
    simd_sum = 0;
    UME::SIMD::SIMDVec<ValueType, simd_width> simd_vec;
    for (auto x = X.begin(); x != X.end(); x += simd_width) {
      simd_vec.load(&*x);
      simd_sum += simd_vec;
    }
	sum = simd_sum.hadd();
	benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state,"UmeV");
}

static void benchReduceSimdUmeV2(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  constexpr IndexType simd_width = 8;
  ValueType sum;
  UME::SIMD::SIMDVec<ValueType, simd_width> simd_sum;
  for (auto _ : state) {
// TODO
// like UmeV, but X.size() is not necessarily a multiple of simd_width
// do not put any if-statements into loops
	benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state,"UmeV2");
}

// Ex 1.2.2
static void benchReduceSimdOmpH(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  ValueType sum;
  for (auto _ : state) {
    sum = 0;
#pragma omp simd reduction(+ : sum)
    for (auto x = X.begin(); x != X.end(); ++x) {
      sum += *x;
    }
	benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state,"OmpH");
}

static void benchReduceSimdOmpV(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  constexpr IndexType simd_width = 8;
  ValueType sum;
  std::array<ValueType, simd_width> simd_sum;
  for (auto _ : state) {
    sum = 0;
    std::fill(simd_sum.begin(), simd_sum.end(), 0);
    for (auto x = X.begin(); x != X.end(); x += simd_width) {
#pragma omp simd
	  for (IndexType i = 0; i < simd_width; ++i) {
        simd_sum[i] += *(x+i);
	  }
    }
    sum = std::reduce(std::execution::unseq, simd_sum.begin(), simd_sum.end());
	benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state,"OmpV");
}

static void benchReduceSimdOmpV2(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  constexpr IndexType simd_width = 8;
  ValueType sum;
  std::array<ValueType, simd_width> simd_sum;
  for (auto _ : state) {
// TODO
// like OmpV, but X.size() is not necessarily a multiple of simd_width
// do not put any if-statements into loops
	benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state,"OmpV2");
}

// TODO add the *2 versions to the benchmarking
BENCHMARK(benchReduceIterator)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceRange)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceRangeFor)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceStl)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceSimdStl)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceSimdUmeH)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceSimdUmeV)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceSimdOmpH)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceSimdOmpV)->Apply(Args)->UseRealTime();

BENCHMARK_MAIN();
