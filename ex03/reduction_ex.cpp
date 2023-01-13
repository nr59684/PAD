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
#include "omp.h"
#include "oneapi/tbb.h"

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

static void GrainSizeArgs(benchmark::internal::Benchmark* b) {
  const auto lowerLimit = 25;
  const auto upperLimit = 25;
  const auto gLow = 0;
  const auto gHigh = 10;
  for (auto x = lowerLimit; x <= upperLimit; ++x) {
    for (auto y = gLow; y <= gHigh; ++y) {
      b->Args({1 << x, 1 << y});
    }
  }
}

void setCustomCounter(benchmark::State& state, std::string name) {
  state.counters["Elements"] = state.range(0);
  state.counters["Bytes"] = state.range(0) * sizeof(ValueType);
  state.SetLabel(name);
}

// TODO
// change the code such that elements = std::distance(Xbegin, Xend) is not necessarily a multiple of simd_width
// do not put any if-statements into loops
template <typename Iter, typename Partitioner>
typename std::iterator_traits<Iter>::value_type reduceTbb(Iter Xbegin,
                                                          Iter Xend,
                                                          Partitioner& part,
                                                          int grain_size = 1) {
  using value_type = typename std::iterator_traits<Iter>::value_type;
  using index_type = typename std::iterator_traits<Iter>::difference_type;
  constexpr index_type simd_width = 8;
  using simd_value_type = std::array<value_type, simd_width>;
  using namespace oneapi::tbb;
  auto elements = std::distance(Xbegin, Xend)/simd_width;
  
  simd_value_type simd_sum = parallel_reduce(
      blocked_range<index_type>(0, elements, grain_size), simd_value_type{0},
      [&](blocked_range<index_type>& r, simd_value_type simd_acc) {
        for (auto u = r.begin(); u != r.end(); ++u) {
	      auto U = u * simd_width;
#pragma omp simd
	      for (index_type i = 0; i < simd_width; ++i) {
            simd_acc[i] += Xbegin[U+i];
	      }		  
        }
        return simd_acc;
      },
      [](simd_value_type x, simd_value_type y) -> simd_value_type {
        simd_value_type sum;
#pragma omp simd
        for (int i = 0; i < simd_width; i++) sum[i] = x[i] + y[i];
        return sum;
      }, part);
  ValueType sum = std::reduce(std::execution::unseq, simd_sum.begin(), simd_sum.end());
  return sum;
}

// Ex 3.1.1
static void benchReduceIteratorScheduleStatic(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  ValueType sum;
  for (auto _ : state) {
    sum = 0;
#pragma omp parallel for simd reduction(+ : sum) schedule(static)
    for (auto x = X.begin(); x != X.end(); ++x) {
      sum += *x;
    }
    benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "IteratorStatic");
}

static void benchReduceIteratorScheduleDynamic(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  ValueType sum;
  for (auto _ : state) {
    sum = 0;
#pragma omp parallel for simd reduction(+ : sum) schedule(dynamic)
    for (auto x = X.begin(); x != X.end(); ++x) {
      sum += *x;
    }
    benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "IteratorDynamic");
}

static void benchReduceIteratorScheduleGuided(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  ValueType sum;
  for (auto _ : state) {
    sum = 0;
#pragma omp parallel for simd reduction(+ : sum) schedule(guided)
    for (auto x = X.begin(); x != X.end(); ++x) {
      sum += *x;
    }
    benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "IteratorGuided");
}

static void benchReduceRange(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  ValueType sum;
  for (auto _ : state) {
    sum = 0;
#pragma omp parallel for simd reduction(+ : sum)
    for (auto i : ranges::iota_view<int64_t, int64_t>(0, state.range(0))) {
      sum += X[i];
    }
    benchmark::DoNotOptimize(X.data());
    benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "Range");
}

static void benchReduceRangeFor(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  ValueType sum;
  for (auto _ : state) {
// TODO
// like Range but with for-range syntax
    benchmark::DoNotOptimize(X.data());
    benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "RangeFor");
}

static void benchReduceStl(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  ValueType sum;
  for (auto _ : state) {
    sum = std::reduce(std::execution::par, X.begin(), X.end());
    benchmark::DoNotOptimize(X.data());
    benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "Stl");
}

static void benchReduceStl2(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  ValueType sum;
  for (auto _ : state) {
// TODO
// like Stl, but use execution policy with parallelism AND SIMD
// do not put any if-statements into loops
    benchmark::DoNotOptimize(X.data());
    benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "Stl2");
}

static void benchReduceTbb(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  oneapi::tbb::auto_partitioner part;
  for (auto _ : state) {
    ValueType sum = reduceTbb(X.begin(), X.end(), part);
    benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "Tbb");
}

// Ex 3.2
// TODO
// Performance differences between partitioners
// Performance differences for different grain sizes

static void benchReduceTbbGrainSizeAuto(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  oneapi::tbb::auto_partitioner part;
  for (auto _ : state) {
    ValueType sum = reduceTbb(X.begin(), X.end(), part, state.range(1));
    benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "TbbAutoPartitioner");
}

static void benchReduceTbbGrainSizeSimple(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  oneapi::tbb::simple_partitioner part;
  for (auto _ : state) {
    ValueType sum = reduceTbb(X.begin(), X.end(), part, state.range(1));
    benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "TbbSimplePartitioner");
}

static void benchReduceTbbGrainSizeAffinity(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  oneapi::tbb::affinity_partitioner part;
  for (auto _ : state) {
    ValueType sum = reduceTbb(X.begin(), X.end(), part, state.range(1));
    benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "TbbAffinityPartitioner");
}

static void benchReduceTbbGrainSizeStatic(benchmark::State& state) {
  ContainerType X(state.range(0));
  std::iota(X.begin(), X.end(), ValueType{1});
  oneapi::tbb::static_partitioner part;
  for (auto _ : state) {
    ValueType sum = reduceTbb(X.begin(), X.end(), part, state.range(1));
    benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "TbbStaticPartitioner");
}


BENCHMARK(benchReduceStl)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceStl2)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceIteratorScheduleStatic)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceIteratorScheduleDynamic)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceIteratorScheduleGuided)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceRange)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceRangeFor)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceTbb)->Apply(Args)->UseRealTime();

BENCHMARK(benchReduceTbbGrainSizeAuto)->Apply(GrainSizeArgs)->UseRealTime();
BENCHMARK(benchReduceTbbGrainSizeSimple)->Apply(GrainSizeArgs)->UseRealTime();
BENCHMARK(benchReduceTbbGrainSizeStatic)->Apply(GrainSizeArgs)->UseRealTime();
BENCHMARK(benchReduceTbbGrainSizeAffinity)->Apply(GrainSizeArgs)->UseRealTime();
BENCHMARK_MAIN();
