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
  const auto lowerLimit = 28;
  const auto upperLimit = 28;
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
  state.counters["Bytes"] = 3 * state.range(0) * sizeof(ValueType);
  state.SetLabel(name);
}

template <typename Constant, typename Iter, typename Partitioner>
void transformTbb(Constant a,
                  Iter Xbegin,
                  Iter Xend,
                  Iter Ybegin,
                  Partitioner& part,
                  int grain_size = 1) {
  using index_type = typename std::iterator_traits<Iter>::difference_type;
  using namespace oneapi::tbb;
  auto elements = std::distance(Xbegin, Xend);
  parallel_for(
      blocked_range<index_type>(0, elements, grain_size),
      [&](const blocked_range<index_type>& r) {
#pragma omp simd
        for (auto i = r.begin(); i != r.end(); ++i) {
          Ybegin[i] = a * Xbegin[i] + Ybegin[i];
        }
      },
      part);
}

// Ex 3.1.1
static void benchTransformIteratorScheduleStatic(benchmark::State& state) {
  ValueType a = -1;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);
  for (auto _ : state) {
    auto y = Y.begin();
#pragma omp parallel for simd schedule(static)
    for (auto x = X.begin(); x != X.end(); ++x) {
      *y = a * (*x) + *y;
      ++y;
    }
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "IteratorStatic");
}

static void benchTransformIteratorScheduleDynamic(benchmark::State& state) {
  ValueType a = -1;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);
  for (auto _ : state) {
   auto y = Y.begin();
#pragma omp parallel for simd schedule(dynamic)
    for (auto x = X.begin(); x != X.end(); ++x) {
      *y = a * (*x) + *y;
      ++y;
    }
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "IteratorDynamic");
}

static void benchTransformIteratorScheduleGuided(benchmark::State& state) {
  ValueType a = -1;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);
  for (auto _ : state) {
   auto y = Y.begin();
#pragma omp parallel for simd schedule(guided)
    for (auto x = X.begin(); x != X.end(); ++x) {
      *y = a * (*x) + *y;
      ++y;
    }
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "IteratorGuided");
}

static void benchTransformRange(benchmark::State& state) {
  ValueType a = -1;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);
  for (auto _ : state) {
#pragma omp parallel for simd
    for (auto i : ranges::iota_view<int64_t, int64_t>(0, state.range(0))) {
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
#pragma omp parallel for
    for (auto u : ranges::iota_view<int64_t, int64_t>(0, state.range(0) / simd_width)) {
      auto U = u * simd_width;
#pragma omp simd
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
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "RangeInnerLoop2");
}


// Ex 3.1.2
static void benchTransformStl(benchmark::State& state) {
  ValueType a = -1;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);

  for (auto _ : state) {
    std::transform(std::execution::par, X.begin(), X.end(), Y.begin(),
                   Y.begin(),
                   [a](const auto x, const auto y) { return a * x + y; });
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "Stl");
}

static void benchTransformStl2(benchmark::State& state) {
  ValueType a = -1;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);

  for (auto _ : state) {
// TODO
// like Stl, but use execution policy with parallelism AND SIMD
// do not put any if-statements into loops
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "Stl2");
}

// Ex 1.1.2
static void benchTransformTbb(benchmark::State& state) {
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);
  oneapi::tbb::auto_partitioner part;
  for (auto _ : state) {
    transformTbb(-1, X.begin(), X.end(), Y.begin(), part);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "Tbb");
}

// Ex 3.2
// TODO
// Performance differences between partitioners
// Performance differences for different grain sizes

static void benchTransformTbbGrainSizeStatic(benchmark::State& state) {
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);
  oneapi::tbb::static_partitioner part;
  for (auto _ : state) {
    transformTbb(-1, X.begin(), X.end(), Y.begin(), part, state.range(1));
    benchmark::ClobberMemory();
  }
  state.counters["GrainSize"] = state.range(1);
  setCustomCounter(state, "TbbStaticPartitioner");
}

static void benchTransformTbbGrainSizeAuto(benchmark::State& state) {
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);
  oneapi::tbb::auto_partitioner part;
  for (auto _ : state) {
    transformTbb(-1, X.begin(), X.end(), Y.begin(), part, state.range(1));
    benchmark::ClobberMemory();
  }
  state.counters["GrainSize"] = state.range(1);
  setCustomCounter(state, "TbbAutoPartitioner");
}

static void benchTransformTbbGrainSizeAffinity(benchmark::State& state) {
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);
  oneapi::tbb::affinity_partitioner part;
  for (auto _ : state) {
    transformTbb(-1, X.begin(), X.end(), Y.begin(), part, state.range(1));
    benchmark::ClobberMemory();
  }
  state.counters["GrainSize"] = state.range(1);
  setCustomCounter(state, "TbbAffinityPartitioner");
}

static void benchTransformTbbGrainSizeSimple(benchmark::State& state) {
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);
  oneapi::tbb::simple_partitioner part;
  for (auto _ : state) {
    transformTbb(-1, X.begin(), X.end(), Y.begin(), part, state.range(1));
    benchmark::ClobberMemory();
  }
  state.counters["GrainSize"] = state.range(1);
  setCustomCounter(state, "TbbSimplePartitioner");
}

BENCHMARK(benchTransformIteratorScheduleStatic)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformIteratorScheduleDynamic)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformIteratorScheduleGuided)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformRange)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformRangeInnerLoop)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformRangeInnerLoop2)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformStl)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformStl2)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformTbb)->Apply(Args)->UseRealTime();

BENCHMARK(benchTransformTbbGrainSizeAuto)->Apply(GrainSizeArgs)->UseRealTime();
BENCHMARK(benchTransformTbbGrainSizeStatic)->Apply(GrainSizeArgs)->UseRealTime();
BENCHMARK(benchTransformTbbGrainSizeAffinity)->Apply(GrainSizeArgs)->UseRealTime();
BENCHMARK(benchTransformTbbGrainSizeSimple)->Apply(GrainSizeArgs)->UseRealTime();
BENCHMARK_MAIN();
