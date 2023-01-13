#include <benchmark/benchmark.h>  // google benchmark
#include <umesimd/UMESimd.h>
#include <algorithm>
#include <array>
#include <execution>
#include <iostream>
// #include <iterator>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include "allocator_adaptor.hpp"
#include "omp.h"
#include "oneapi/tbb.h"
#include "range/v3/view.hpp"

using IndexType = ssize_t;
using ValueType = float;
using ContainerType = std::vector<ValueType>;
using ContainerTypeDefaultInit =
    std::vector<ValueType, numa::default_init_allocator<ValueType>>;
using ContainerTypeNoInit =
    std::vector<ValueType, numa::no_init_allocator<ValueType>>;

static void Args(benchmark::internal::Benchmark* b) {
  const auto lowerLimit = 15;
  const auto upperLimit = 30;

  for (auto x = lowerLimit; x <= upperLimit; ++x) {
    b->Args({1 << x});
  }
}

void setCustomCounter(benchmark::State& state, std::string name) {
  state.counters["Elements"] = state.range(0);
  state.counters["Bytes"] = state.range(0) * sizeof(ValueType);
  state.SetLabel(name);
}

// Ex 4.1
struct ValueTypeCast {
  template <typename T>
  auto operator()(T idx) const noexcept {
    return static_cast<ValueType>(idx);
  }
};

static void benchReduceIteratorStd(benchmark::State& state) {
  auto iota =
      ranges::views::iota(1) | ranges::views::transform(ValueTypeCast{});
  // Impossible to get NUMA-friendly initialization without a default-init
  // allocator, so this is expected to perform badly
  ContainerType X(iota.begin(), iota.begin() + state.range(0));
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
  setCustomCounter(state, "IteratorStd");
}

static void benchReduceIteratorDefaultInit(benchmark::State& state) {
  ContainerTypeDefaultInit X(state.range(0));
  ValueType sum;
#pragma omp parallel for simd schedule(static)
  for (auto x = X.begin(); x != X.end(); ++x) {
    *x = x - X.begin() + 1;
  }

  for (auto _ : state) {
    sum = 0;
#pragma omp parallel for simd reduction(+ : sum) schedule(static)
    for (auto x = X.begin(); x != X.end(); ++x) {
      sum += *x;
    }
    benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "IteratorDefaultInit");
}

static void benchReduceIteratorNoInit(benchmark::State& state) {
  ContainerTypeNoInit X(state.range(0));
  ValueType sum;
  auto iota =
      ranges::views::iota(1) | ranges::views::transform(ValueTypeCast{});
  std::uninitialized_copy_n(std::execution::par_unseq, iota.begin(),
                            state.range(0), X.begin());

  for (auto _ : state) {
    sum = 0;
#pragma omp parallel for simd reduction(+ : sum) schedule(static)
    for (auto x = X.begin(); x != X.end(); ++x) {
      sum += *x;
    }
    benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "IteratorNoInit");
}

static void benchReduceIteratorNoInit2(benchmark::State& state) {
  ContainerTypeNoInit X(state.range(0));
  ValueType sum;
#pragma omp parallel for simd schedule(static)
  for (auto x = X.begin(); x != X.end(); ++x) {
    new (&*x) ContainerTypeNoInit::value_type{
        static_cast<ValueType>(x - X.begin() + 1)};
  }

  for (auto _ : state) {
    sum = 0;
#pragma omp parallel for simd reduction(+ : sum) schedule(static)
    for (auto x = X.begin(); x != X.end(); ++x) {
      sum += *x;
    }
    benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "IteratorNoInit2");
}

// Ex 4.2
static void benchReduceTbbStd(benchmark::State& state) {
  using namespace oneapi::tbb;
  auto iota =
      ranges::views::iota(1) | ranges::views::transform(ValueTypeCast{});
  // Impossible to get NUMA-friendly initialization without a default-init
  // allocator, so this is expected to perform badly
  ContainerType X(iota.begin(), iota.begin() + state.range(0));
  static_partitioner part;

  for (auto _ : state) {
    ValueType sum = parallel_reduce(
        blocked_range<IndexType>(0, X.size()), ValueType{0},
        [=](blocked_range<IndexType>& r, ValueType acc) {
#pragma omp simd reduction(+ : acc)
          for (auto i = r.begin(); i != r.end(); ++i) {
            acc += X[i];
          }
          return acc;
        },
        std::plus<ValueType>(), part);
    benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "TbbStd");
}

static void benchReduceTbbDefaultInit(benchmark::State& state) {
  using namespace oneapi::tbb;
  ContainerTypeDefaultInit X(state.range(0));
  static_partitioner part;
  parallel_for(
      blocked_range<IndexType>(0, state.range(0)),
      [&](const blocked_range<IndexType>& r) {
#pragma omp simd
        for (auto i = r.begin(); i != r.end(); ++i) {
          X[i] = ValueTypeCast{}(i + 1);
        }
      },
      part);

  for (auto _ : state) {
    ValueType sum = parallel_reduce(
        blocked_range<IndexType>(0, X.size()), ValueType{0},
        [=](blocked_range<IndexType>& r, ValueType acc) {
#pragma omp simd reduction(+ : acc)
          for (auto i = r.begin(); i != r.end(); ++i) {
            acc += X[i];
          }
          return acc;
        },
        std::plus<ValueType>(), part);
    benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "TbbDefaultInit");
}

static void benchReduceTbbNoInit(benchmark::State& state) {
  using namespace oneapi::tbb;
  ContainerTypeNoInit X(state.range(0));
  static_partitioner part;
  auto iota =
      ranges::views::iota(1) | ranges::views::transform(ValueTypeCast{});
  std::uninitialized_copy_n(std::execution::par_unseq, iota.begin(),
                            state.range(0), X.begin());

  for (auto _ : state) {
    ValueType sum = parallel_reduce(
        blocked_range<IndexType>(0, X.size()), ValueType{0},
        [=](blocked_range<IndexType>& r, ValueType acc) {
#pragma omp simd reduction(+ : acc)
          for (auto i = r.begin(); i != r.end(); ++i) {
            acc += X[i];
          }
          return acc;
        },
        std::plus<ValueType>(), part);
    benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "TbbNoInit");
}

static void benchReduceTbbNoInit2(benchmark::State& state) {
  using namespace oneapi::tbb;
  ContainerTypeNoInit X(state.range(0));
  static_partitioner part;
  parallel_for(
      blocked_range<IndexType>(0, state.range(0)),
      [&](const blocked_range<IndexType>& r) {
#pragma omp simd
        for (auto i = r.begin(); i != r.end(); ++i) {
          new (&X[i])
              ContainerTypeNoInit::value_type{static_cast<ValueType>(i + 1)};
        }
      },
      part);

  for (auto _ : state) {
    ValueType sum = parallel_reduce(
        blocked_range<IndexType>(0, X.size()), ValueType{0},
        [=](blocked_range<IndexType>& r, ValueType acc) {
#pragma omp simd reduction(+ : acc)
          for (auto i = r.begin(); i != r.end(); ++i) {
            acc += X[i];
          }
          return acc;
        },
        std::plus<ValueType>(), part);
    benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "TbbNoInit2");
}

BENCHMARK(benchReduceIteratorStd)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceIteratorDefaultInit)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceIteratorNoInit)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceIteratorNoInit2)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceTbbStd)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceTbbDefaultInit)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceTbbNoInit)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceTbbNoInit2)->Apply(Args)->UseRealTime();
BENCHMARK_MAIN();
