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
#include "allocator_adaptor.hpp"

using IndexType = ssize_t;
using ValueType = float;
using ContainerType = std::vector<ValueType>;
using ContainerTypeDefaultInit = std::vector<ValueType, numa::default_init_allocator<ValueType>>;
using ContainerTypeNoInit = std::vector<ValueType, numa::no_init_allocator<ValueType>>;

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

// Ex 4.1
static void benchTransformIteratorStd(benchmark::State& state) {
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
  setCustomCounter(state, "IteratorStd");
}

static void benchTransformIteratorStd2(benchmark::State& state) {
  ValueType a = -1;
  ContainerType X;
  ContainerType Y;
  X.resize(state.range(0), 1);
  Y.resize(state.range(0), 2);
  
  for (auto _ : state) {
    auto y = Y.begin();
#pragma omp parallel for simd schedule(static)
    for (auto x = X.begin(); x != X.end(); ++x) {
      *y = a * (*x) + *y;
      ++y;
    }
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "IteratorStd2");
}

static void benchTransformIteratorDefaultInit(benchmark::State& state) {
  ValueType a = -1;
  ContainerTypeDefaultInit X(state.range(0));
  ContainerTypeDefaultInit Y(state.range(0));
  std::fill(std::execution::par_unseq, X.begin(), X.end(), 1);
  std::fill(std::execution::par_unseq, Y.begin(), Y.end(), 2);

  for (auto _ : state) {
    auto y = Y.begin();
#pragma omp parallel for simd schedule(static)
    for (auto x = X.begin(); x != X.end(); ++x) {
      *y = a * (*x) + *y;
      ++y;
    }
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "IteratorDefaultInit");
}

static void benchTransformIteratorDefaultInit2(benchmark::State& state) {
  ValueType a = -1;
  ContainerTypeDefaultInit X(state.range(0));
  ContainerTypeDefaultInit Y(state.range(0));
  auto y = Y.begin();
#pragma omp parallel for simd schedule(static)
  for (auto x = X.begin(); x != X.end(); ++x) {
    *x = 1;
    *y = 2;
    ++y;
  }

  for (auto _ : state) {
    y = Y.begin();
#pragma omp parallel for simd schedule(static)
    for (auto x = X.begin(); x != X.end(); ++x) {
      *y = a * (*x) + *y;
      ++y;
    }
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "IteratorDefaultInit2");
}

static void benchTransformIteratorNoInit(benchmark::State& state) {
  ValueType a = -1;
  ContainerTypeNoInit X(state.range(0));
  ContainerTypeNoInit Y(state.range(0));
  std::uninitialized_fill(std::execution::par_unseq, X.begin(), X.end(), 1);
  std::uninitialized_fill(std::execution::par_unseq, Y.begin(), Y.end(), 2);

  for (auto _ : state) {
    auto y = Y.begin();
#pragma omp parallel for simd schedule(static)
    for (auto x = X.begin(); x != X.end(); ++x) {
      *y = a * (*x) + *y;
      ++y;
    }
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "IteratorNoInit");
}

static void benchTransformIteratorNoInit2(benchmark::State& state) {
  ValueType a = -1;
  ContainerTypeNoInit X(state.range(0));
  ContainerTypeNoInit Y(state.range(0));
  auto y = Y.begin();
#pragma omp parallel for simd schedule(static)
  for (auto x = X.begin(); x != X.end(); ++x) {
    new(&*x) ContainerTypeNoInit::value_type{1};
    new(&*y) ContainerTypeNoInit::value_type{2};
    ++y;
  }

  for (auto _ : state) {
    y = Y.begin();
#pragma omp parallel for simd schedule(static)
    for (auto x = X.begin(); x != X.end(); ++x) {
      *y = a * (*x) + *y;
      ++y;
    }
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "IteratorNoInit2");
}


// Ex 4.2
static void benchTransformTbbStd(benchmark::State& state) {
  using namespace oneapi::tbb;
  ValueType a = -1;
  ContainerType X(state.range(0), 1);
  ContainerType Y(state.range(0), 2);
  static_partitioner part;
  
  for (auto _ : state) {
    parallel_for(
        blocked_range<IndexType>(0, state.range(0)),
        [&](const blocked_range<IndexType>& r) {
#pragma omp simd
          for (auto i = r.begin(); i != r.end(); ++i) {
            Y[i] = a * X[i] + Y[i];
          }
        },
        part);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "TbbStd");
}

static void benchTransformTbbStd2(benchmark::State& state) {
  using namespace oneapi::tbb;
  ValueType a = -1;
  ContainerType X;
  ContainerType Y;
  static_partitioner part;
  X.resize(state.range(0), 1);
  Y.resize(state.range(0), 2);
  
  for (auto _ : state) {
    parallel_for(
        blocked_range<IndexType>(0, state.range(0)),
        [&](const blocked_range<IndexType>& r) {
#pragma omp simd
          for (auto i = r.begin(); i != r.end(); ++i) {
            Y[i] = a * X[i] + Y[i];
          }
        },
        part);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "TbbStd2");
}

static void benchTransformTbbDefaultInit(benchmark::State& state) {
  using namespace oneapi::tbb;
  ValueType a = -1;
  ContainerTypeDefaultInit X(state.range(0));
  ContainerTypeDefaultInit Y(state.range(0));
  static_partitioner part;
  std::fill(std::execution::par_unseq, X.begin(), X.end(), 1);
  std::fill(std::execution::par_unseq, Y.begin(), Y.end(), 2);  
  
  for (auto _ : state) {
    parallel_for(
        blocked_range<IndexType>(0, state.range(0)),
        [&](const blocked_range<IndexType>& r) {
#pragma omp simd
          for (auto i = r.begin(); i != r.end(); ++i) {
            Y[i] = a * X[i] + Y[i];
          }
        },
        part);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "TbbDefaultInit");
}

static void benchTransformTbbDefaultInit2(benchmark::State& state) {
  using namespace oneapi::tbb;
  ValueType a = -1;
  ContainerTypeDefaultInit X(state.range(0));
  ContainerTypeDefaultInit Y(state.range(0));
  static_partitioner part;
    parallel_for(
        blocked_range<IndexType>(0, state.range(0)),
        [&](const blocked_range<IndexType>& r) {
#pragma omp simd
          for (auto i = r.begin(); i != r.end(); ++i) {
            X[i] = 1;
			Y[i] = 2;
          }
        },
        part);  
  
  for (auto _ : state) {
    parallel_for(
        blocked_range<IndexType>(0, state.range(0)),
        [&](const blocked_range<IndexType>& r) {
#pragma omp simd
          for (auto i = r.begin(); i != r.end(); ++i) {
            Y[i] = a * X[i] + Y[i];
          }
        },
        part);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "TbbDefaultInit2");
}

static void benchTransformTbbNoInit(benchmark::State& state) {
  using namespace oneapi::tbb;
  ValueType a = -1;
  ContainerTypeNoInit X(state.range(0));
  ContainerTypeNoInit Y(state.range(0));
  static_partitioner part;
  std::uninitialized_fill(std::execution::par_unseq, X.begin(), X.end(), 1);
  std::uninitialized_fill(std::execution::par_unseq, Y.begin(), Y.end(), 2);
  
  for (auto _ : state) {
    parallel_for(
        blocked_range<IndexType>(0, state.range(0)),
        [&](const blocked_range<IndexType>& r) {
#pragma omp simd
          for (auto i = r.begin(); i != r.end(); ++i) {
            Y[i] = a * X[i] + Y[i];
          }
        },
        part);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "TbbNoInit");
}

static void benchTransformTbbNoInit2(benchmark::State& state) {
  using namespace oneapi::tbb;
  ValueType a = -1;
  ContainerTypeNoInit X(state.range(0));
  ContainerTypeNoInit Y(state.range(0));
  static_partitioner part;
    parallel_for(
        blocked_range<IndexType>(0, state.range(0)),
        [&](const blocked_range<IndexType>& r) {
#pragma omp simd
          for (auto i = r.begin(); i != r.end(); ++i) {
            new(&X[i]) ContainerTypeNoInit::value_type{1};
            new(&Y[i]) ContainerTypeNoInit::value_type{2};
          }
        },
        part);  
  
  for (auto _ : state) {
    parallel_for(
        blocked_range<IndexType>(0, state.range(0)),
        [&](const blocked_range<IndexType>& r) {
#pragma omp simd
          for (auto i = r.begin(); i != r.end(); ++i) {
            Y[i] = a * X[i] + Y[i];
          }
        },
        part);
    benchmark::ClobberMemory();
  }
  setCustomCounter(state, "TbbNoInit2");
}


BENCHMARK(benchTransformIteratorStd)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformIteratorStd2)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformIteratorDefaultInit)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformIteratorDefaultInit2)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformIteratorNoInit)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformIteratorNoInit2)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformTbbStd)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformTbbStd2)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformTbbDefaultInit)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformTbbDefaultInit2)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformTbbNoInit)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformTbbNoInit2)->Apply(Args)->UseRealTime();
BENCHMARK_MAIN();
