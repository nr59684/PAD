#include <vector>
#include <algorithm>
#include <thread>
#include <future>
#include <hwloc.h>
#include <iostream>
#include <execution>
#include <omp.h>
#include <benchmark/benchmark.h>

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/partitioner.h>

#include "arena.hpp"

using ValueType = float;
using ContainerTypeNoInit = std::vector<ValueType, numa::no_init_allocator<ValueType>>;
using Partitioner = tbb::static_partitioner;

static constexpr int thrds_per_node = 16;

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

static void benchTransformTbbNoInitV3(benchmark::State& state){
    numa::ArenaMgtTBB arenas(thrds_per_node);

    ContainerTypeNoInit X(state.range(0));
    ContainerTypeNoInit Y(state.range(0));
    Partitioner part;
    const ValueType alpha = 2;

    #pragma omp parallel
    {
        auto mth = omp_get_thread_num();
        auto [start, end] = arenas.index_range(mth, X.size());
        std::uninitialized_fill(std::execution::unseq, X.begin() + start, X.begin() + end, 1);
        std::uninitialized_fill(std::execution::unseq, Y.begin() + start, Y.begin() + end, 2);
    }

    for (auto _ : state){
        #pragma omp parallel
        {
            auto mth = omp_get_thread_num();
            auto [start, end] = arenas.index_range(mth, X.size());
            auto s = start;
            auto e = end;
            arenas[mth]->execute([&](){
                tbb::parallel_for(tbb::blocked_range<size_t>(s, e), [&](const tbb::blocked_range<size_t> r){
                    #pragma omp simd
                    for(size_t i = r.begin(); i < r.end(); i++){
                        Y[i] = alpha * X[i] + Y[i];
                    }
                }, part);
            });
        }

        benchmark::DoNotOptimize(Y.data());
        benchmark::ClobberMemory();
    }

    setCustomCounter(state, "TransformTbbNoInitV3");
}

static void benchTransformTbbNoInit2V3(benchmark::State& state){
    numa::ArenaMgtTBB arenas(thrds_per_node);    

    ContainerTypeNoInit X(state.range(0));
    ContainerTypeNoInit Y(state.range(0));
    Partitioner part;
    const ValueType alpha = 2;

    #pragma omp parallel
    {
        auto mth = omp_get_thread_num();
        auto [start, end] = arenas.index_range(mth, X.size());
        auto s = start;
        auto e = end;
        arenas[mth]->execute([&](){
            tbb::parallel_for(tbb::blocked_range<size_t>(s, e), [&](const tbb::blocked_range<size_t> r){
                for(size_t i = r.begin(); i < r.end(); i++){
                    new(&X[i]) ContainerTypeNoInit::value_type{1};
                    new(&X[i]) ContainerTypeNoInit::value_type{2};
                }
            }, part);                
        });
    }

    for (auto _ : state){
        #pragma omp parallel
        {
            auto mth = omp_get_thread_num();
            auto [start, end] = arenas.index_range(mth, X.size());
            auto s = start;
            auto e = end;
            arenas[mth]->execute([&](){
                tbb::parallel_for(tbb::blocked_range<size_t>(s, e), [&](const tbb::blocked_range<size_t> r){
                    #pragma omp simd
                    for(size_t i = r.begin(); i < r.end(); i++){
                        Y[i] = alpha * X[i] + Y[i];
                    }
                }, part);
            }
        );}


        benchmark::DoNotOptimize(Y.data());
        benchmark::ClobberMemory();      
    }

    setCustomCounter(state, "TransformTbbNoInit2V3");
}

BENCHMARK(benchTransformTbbNoInitV3)->Apply(Args)->UseRealTime()->Iterations(100);
BENCHMARK(benchTransformTbbNoInit2V3)->Apply(Args)->UseRealTime()->Iterations(100);
BENCHMARK_MAIN();