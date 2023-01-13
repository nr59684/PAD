#include <vector>
#include <algorithm>
#include <thread>
#include <future>
#include <hwloc.h>
#include <iostream>
#include <execution>
#include <omp.h>
#include <benchmark/benchmark.h>

#include <oneapi/tbb/parallel_reduce.h>
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
  state.counters["Bytes"] = state.range(0) * sizeof(ValueType);
  state.SetLabel(name);
}



static void benchReduceTbbNoInitV3(benchmark::State& state){
    numa::ArenaMgtTBB arenas(thrds_per_node);
    ContainerTypeNoInit X(state.range(0));

    #pragma omp parallel 
    {
        auto mth = omp_get_thread_num();
        auto [start, end] = arenas.index_range(mth, X.size());
        std::uninitialized_fill(std::execution::unseq, X.begin() + start, X.begin() + end, 1);
    }
    Partitioner part;

    std::atomic<ValueType> total_sum;
    ValueType part_sum;
    for (auto _ : state){
        total_sum = 0; 
        #pragma omp parallel private(part_sum) shared(total_sum)  
        {
            auto mth = omp_get_thread_num();
            auto [start, end] = arenas.index_range(mth, X.size());
            auto s = start;                                         // icp won't let us capture structural bindings directly
	        auto e = end;
	        part_sum = arenas[mth]->execute([&]() -> ValueType {
                ValueType sum = tbb::parallel_reduce(tbb::blocked_range<size_t>(s, e), 0,
                                                [&](const tbb::blocked_range<size_t> r, ValueType ret) -> ValueType {
                                                    #pragma omp simd reduction(+ : ret)
                                                    for (size_t i = r.begin(); i < r.end(); i++){
                                                        ret += X[i];
                                                    }
                                                    return ret;
                                                }, std::plus<ValueType>(), part);
                return sum;
            });
            total_sum += part_sum;
        }
        
    benchmark::DoNotOptimize(&total_sum);
    benchmark::ClobberMemory();
    }
    if (total_sum != static_cast<ValueType>(state.range(0))) std::cout << "wrong result" << std::endl;
    setCustomCounter(state, "ReduceTbbNoInitV3");
}

static void benchReduceTbbNoInit2V3(benchmark::State& state){
    numa::ArenaMgtTBB arenas(thrds_per_node);

    ContainerTypeNoInit X(state.range(0));
    Partitioner part;

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
                }
            }, part);                 
        });
    }

    ValueType part_sum;
    std::atomic<ValueType> total_sum;
    for (auto _ : state){
        total_sum = 0;
        #pragma omp parallel private(part_sum) shared(total_sum)
        {
            auto mth = omp_get_thread_num();
            auto [start, end] = arenas.index_range(mth, X.size());
            auto s = start;                                         // icp won't let us capture structural bindings directly
            auto e = end;
            part_sum = arenas[mth]->execute([&]() -> ValueType {
                ValueType sum = tbb::parallel_reduce(tbb::blocked_range<size_t>(s, e), 0,
                                                [&](const tbb::blocked_range<size_t> r, ValueType ret) -> ValueType {
                                                    #pragma omp simd reduction(+ : ret)
                                                    for (size_t i = r.begin(); i < r.end(); i++){
                                                        ret += X[i];
                                                    }
                                                    return ret;
                                                }, std::plus<ValueType>(), part);
                return sum;
            });
            total_sum += part_sum;
        }  

        benchmark::DoNotOptimize(&total_sum);
        benchmark::ClobberMemory();
    }
    if (total_sum != static_cast<ValueType>(state.range(0))) std::cout << "wrong result" << std::endl;
    setCustomCounter(state, "ReduceTbbNoInit2V3");
}

BENCHMARK(benchReduceTbbNoInitV3)->Apply(Args)->UseRealTime()->Iterations(100);
BENCHMARK(benchReduceTbbNoInit2V3)->Apply(Args)->UseRealTime()->Iterations(100);
BENCHMARK_MAIN();
