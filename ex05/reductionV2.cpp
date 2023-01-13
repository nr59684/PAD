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

#include "allocator_adaptor.hpp"

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

void initThreads(std::vector<tbb::task_arena, numa::no_init_allocator<tbb::task_arena>>& numa_arenas, hwloc_topology_t& topo, const int num_nodes){
    omp_set_dynamic(0);
    omp_set_num_threads(num_nodes);

    #pragma omp parallel for 
    for(int i = 0; i < num_nodes; i++){
        hwloc_obj_t numa_node = hwloc_get_obj_by_type(topo, HWLOC_OBJ_NUMANODE, i);
        hwloc_set_cpubind(topo, numa_node->cpuset, HWLOC_CPUBIND_THREAD);

        new (&numa_arenas[i]) tbb::task_arena{thrds_per_node};
        numa::PinningObserver p{numa_arenas[i], topo, i, thrds_per_node};
    }
}

static void benchReduceTbbNoInitV2(benchmark::State& state){
    hwloc_topology_t topo;
    hwloc_topology_init(&topo);
    hwloc_topology_load(topo);
    int num_nodes = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
    int size = state.range(0) / num_nodes;
    std::vector<tbb::task_arena, numa::no_init_allocator<tbb::task_arena>> numa_arenas(num_nodes);

    initThreads(numa_arenas, topo, num_nodes);
    ContainerTypeNoInit X(state.range(0));

    #pragma omp parallel for
    for(int i = 0; i < num_nodes; i++){
        std::uninitialized_fill(std::execution::unseq, X.begin() + i * size, X.begin() + (i + 1) * size , 1);
    }
    Partitioner part;

    ValueType total_sum;
    for (auto _ : state){
        total_sum = 0;
        std::vector<ValueType> part_sums(num_nodes, 0);  
        #pragma omp parallel for
        for(int i = 0; i < num_nodes; i++){
            part_sums[i] = numa_arenas[i].execute([&, i]() -> ValueType {
                ValueType sum = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, size), 0,
                                                [&](const tbb::blocked_range<size_t> r, ValueType ret) -> ValueType {
						    for (size_t j = r.begin(); j < r.end(); j++){
                                                        ret += X[i * size + j];
                                                    }
                                                    return ret;
                                                }, std::plus<ValueType>(), part);
                return sum;
            });

        }
        for (auto sums : part_sums) total_sum += sums;   
        benchmark::DoNotOptimize(&total_sum);
        benchmark::ClobberMemory();
    }
    if (total_sum != static_cast<ValueType>(state.range(0))) std::cout << "wrong result" << std::endl;   
    hwloc_topology_destroy(topo);
    setCustomCounter(state, "ReduceTbbNoInitV2");
}

static void benchReduceTbbNoInit2V2(benchmark::State& state){
    hwloc_topology_t topo;
    hwloc_topology_init(&topo);
    hwloc_topology_load(topo);
    int num_nodes = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
    int size = state.range(0) / num_nodes;
    std::vector<tbb::task_arena, numa::no_init_allocator<tbb::task_arena>> numa_arenas(num_nodes);

    initThreads(numa_arenas, topo, num_nodes);
    ContainerTypeNoInit X(state.range(0));
    Partitioner part;

    #pragma omp parallel for
    for (int i = 0; i < num_nodes; i++){
        numa_arenas[i].execute([&](){
            tbb::parallel_for(tbb::blocked_range<size_t>(0, size), [&](const tbb::blocked_range<size_t> r){
                for(size_t j = r.begin(); j < r.end(); j++){
                    new(&X[i * size + j]) ContainerTypeNoInit::value_type{1};
                }
            }, part);                 
        });
    }

    ValueType total_sum;
    for (auto _ : state){
        total_sum = 0;
        std::vector<ValueType> part_sums(num_nodes, 0);  
        #pragma omp parallel for
        for(int i = 0; i < num_nodes; i++){
            part_sums[i] = numa_arenas[i].execute([&, i]() -> ValueType {
                ValueType sum = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, size), 0,
                                                [&](const tbb::blocked_range<size_t> r, ValueType ret) -> ValueType {
                                                    for (size_t j = r.begin(); j < r.end(); j++){
                                                        ret += X[i * size + j];
                                                    }
                                                    return ret;
                                                }, std::plus<ValueType>(), part);
                return sum;
            });
        }
        for(auto sums : part_sums) total_sum += sums;     

        benchmark::DoNotOptimize(&total_sum);
        benchmark::ClobberMemory();
    }
    if (total_sum != static_cast<ValueType>(state.range(0))) std::cout << "wrong result" << std::endl;
    hwloc_topology_destroy(topo);
    setCustomCounter(state, "ReduceTbbNoInit2V2");
}

BENCHMARK(benchReduceTbbNoInitV2)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceTbbNoInit2V2)->Apply(Args)->UseRealTime();
BENCHMARK_MAIN();
