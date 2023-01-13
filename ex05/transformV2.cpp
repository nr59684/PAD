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
  state.counters["Bytes"] = 3 * state.range(0) * sizeof(ValueType);
  state.SetLabel(name);
}

void initThreads(std::vector<tbb::task_arena, numa::no_init_allocator<tbb::task_arena>>& numa_arenas, hwloc_topology_t& topo, const int num_nodes){
    omp_set_dynamic(0);
    omp_set_num_threads(num_nodes);

    #pragma omp parallel for num_threads(4)
    for(int i = 0; i < num_nodes; i++){
        hwloc_obj_t numa_node = hwloc_get_obj_by_type(topo, HWLOC_OBJ_NUMANODE, i);
        hwloc_set_cpubind(topo, numa_node->cpuset, HWLOC_CPUBIND_THREAD);

        new (&numa_arenas[i]) tbb::task_arena{thrds_per_node};
        numa::PinningObserver p{numa_arenas[i], topo, i, thrds_per_node};
    }
}

static void benchTransformTbbNoInitV2(benchmark::State& state){
    hwloc_topology_t topo;
    hwloc_topology_init(&topo);
    hwloc_topology_load(topo);
    int num_nodes = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
    int size = state.range(0) / num_nodes;
    
    std::vector<tbb::task_arena, numa::no_init_allocator<tbb::task_arena>> numa_arenas(num_nodes);
    initThreads(numa_arenas, topo, num_nodes);

    ContainerTypeNoInit X(state.range(0));
    ContainerTypeNoInit Y(state.range(0));
    Partitioner part;
    const ValueType alpha = 2;

    #pragma omp parallel for
    for(int i = 0; i < num_nodes; i++){
        std::uninitialized_fill(std::execution::unseq, X.begin() + i * size, X.begin() + (i + 1) * size, 1);
        std::uninitialized_fill(std::execution::unseq, Y.begin() + i * size, Y.begin() + (i + 1) * size, 2);
    }

    for (auto _ : state){
        #pragma omp parallel for
        for (int i = 0; i < num_nodes; i++){
            numa_arenas[i].execute([&](){
                tbb::parallel_for(tbb::blocked_range<size_t>(0, size), [&](const tbb::blocked_range<size_t> r){
                    for(size_t j = r.begin(); j < r.end(); j++){
                        Y[i * size + j] = alpha * X[i * size + j] + Y[i * size + j];
                    }
                }, part);
            });
        }

        benchmark::DoNotOptimize(Y.data());
        benchmark::ClobberMemory();
    }

    hwloc_topology_destroy(topo);

    setCustomCounter(state, "TransformTbbNoInitV2");
}

static void benchTransformTbbNoInit2V2(benchmark::State& state){
    hwloc_topology_t topo;
    hwloc_topology_init(&topo);
    hwloc_topology_load(topo);
    int num_nodes = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
    int size = state.range(0) / num_nodes;

    std::vector<tbb::task_arena, numa::no_init_allocator<tbb::task_arena>> numa_arenas(num_nodes);
    initThreads(numa_arenas, topo, num_nodes);

    ContainerTypeNoInit X(state.range(0));
    ContainerTypeNoInit Y(state.range(0));
    Partitioner part;
    const ValueType alpha = 2;

    #pragma omp parallel for
    for (int i = 0; i < num_nodes; i++){
        numa_arenas[i].execute([&](){
            tbb::parallel_for(tbb::blocked_range<size_t>(0, size), [&](const tbb::blocked_range<size_t> r){
                for(size_t j = r.begin(); j < r.end(); j++){
                    new(&X[i * size + j]) ContainerTypeNoInit::value_type{1};
                    new(&X[i * size + j]) ContainerTypeNoInit::value_type{2};
                }
            }, part);                
        });
    }

    for (auto _ : state){
        #pragma omp parallel for
        for (int i = 0; i < num_nodes; i++){
            numa_arenas[i].execute([&](){
                tbb::parallel_for(tbb::blocked_range<size_t>(0, size), [&](const tbb::blocked_range<size_t> r){
                    for(size_t j = r.begin(); j < r.end(); j++){
                        Y[i * size + j] = alpha * X[i * size + j] + Y[i * size + j];
                    }
                }, part);
            }
        );}


        benchmark::DoNotOptimize(Y.data());
        benchmark::ClobberMemory();      
    }

    hwloc_topology_destroy(topo);

    setCustomCounter(state, "TransformTbbNoInit2V2");
}

BENCHMARK(benchTransformTbbNoInitV2)->Apply(Args)->UseRealTime()->Iterations(100);
BENCHMARK(benchTransformTbbNoInit2V2)->Apply(Args)->UseRealTime()->Iterations(100);
BENCHMARK_MAIN();