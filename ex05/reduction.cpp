#include <vector>
#include <algorithm>
#include <thread>
#include <future>
#include <hwloc.h>
#include <iostream>
#include <execution>
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

static void benchReduceTbbNoInit(benchmark::State& state){
    hwloc_topology_t topo;
    hwloc_topology_init(&topo);
    hwloc_topology_load(topo);
    int num_nodes = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
    int size = state.range(0) / num_nodes;
    ContainerTypeNoInit X(state.range(0));
    std::uninitialized_fill(std::execution::par_unseq, X.begin(), X.end(), 1);
    Partitioner part;

    ValueType total_sum;
    for (auto _ : state){
        total_sum = 0;
        std::vector<std::thread> vth;
        std::promise<ValueType> partSumPromise[num_nodes];
        std::future<ValueType> partSumFuture[num_nodes];
        for (int i = 0; i < num_nodes; i++){
            partSumFuture[i] = partSumPromise[i].get_future();
            vth.push_back(std::thread{
                [&, i](){
                    hwloc_obj_t numa_node = hwloc_get_obj_by_type(topo, HWLOC_OBJ_NUMANODE, i);
                    hwloc_set_cpubind(topo, numa_node->cpuset, HWLOC_CPUBIND_THREAD);

                    tbb::task_arena numa_arena{thrds_per_node};
                    numa::PinningObserver p{numa_arena, topo, i, thrds_per_node};
                    numa_arena.execute([&](){
                        auto sum = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, size), 0,
                                                        [&](const tbb::blocked_range<size_t> r, ValueType ret){
                                                            for (size_t j = r.begin(); j < r.end(); j++){
                                                                ret += X[i * size + j];
                                                            }
                                                            return ret;
                                                        }, std::plus<ValueType>(), part);
                        partSumPromise[i].set_value(sum);
                    });
                }
            });
        }
        for (int i = 0; i < num_nodes; i++){
            vth[i].join();
            total_sum += partSumFuture[i].get();
        }

        benchmark::DoNotOptimize(&total_sum);
        benchmark::ClobberMemory();
    }
    
    hwloc_topology_destroy(topo);
    setCustomCounter(state, "ReduceTbbNoInit");
}

static void benchReduceTbbNoInit2(benchmark::State& state){
    hwloc_topology_t topo;
    hwloc_topology_init(&topo);
    hwloc_topology_load(topo);
    int num_nodes = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
    int size = state.range(0) / num_nodes;
    ContainerTypeNoInit X(state.range(0));
    Partitioner part;

    std::vector<std::thread> initThreads;
    for (int i = 0; i < num_nodes; i++){
        initThreads.push_back(std::thread{
            [&, i](){
                hwloc_obj_t numa_node = hwloc_get_obj_by_type(topo, HWLOC_OBJ_NUMANODE, i);
                hwloc_set_cpubind(topo, numa_node->cpuset, HWLOC_CPUBIND_THREAD);

                tbb::task_arena numa_arena{thrds_per_node};
                numa::PinningObserver p{numa_arena, topo, i, thrds_per_node};
                numa_arena.execute([&](){
                    tbb::parallel_for(tbb::blocked_range<size_t>(0, size), [&](const tbb::blocked_range<size_t> r){
                        for(size_t j = r.begin(); j < r.end(); j++){
                            new(&X[i * size + j]) ContainerTypeNoInit::value_type{1};
                        }
                    }, part);                 
                });
            }
        });
    }

    for(auto& th : initThreads){
        th.join();
    }

    ValueType total_sum;
    for (auto _ : state){
        total_sum = 0;
        std::vector<std::thread> vth;
        std::promise<ValueType> partSumPromise[num_nodes];
        std::future<ValueType> partSumFuture[num_nodes];
        for (int i = 0; i < num_nodes; i++){
            partSumFuture[i] = partSumPromise[i].get_future();
            vth.push_back(std::thread{
                [&, i](){
                    hwloc_obj_t numa_node = hwloc_get_obj_by_type(topo, HWLOC_OBJ_NUMANODE, i);
                    hwloc_set_cpubind(topo, numa_node->cpuset, HWLOC_CPUBIND_THREAD);

                    tbb::task_arena numa_arena{thrds_per_node};
                    numa::PinningObserver p{numa_arena, topo, i, thrds_per_node};
                    numa_arena.execute([&](){
                        auto sum = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, size), 0,
                                                        [&](const tbb::blocked_range<size_t> r, ValueType ret){
                                                            for (size_t j = r.begin(); j < r.end(); j++){
                                                                ret += X[i * size + j];
                                                            }
                                                            return ret;
                                                        }, std::plus<ValueType>(), part);
                        partSumPromise[i].set_value(sum);
                    });
                }
            });
        }
        for (int i = 0; i < num_nodes; i++){
            vth[i].join();
            total_sum += partSumFuture[i].get();
        }

        benchmark::DoNotOptimize(&total_sum);
        benchmark::ClobberMemory();
    }

    hwloc_topology_destroy(topo);
    setCustomCounter(state, "ReduceTbbNoInit2");
}

BENCHMARK(benchReduceTbbNoInit)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceTbbNoInit2)->Apply(Args)->UseRealTime();
BENCHMARK_MAIN();