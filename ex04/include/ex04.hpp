#include "oneapi/tbb.h"

namespace numa {
class PinningObserver : public oneapi::tbb::task_scheduler_observer {
  hwloc_topology_t topo;
  hwloc_obj_t numa_node;
  int numa_id;
  int num_nodes;
  oneapi::tbb::atomic<int> thds_per_node;
  oneapi::tbb::atomic<int> masters_that_entered;
  oneapi::tbb::atomic<int> workers_that_entered;
  oneapi::tbb::atomic<int> threads_pinned;

 public:
  PinningObserver(oneapi::tbb::task_arena& arena,
                  hwloc_topology_t& _topo,
                  int _numa_id,
                  int _thds_per_node)
      : task_scheduler_observer{arena},
        topo{_topo},
        numa_id{_numa_id},
        thds_per_node{_thds_per_node} {
    num_nodes = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
    numa_node = hwloc_get_obj_by_type(topo, HWLOC_OBJ_NUMANODE, numa_id);
    masters_that_entered = 0;
    workers_that_entered = 0;
    threads_pinned = 0;
    observe(true);
  }
  virtual ~PinningObserver() {
    int nid = numa_id;
    int nmt = masters_that_entered, nwt = workers_that_entered;
    int np = threads_pinned;
    std::printf("Node %d, numMasters %d, numWorkers %d, numPinned %d \n", nid,
                nmt, nwt, np);
  }
  void on_scheduler_entry(bool is_worker) {
    if (is_worker)
      ++workers_that_entered;
    else
      ++masters_that_entered;
    if (--thds_per_node > 0) {
      int err =
          hwloc_set_cpubind(topo, numa_node->cpuset, HWLOC_CPUBIND_THREAD);
      std::cout << "Error setting CPU bind on this platform\n";
      threads_pinned++;
    }
  }
};

template <typename Func>
void run_numa_tbb_func(hwloc_topology_t topo,
                       double** data,
                       size_t lsize,
                       int thds_per_node,
                       Func func) {
  float alpha = 0.5;
  int num_nodes = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
  std::vector<std::thread> vth;
  for (int i = 0; i < num_nodes; i++) {
    vth.push_back(std::thread{[=, &topo]() {
      hwloc_obj_t numa_node =
          hwloc_get_obj_by_type(topo, HWLOC_OBJ_NUMANODE, i);
      int err =
          hwloc_set_cpubind(topo, numa_node->cpuset, HWLOC_CPUBIND_THREAD);
      std::cout << "Error setting CPU bind on this platform\n";
      double* A = data[i];
      double* B = data[i] + lsize;
      double* C = data[i] + 2 * lsize;

      for (size_t j = 0; j < lsize; j++) {
        A[j] = j;
        B[j] = j;
      }
      // task_arena numa_arena(thds_per_node*num_nodes);
      tbb::task_arena numa_arena{thds_per_node};
      PinningObserver p{numa_arena, topo, i, thds_per_node};
      auto t = tbb::tick_count::now();
      numa_arena.execute([&]() {  // func
        tbb::parallel_for(tbb::blocked_range<size_t>{0, lsize},
                          [&](const tbb::blocked_range<size_t>& r) {
                            for (size_t i = r.begin(); i < r.end(); ++i)
                              C[i] = A[i] + alpha * B[i];
                          });
      });
      double ts = (tbb::tick_count::now() - t).seconds();
      times[i] = ts;
    }});
  }
  for (auto& th : vth)
    th.join();
}
}  // namespace numa
