#include <memory>
#include <type_traits>
#include <utility>
#include <tbb/task_scheduler_observer.h>
#include <tbb/atomic.h>
#include <tbb/task_arena.h>
#include <hwloc.h>

namespace numa {
template <typename T, typename A = std::allocator<T>>
class default_init_allocator : public A {
  // Implementation taken from https://stackoverflow.com/a/21028912
  // see also https://hackingcpp.com/cpp/recipe/uninitialized_numeric_array.html
 public:
  using A::A;

  template <typename U>
  struct rebind {
    using other = default_init_allocator<
        U,
        typename std::allocator_traits<A>::template rebind_alloc<U>>;
  };

  template <typename U>
  void construct(U* ptr) noexcept(
      std::is_nothrow_default_constructible<U>::value) {
    ::new (static_cast<void*>(ptr)) U;
  }
  template <typename U, typename... ArgsT>
  void construct(U* ptr, ArgsT&&... args) {
    std::allocator_traits<A>::construct(static_cast<A&>(*this), ptr,
                                        std::forward<ArgsT>(args)...);
  }
};

template <typename T, typename A = std::allocator<T>>
class no_init_allocator : public A {
  // Implementation adapted from https://stackoverflow.com/a/21028912
  // see also https://hackingcpp.com/cpp/recipe/uninitialized_numeric_array.html
 public:
  using A::A;

  template <typename U>
  struct rebind {
    using other = no_init_allocator<
        U,
        typename std::allocator_traits<A>::template rebind_alloc<U>>;
  };

  template <typename U, typename... ArgsT>
  void construct(U* ptr, ArgsT&&... args) { }
};

class PinningObserver : public tbb::task_scheduler_observer {
    hwloc_topology_t topo;
    hwloc_obj_t numa_node;
    int numa_id;
    int numa_nodes;
    tbb::atomic<int> thds_per_node;
    tbb::atomic<int> masters_that_entered;
    tbb::atomic<int> workers_that_entered;
    tbb::atomic<int> threads_pinned;
public:
    PinningObserver(tbb::task_arena& arena, hwloc_topology_t& _topo, int _numa_id,
                    int _thds_per_node) : task_scheduler_observer(arena), topo(_topo),
                    numa_id(_numa_id), thds_per_node(_thds_per_node)
    {
        numa_nodes = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
        numa_node = hwloc_get_obj_by_type(topo, HWLOC_OBJ_NUMANODE, numa_id);
        masters_that_entered = 0;
        workers_that_entered = 0;
        threads_pinned = 0;
        observe(true);
    }                 

    void on_scheduler_entry(bool is_worker)
    {
        if (is_worker) ++workers_that_entered;
        else ++masters_that_entered;
        if(--thds_per_node > 0)
        {
            int err = hwloc_set_cpubind(topo, numa_node->cpuset, HWLOC_CPUBIND_THREAD);
            assert(!err);
            threads_pinned++;
        }
    }
};

}  // namespace numa