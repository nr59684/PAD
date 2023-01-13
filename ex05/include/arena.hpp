#include <hwloc.h>
#include <tuple>
#include <vector>
#include <unistd.h>
#include <cstdlib>
#include <oneapi/tbb/task_arena.h>
#include "allocator_adaptor.hpp"

namespace numa{
class ArenaMgtTBB{
    public:
        ArenaMgtTBB(int thrds) : threads_per_node(thrds) {
            hwloc_topology_init(&topology);
            hwloc_topology_load(topology);
            size = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE);

            arenas.reserve(size);
            init_threads();
        }
        ~ArenaMgtTBB(){
            hwloc_topology_destroy(topology);
            
            for (auto& a : arenas) std::free(a);
        }

        std::tuple<int, int> index_range(const int mth, const size_t vec_size){
            int part = vec_size / size;
            int rest = vec_size % size;
            
            if (mth == 0){
                return std::make_tuple(0, part + rest);
            }
            else{
                return std::make_tuple(mth * part + rest, (mth + 1) * part + rest);
            }
        }

        tbb::task_arena* operator[](int idx){
            return arenas[idx];
        }

        int get_size(){
            return size;
        }
        

    private:
        void init_threads(){
            omp_set_dynamic(0);
            omp_set_num_threads(size);

            #pragma omp parallel for 
            for (int i = 0; i < size; i++){
                hwloc_obj_t numa_node = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, i);
                hwloc_set_cpubind(topology, numa_node->cpuset, HWLOC_CPUBIND_THREAD);

                tbb::task_arena* arena_p = static_cast<tbb::task_arena*>(std::aligned_alloc(sysconf(_SC_PAGE_SIZE), sizeof(tbb::task_arena)));
                new (arena_p) tbb::task_arena{threads_per_node};   
                
                arenas[i] = arena_p;
                numa::PinningObserver p{*arenas[i], topology, i, threads_per_node};
            }             
        }

        int size;
        std::vector<tbb::task_arena*> arenas;
        const int threads_per_node;
        hwloc_topology_t topology;
};

}
