1.1 transform / reduce: 1.1.1 iterator-schleife & range-based for  1.1.2 stl
1.2 simd transform / reduce 1.2.1 umeSIMD 1.2.2 pragma omp simd single threaded
    add simd loop like project tbb implementation
2 Theory

3.0 show execution policies
3.1 transform / reduce parallel + simd: 3.1.1 OpenMP 3.1.2 TBB
3.2 scheduling / partitioner / grain size

4.1 NUMA: Allocation 4.1.1 unitialized_fill - OpenMP places 4.1.2 TBB hwloc container based
    

5.1 Mandelbrot + NUMA OpenMP + TBB


Build & execute:
1. create .epoch file in home directory and write 2020 into it. Save and create a new shell session.

2. source load-env.sh

3. now everything is set up to build and execute. An example script for execution can be found in /ex05
