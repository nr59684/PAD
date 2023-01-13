cmake_minimum_required( VERSION 3.14 )
project(ex05 CXX)

find_package( Threads REQUIRED )
find_package( TBB REQUIRED )
find_package( benchmark REQUIRED )
find_package( OpenMP REQUIRED COMPONENTS CXX)

find_library(HWLOC_LIB hwloc)
find_path(HWLOC_INC hwloc.h)

function( configure_exercise_target targetname )
	target_compile_features( ${targetname} PRIVATE cxx_std_20 )
	target_compile_options( ${targetname} PRIVATE -march=core-avx2 -mtune=core-avx2 ) 
	target_include_directories( ${targetname} PRIVATE include ${HWLOC_INC} )
	target_link_libraries( ${targetname} PRIVATE benchmark::benchmark TBB::tbb Threads::Threads OpenMP::OpenMP_CXX ${HWLOC_LIB} )
endfunction()

add_executable( reduction-benchmark05v3 reductionV3.cpp )
configure_exercise_target( reduction-benchmark05v3 )

add_executable( reduction-benchmark05v2 reductionV2.cpp )
configure_exercise_target( reduction-benchmark05v2 )

add_executable( reduction-benchmark05 reduction.cpp )
configure_exercise_target( reduction-benchmark05 )

add_executable( transform-benchmark05v3 transformV3.cpp )
configure_exercise_target( transform-benchmark05v3 )

add_executable( transform-benchmark05v2 transformV2.cpp )
configure_exercise_target( transform-benchmark05v2 )

add_executable( transform-benchmark05 transform.cpp )
configure_exercise_target( transform-benchmark05 )

