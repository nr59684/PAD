cmake_minimum_required( VERSION 3.14 )
project(ex04)

find_package( benchmark REQUIRED )
find_package( Threads REQUIRED )
#find_package( range-v3 REQUIRED )
find_package( TBB REQUIRED )
find_package( OpenMP REQUIRED COMPONENTS CXX )

function( configure_exercise_target targetname )
	target_compile_features( ${targetname} PRIVATE cxx_std_17 )
	target_compile_options( ${targetname} PRIVATE -march=core-avx2 -mtune=core-avx2 ) 
	target_include_directories( ${targetname} PRIVATE $ENV{UMESIMD_ROOT} include)
	target_link_libraries( ${targetname} PRIVATE benchmark::benchmark range-v3 TBB::tbb Threads::Threads OpenMP::OpenMP_CXX )
endfunction()

add_executable( reduction-benchmark04 reduction.cpp )
configure_exercise_target( reduction-benchmark04 )

add_executable( transform-benchmark04 transform.cpp )
configure_exercise_target( transform-benchmark04 )
