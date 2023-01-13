#!/bin/bash

echo "Setting up environment variables"
spack load gcc@11
spack load intel-oneapi-compilers
spack load intel-oneapi-tbb
spack load benchmark
spack load cmake@3.22.0
export CC=icx
export CXX=icpx
export UMESIMD_ROOT=/opt/asc/pub/spack/opt/spack/linux-centos7-x86_64_v2/clang-14.0.1/umesimd-0.8.1-1-h6nzw3abgywvrkpo2cmt7yq7lfjrhxxk/include

