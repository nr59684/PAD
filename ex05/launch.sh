#!/usr/bin/env bash
#SBATCH -p rome
#SBATCH -w ziti-rome1
#SBATCH --exclusive
#SBATCH -o test.txt

./../build/ex05/reduction-benchmark05v3 

./../build/ex05/transform-benchmark05v3
