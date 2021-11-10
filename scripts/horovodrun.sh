#!/bin/bash
export OMP_NUM_THREADS=1
workspace=$(pwd)
num_proc=$1
single_script=$2

mpirun -np ${num_proc} \
    --allow-run-as-root \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH \
    -x PYTHONPATH \
    -x OMP_NUM_THREADS=1 \
    -mca pml ob1 \
    -mca btl ^openib \
    ${single_script}
