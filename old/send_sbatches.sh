#!/bin/bash

# Array of different numbers of nodes
nnodes_array=(2 4 8)

# Iterate over the node numbers and submit the job with the specified number of nodes
for node in "${nnodes_array[@]}"; do
    sbatch --nodes=$node --job-name="${node}_deepspeed" --output="${node}_deepspeed.out" --error="${node}_deepspeed.err" launch.slurm
done
