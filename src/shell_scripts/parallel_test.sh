#!/bin/bash

# Define the arrays for each parameter
optimizers=("SGD")
batch_sizes=(100)
learning_rates=(0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.0)
trials=1
epochs=100
nodes=2
datasets=("cifar10")

# Iterate over each combination of parameters
for optimizer in "${optimizers[@]}"
do
    for batch_size in "${batch_sizes[@]}"
    do
        for learning_rate in "${learning_rates[@]}"
        do
            for dataset in "${datasets[@]}"
            do
                # Make job name depend on the parameters
                job_name="test_${optimizer}_${batch_size}_${learning_rate}_${dataset}"
                error_file="${job_name}.err"
                output_file="${job_name}.out"
                
                sbatch --nodes=$nodes --job-name="$job_name" --output="$output_file" --error="$error_file" parallel_test.job "$optimizer" "$batch_size" "$learning_rate" "$trials" "$epochs" "$dataset"
            done
        done
    done
done

echo "All tests submitted to cluster."
