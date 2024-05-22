#!/bin/bash

# Define the arrays for each parameter
optimizers=("SGD")
batch_sizes=(200)
trials=5
epochs=10
nodes=2
datasets=("mnist")

# Iterate over each combination of parameters
for optimizer in "${optimizers[@]}"
do
    for batch_size in "${batch_sizes[@]}"
    do   
        for dataset in "${datasets[@]}"
        do
            # If optimizer is SGD, set learning rate to 0.1. Else, 0.001
            if [ "$optimizer" == "SGD" ]
            then
                # If dataset is CIFAR-10, set learning rate to 0.01
                if [ "$dataset" == "cifar10" ]
                then
                    learning_rate=0.01
                else
                    learning_rate=0.01
                fi
            else
                if [ "$dataset" == "mnist" ]
                then
                    learning_rate=0.001
                else
                    learning_rate=0.001
                fi
            fi

            # Make job name depend on the parameters
            job_name="test_${optimizer}_${batch_size}_${learning_rate}_${dataset}"
            error_file="${job_name}.err"
            output_file="${job_name}.out"
            
            sbatch --nodes="$nodes" --job-name="$job_name" --output="$output_file" --error="$error_file" parallel_test.job "$optimizer" "$batch_size" "$learning_rate" "$trials" "$epochs" "$dataset"
        done
    done
done

echo "All tests submitted to cluster."
