#!/bin/bash

# Define the arrays for each parameter
# optimizers=("SGD" "Adam" "APTS")
optimizers=("SGD")
batch_sizes=(200)
epochs=1
nodes_SGD_Adam=2
nodes_APTS=(2 4 8)
# trial_numbers=(1 2 3)
trial_numbers=(2)
datasets=("cifar10")

submit_job() {
    local optimizer=$1
    local batch_size=$2
    local dataset=$3
    local trial_number=$4
    local learning_rate=$5
    local nodes=$6

    job_name="${optimizer}_nl6_${dataset}_${batch_size}_${learning_rate}_${epochs}_${nodes}_t${trial_number}"
    error_file="${job_name}.err"
    output_file="${job_name}.out"

    sbatch --nodes="$nodes" --job-name="$job_name" --output="$output_file" --error="$error_file" parallel_test.job "$optimizer" "$batch_size" "$learning_rate" "$trial_number" "$epochs" "$dataset"
}

# Iterate over each combination of parameters
for optimizer in "${optimizers[@]}"
do
    for batch_size in "${batch_sizes[@]}"
    do   
        for dataset in "${datasets[@]}"
        do
            for trial_number in "${trial_numbers[@]}"
            do
                if [ "$optimizer" == "SGD" ] || [ "$optimizer" == "Adam" ]
                then
                    nodes="$nodes_SGD_Adam"
                    if [ "$optimizer" == "SGD" ] && [ "$dataset" == "cifar10" ]
                    then
                        learning_rate=0.01
                    else
                        learning_rate=0.001
                    fi
                    submit_job "$optimizer" "$batch_size" "$dataset" "$trial_number" "$learning_rate" "$nodes"
                else
                    for nodes in "${nodes_APTS[@]}"
                    do
                        learning_rate=0.001
                        submit_job "$optimizer" "$batch_size" "$dataset" "$trial_number" "$learning_rate" "$nodes"
                    done
                fi
            done
        done
    done
done
