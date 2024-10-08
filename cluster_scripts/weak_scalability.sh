#!/bin/bash

# Weak Scalability Test Script

RUNPATH=/scratch/snx3000/scruzale/ML_APTS/
cd $RUNPATH || exit

# Fixed parameters
OPTIMIZERS=("APTS")
LEARNING_RATES=(1.0)
DATASETS=("MNIST")
BATCH_SIZES=(10000)
MODELS=("feedforward")
SUBDOMAIN_BASE=1
REPLICAS_BASE=1
STAGES_BASE=3
EPOCHS=15
NUM_TRIALS=3
seed_base=12345

# Output log directory
LOG_DIR="./scaling_logs"

# Check if LOG_DIR exists. If not, create it.
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p $LOG_DIR
fi

submit_job() {
    local optimizer=$1
    local lr=$2
    local dataset=$3
    local batch_size=$4
    local model=$5
    local num_subdomains=$6
    local num_replicas_per_subdomain=$7
    local num_stages_per_replica=$8
    local seed=$9
    local trial=${10}

    echo "Running $optimizer on $dataset with batch size $batch_size, model $model, $num_subdomains subdomains, $num_replicas_per_subdomain replicas per subdomain, $num_stages_per_replica stages per replica, with seed $seed for trial $trial" for $EPOCHS epochs

    job_name="${optimizer}_${lr}_${dataset}_${batch_size}_${model}_${num_subdomains}_${num_replicas_per_subdomain}_${num_stages_per_replica}_${EPOCHS}_${seed}_t${trial}"
    error_file="$LOG_DIR/${job_name}.err"
    output_file="$LOG_DIR/${job_name}.out"

    nodes=$((num_subdomains * num_replicas_per_subdomain * num_stages_per_replica))
    sbatch --nodes="$nodes" --job-name="$job_name" --output="$output_file" --error="$error_file" cluster_scripts/weak_scalability.job "$optimizer" "$lr" "$dataset" "$batch_size" "$model" "$num_subdomains" "$num_replicas_per_subdomain" "$num_stages_per_replica" "$seed" "$trial" "$EPOCHS"
}

# Start scaling
for optimizer in "${OPTIMIZERS[@]}"
do
    for lr in "${LEARNING_RATES[@]}"
    do
        for dataset in "${DATASETS[@]}"
        do
            for model in "${MODELS[@]}"
            do
                for scale_factor in {1..4}  # Scale from 1 to 4 times the base configuration
                do
                    num_subdomains=$((SUBDOMAIN_BASE * scale_factor))
                    num_replicas_per_subdomain=$((REPLICAS_BASE * scale_factor))
                    num_stages_per_replica=$((STAGES_BASE * scale_factor))
                    batch_size=$((BATCH_SIZES[0] * scale_factor))

                    for trial in $(seq 0 $(($NUM_TRIALS - 1)))
                    do 
                        # Define a seed here
                        seed=$((seed_base * trial))

                        # Define the file name
                        output_file="$LOG_DIR/${optimizer}_${lr}_${dataset}_${batch_size}_${model}_${num_subdomains}_${num_replicas_per_subdomain}_${num_stages_per_replica}_${EPOCHS}_${seed}_t${trial}.out"

                        # If the file doesn't exist, proceed to submit the job
                        if [ ! -f "$output_file" ]; then
                            submit_job "$optimizer" "$lr" "$dataset" "$batch_size" "$model" "$num_subdomains" "$num_replicas_per_subdomain" "$num_stages_per_replica" "$seed" "$trial"
                        fi
                    done
                done
            done
        done
    done     
done
