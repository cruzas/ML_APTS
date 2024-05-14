#!/bin/bash

echo "Running apts_w.sh...."

cd "../../" || exit
optimizer_name="APTS_W"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --nr_models_array)
            shift
            IFS=',' read -ra nr_models_array <<< "$1"
            shift
            ;;
        --minibatch_sizes)
            shift
            IFS=',' read -ra minibatch_sizes <<< "$1"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Print the arrays and additional options
IFS=','  # Set the field separator to comma
echo "Optimizer name: $optimizer_name"
echo "Number of Models Array: ${nr_models_array[*]}"
echo "Minibatch sizes: ${minibatch_sizes[*]}"
for nr_models in "${nr_models_array[@]}"; do
    for minibatch_size in "${minibatch_sizes[@]}"; do
        # Skip the following code if both momentum and second order equal 1
        # Arguments to be (finally) passed to the python script
        ARGS=(
             --nr_models="$nr_models"
             --minibatch_size="$minibatch_size"
        )

        echo "ARGS to be passed: ${ARGS[*]}"

        job_name="./src/cluster_execution_scripts/${optimizer_name}_${nr_models}_${minibatch_size}.job"
        output_filename="${job_name}.out"
        error_filename="${job_name}.err"
        sbatch --nodes=$nr_models --job-name=$job_name --output=$output_filename --error=$error_filename ./src/cluster_execution_scripts/apts_w.job "${ARGS[@]}" 
        echo "Submitted APTS_W test."
    done
done


