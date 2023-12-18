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

for nr_models in "${nr_models_array[@]}"; do
    # Skip the following code if both momentum and second order equal 1
    if [ "$momentum" -eq 1 ] && [ "$second_order" -eq 1 ]; then
        continue
    else
        # Arguments to be (finally) passed to the python script
        ARGS=(
            --optimizer_name="$optimizer_name"
            --nr_models="$nr_models"
        )
        if [ "$on_cluster" -eq 1 ]; then # execute on cluster
            job_name="./src/cluster_execution_scripts/${optimizer_name}_${nr_models}.job"
            output_filename="${job_name}.out"
            error_filename="${job_name}.err"
            # TODO: change --nodes=1 to --nodes=$nr_models in case we ever execute APTS_W in parallel!
            sbatch --nodes=$nr_models --job-name=$job_name --output=$output_filename --error=$error_filename ./src/cluster_execution_scripts/apts_w.job "${ARGS[@]}" 
        fi
        echo "Submitted APTS_W test with counter $counter"
        ((counter++))
    fi
done


echo $counter