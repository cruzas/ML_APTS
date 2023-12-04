#!/bin/bash

echo "Running apts_w.sh...."

cd "../../" || exit
optimizer_name="APTS_W"
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --on_cluster)
            shift
            on_cluster="$1"
            shift
            ;;
        --on_kens_computer)
            shift
            on_kens_computer="$1"
            shift
            ;;
        --epochs)
            shift
            epochs="$1"
            shift
            ;;
        --trials)
            shift
            trials="$1"
            shift
            ;;
        --db_path)
            shift
            db_path="$1"
            shift
            ;;
        --counter)
            shift
            counter="$1"
            shift
            ;;
        --network_numbers)
            shift
            IFS=',' read -ra network_numbers <<< "$1"
            shift
            ;;
        --dataset_names)
            shift
            IFS=',' read -ra dataset_names <<< "$1"
            shift
            ;;
        --minibatch_sizes)
            shift
            IFS=',' read -ra minibatch_sizes <<< "$1"
            shift
            ;;
        --overlap_ratios)
            shift
            IFS=',' read -ra overlap_ratios <<< "$1"
            shift
            ;;
        --radius)
            shift
            radius="$1"
            shift
            ;;
        --second_order_array)
            shift
            IFS=',' read -ra second_order_array <<< "$1"
            shift
            ;;
        --momentum_array)
            shift
            IFS=',' read -ra momentum_array <<< "$1"
            shift
            ;;
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
echo "On cluster: $on_cluster"
echo "On Ken's computer: $on_kens_computer"
echo "Number of Epochs: $epochs"
echo "Number of Trials: $trials"
echo "DB path: $db_path"
echo "Counter: $counter"
echo "Network Numbers: ${network_numbers[*]}"
echo "Dataset Names: ${dataset_names[*]}"
echo "Minibatch Sizes: ${minibatch_sizes[*]}"
echo "Overlap Ratios: ${overlap_ratios[*]}"
echo "Radius: ${radius}"
echo "Momentum Array: ${momentum_array[*]}"
echo "Second Order Array: ${second_order_array[*]}"
echo "Number of Models Array: ${nr_models_array[*]}"

for net_nr in "${network_numbers[@]}"; do
    for dataset in "${dataset_names[@]}"; do
        for minibatch_size in "${minibatch_sizes[@]}"; do
            for overlap_ratio in "${overlap_ratios[@]}"; do 
                for momentum in "${momentum_array[@]}"; do # this is 0 or 1
                    for second_order in "${second_order_array[@]}"; do # this is 0 or 1
                        for nr_models in "${nr_models_array[@]}"; do
                            # Skip the following code if both momentum and second order equal 1
                            if [ "$momentum" -eq 1 ] && [ "$second_order" -eq 1 ]; then
                                continue
                            else
                                # Arguments to be (finally) passed to the python script
                                ARGS=(
                                    --optimizer_name="$optimizer_name"
                                    --epochs="$epochs"
                                    --db_path="$db_path"
                                    --trials="$trials"
                                    --net_nr="$net_nr"
                                    --dataset="$dataset"
                                    --minibatch_size="$minibatch_size"
                                    --overlap_ratio="$overlap_ratio"
                                    --radius="$radius"
                                    --second_order="$second_order"
                                    --momentum="$momentum"
                                    --nr_models="$nr_models"
                                    --counter="$counter"
                                )
                                if [ "$on_cluster" -eq 1 ]; then # execute on cluster
                                    job_name="./src/cluster_execution_scripts/${optimizer_name}_${dataset}_N${net_nr}_M${minibatch_size}_O${overlap_ratio}_R${radius}_S${second_order}_M${momentum}"
                                    output_filename="${job_name}.out"
                                    error_filename="${job_name}.err"
                                    # TODO: change --nodes=1 to --nodes=$nr_models in case we ever execute APTS_W in parallel!
                                    sbatch --nodes=2 --job-name=$job_name --output=$output_filename --error=$error_filename ./src/cluster_execution_scripts/apts_w.job "${ARGS[@]}" 
                                elif [ "$on_kens_computer" -eq 1 ]; then # execute on Ken's computer
                                    echo "Executing Python script on Ken's computer $(date)"
                                    python ./src/main.py "${ARGS[@]}" 
                                    echo "All done $(date)"
                                else # execute on Sam's computer
                                    echo "Activating pytorch_mac `date`"
                                    conda init zsh
                                    source ~/.zshrc
                                    conda activate pytorch_mac

                                    echo "Calling python script $(date)"
                                    /Users/cruzalegriasamueladolfo/opt/anaconda3/envs/pytorch_mac/bin/python -u ./src/main.py "${ARGS[@]}" 
                                    echo "All done $(date)"
                                fi
                                echo "Submitted APTS_W test with counter $counter"
                                ((counter++))
                            fi
                        done
                    done
                done
            done
        done
    done
done

echo $counter