#!/bin/bash -l

echo "Running main.sh...."

# Check which computer/cluster to execute the tests on
if [[ $PWD == *"scratch"* ]]; then
    on_cluster=1
    on_kens_computer=0
    db_path="./results/main_daint_test.db"
elif [[ $PWD == *"cruzalegria"* ]]; then
    on_cluster=0
    on_kens_computer=0
    db_path="./results/main_daint_test_copy.db"
else
    on_cluster=0
    on_kens_computer=1 
    db_path="./results/TEMP.db"
fi

# Check that we are only executing on one device. If both are 0, then we 
# are executing on Sam's computer. 
if [ "$on_cluster" -eq 1 ] && [ "$on_kens_computer" -eq 1 ]; then
    echo "Error: on_cluster and on_kens_computer cannot both be 1."
    exit 1
fi

### Parameters common to every optimizer test ### 
epochs=10 # number of epochs 
trials=1 # number of trials 
# !!!!!!!!!! DANGER IMPORTANT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!! separate array elements by spaces, not commas !!!!!!!!!
dataset_names=("MNIST") # datasets
minibatch_sizes=(60000)   # minibatch sizes
overlap_ratios=(0.0)   # minibatch overlap ratios
network_numbers=(3)    # network numbers (each number is a different model)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

### Turn on/off tests for each optimizer ###
turn_on_sqlhandler=0
do_APTS_W_tests=1
counter=0 # counter for having separate dataframes for each test, since there are currently some issues with having multiple tests in the same dataframe on Piz Daint
############################################

# Check that SQLhandler is on and that no tests are on
if [ "$turn_on_sqlhandler" -eq 1 ] && [ "$do_APTS_W_tests" -eq 1 ]; then
    echo "Error: SQLhandler and APTS_W tests cannot both be on."
    exit 1
else 
        if [ "$turn_on_sqlhandler" -eq 1 ]; then
                ./request_handler.sh \
                --db_path $db_path \
                --on_cluster $on_cluster \
                --on_kens_computer $on_kens_computer 
                exit 0
        fi
fi

# APTS_W tests
radius=0.01 # learning rate for APTS global optimizer
second_order_array=(0)
momentum_array=(0)
nr_models_array=(2)
if [ "$do_APTS_W_tests" -eq 1 ]; then
    echo "Current directory is $PWD"
    echo "Counter before starting APTS_W tests: $counter"
    IFS=',' 
    # counter=$(
    echo "Trying to calling apts_w.sh..."
        ./apts_w.sh \
    --on_cluster $on_cluster \
    --on_kens_computer $on_kens_computer \
    --epochs $epochs \
    --trials $trials \
    --db_path $db_path \
    --counter $counter \
    --network_numbers "${network_numbers[*]}" \
    --dataset_names "${dataset_names[*]}" \
    --minibatch_sizes "${minibatch_sizes[*]}" \
    --overlap_ratios "${overlap_ratios[*]}" \
    --radius "${radius[*]}"  \
    --second_order_array "${second_order_array[*]}" \
    --momentum_array "${momentum_array[*]}" \
    --nr_models_array "${nr_models_array[*]}" 
    # | tail -n 1)
    sleep 1
fi

echo "Counter after all tests: $counter"