#!/bin/bash -l

echo "Running main.sh...."

nr_models_array=(1 2)
if [ "$do_APTS_W_tests" -eq 1 ]; then
    echo "Current directory is $PWD"
    echo "Counter before starting APTS_W tests: $counter"
    IFS=',' 
    echo "Trying to calling apts_w.sh..."
        ./apts_w.sh \
    --nr_models_array "${nr_models_array[*]}" 
    sleep 1
fi