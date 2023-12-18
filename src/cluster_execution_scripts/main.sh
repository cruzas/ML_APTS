#!/bin/bash -l

echo "Running main.sh...."

nr_models_array=(1 2)
echo "Current directory is $PWD"
echo "Counter before starting APTS_W tests: $counter"
IFS=',' 
echo "Trying to calling apts_w.sh..."
    ./apts_w.sh \
--nr_models_array "${nr_models_array[*]}" 
sleep 1