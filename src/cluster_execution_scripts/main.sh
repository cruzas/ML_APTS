#!/bin/bash -l

echo "Running main.sh...."

nr_models_array=(2)
IFS=',' 
echo "Trying to calling apts_w.sh..."
    ./apts_w.sh \
--nr_models_array "${nr_models_array[*]}" 
