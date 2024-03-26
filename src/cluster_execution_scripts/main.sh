#!/bin/bash -l

echo "Running main.sh...."

nr_models_array=(2 4)
minibatch_sizes=(10000 60000)
IFS=',' 
echo "Trying to calling apts_w.sh..."
    ./apts_w.sh \
--nr_models_array "${nr_models_array[*]}"\
--minibatch_sizes "${minibatch_sizes[*]}"
 
