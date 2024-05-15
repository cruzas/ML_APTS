#!/bin/bash

# Define the arrays for each parameter
optimizers=("SGD" "Adam")
batch_sizes=(100 500)
learning_rates=(0.01 0.001)
trials=3
epochs=2
datasets=("mnist" "cifar10")

# Iterate over each combination of parameters
for optimizer in "${optimizers[@]}"
do
    for batch_size in "${batch_sizes[@]}"
    do
        for learning_rate in "${learning_rates[@]}"
        do
            for dataset in "${datasets[@]}"
            do
                # Run the training script in parallel
                python train.py --optimizer $optimizer --batch_size $batch_size --learning_rate $learning_rate --epochs $epochs --dataset $dataset --trials $trials 
            done
        done
    done
done
