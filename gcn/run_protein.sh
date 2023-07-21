#!/bin/bash

source ~/miniconda3/bin/activate deeplearning

for units in 10000 15000 20000; do
    for lr in 0.1 0.01 0.001 0.0001 0.00001; do
        for layer in 1 2 3; do
            echo "$units $lr $layer"
            python train_gist_noniid.py --dataset=PROTEINS --n-hidden="$units" --lr="$lr" --n-layers="$layer" --num_subnet=10
        done
    done
done
