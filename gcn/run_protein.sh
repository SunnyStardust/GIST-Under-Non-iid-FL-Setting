#!/bin/bash


for units in 25000 50000; do
    for lr in 0.001 0.0001 0.00001; do
        for layer in 1 2 3; do
            echo "$units $lr $layer"
            python train_gist_noniid.py --dataset=PROTEINS --n-hidden="$units" --lr="$lr" --n-layers="$layer" --num_subnet=10
        done
    done
done
