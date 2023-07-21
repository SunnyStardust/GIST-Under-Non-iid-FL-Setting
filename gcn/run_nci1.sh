#!/bin/bash

source ~/miniconda3/bin/activate deeplearning

N=3
for units in 12000 24000 36000; do
    for lr in 0.1 0.01 0.001 0.0001 0.00001; do
        for layer in 1 2 3; do
            ((i=i%N)); ((i++==0)) && wait
            echo "$units $lr $layer"
            python train_gist_noniid.py --dataset=NCI1 --n-hidden="$units" --lr="$lr" --n-layers="$layer" --num_subnet=30 &
        done
        # python train_gist_noniid.py --dataset=NCI1 --n-hidden="$units" --lr="$lr" --n-layers=1 --num_subnet=30 &
        # python train_gist_noniid.py --dataset=NCI1 --n-hidden="$units" --lr="$lr" --n-layers=2 --num_subnet=30 &
        # python train_gist_noniid.py --dataset=NCI1 --n-hidden="$units" --lr="$lr" --n-layers=3 --num_subnet=30 &
    done
done
