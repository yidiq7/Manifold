#!/bin/bash

for seed in 1357; do
    #for layers in "500_500_500_1000_1" "500_500_500_2000_1" "500_500_500_3000_1" ; do
    for layers in "2000_2000_2000_2000_1" ; do
        python biholoNN_train_4layers.py $seed $layers
    done
done
