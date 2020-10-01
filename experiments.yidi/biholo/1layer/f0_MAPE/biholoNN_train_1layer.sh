#!/bin/bash
#for layers in "5_1" "10_1" "50_1" "100_1" ; do
for layers in "300_1" "500_1" ; do
    for seed in 1234; do
        python biholoNN_train_1layer.py $seed $layers
    done
done
