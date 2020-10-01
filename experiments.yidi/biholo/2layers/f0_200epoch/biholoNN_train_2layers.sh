#!/bin/bash
for layers in "100_100_1" "300_300_1" "500_500_1" ; do
    for seed in 1234 1235; do
        python biholoNN_train_2layers.py $seed $layers
    done
done
