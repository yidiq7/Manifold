#!/bin/bash

for seed in 1234 1235 2345 ; do
    for layers in "1000_500_500_100_1" "500_1000_500_100_1" "1000_1000_500_100_1" "1000_500_500_500_1" ; do
        python biholoNN_train_4layers.py $seed $layers
    done
done
