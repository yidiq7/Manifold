#!/bin/bash
for layers in "70_70_70_1"; do
    for seed in 1234; do
        python biholoNN_train_3layers.py $seed $layers
   done
done
#for layers in "100_50_1"; do
#    for seed in 1234; do
#        python biholoNN_train_2layers.py $seed $layers
#    done
#done
