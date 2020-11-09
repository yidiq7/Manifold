#!/bin/bash
for psi in  -2 -1 0 1 2 3 ; do
    for phi in 0 0.4 0.8 1.2 1.6 2 2.4 2.8 3 ; do
        for layers in "" "300_300_300_1"; do
            for np in 4000; do
                python biholoNN_train.py --seed 4242 \
                                         --n_pairs 2000 \
                                         --batch_size 10000 \
                                         --function "f1" \
                                         --psi $psi \
                                         --phi $phi \
                                         --layers $layers \
                                         --load_model "experiments.final/output62/f1_psi${psi}_phi${phi}/$layers/" \
                                         --save_dir "experiments.final/output69v/f1_psi${psi}_phi${phi}_${np}/" \
                                         --save_name $layers \
                                         --optimizer 'Adam'\
                                         --learning_rate 0.001 \
                                         --max_epochs 0\
                                         --loss_func "weighted_MAPE" 
            done
        done
    done
done

