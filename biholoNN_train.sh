#!/bin/bash
for psi in -1 0 1; do
    for layers in "50_50_1" "70_70_70_1" ; do
        python biholoNN_train.py --seed 1234 \
                                 --n_pairs 1000 \
                                 --batch_size 2 \
                                 --function "f0" \
                                 --psi $psi \
                                            \
                                 --layers $layers \
                                 --load_model "experiments.yidi/biholo/f0_psi0.5/$layers" \
                                 --save_dir "experiments.yidi/biholo/f0_psi$psi/" \
                                 --save_name $layers \
                                                     \
                                 --max_epochs 10 \
                                 --loss_func "weighted_MAPE" 
    done
done

for psi in -1 0 1; do
    for phi in -1 0 1; do
        for layers in "50_50_1" "70_70_70_1" ; do
				    python biholoNN_train.py --seed 1234 \
					    											 --n_pairs 100000 \
						            						 --batch_size 5000 \
												          	 --function "f1" \
																	   --psi $psi \
                                     --phi $phi \
                                                \
																		 --layers $layers \
																		 --load_model "experiments.yidi/biholo/f0_psi0.5/$layers" \
																		 --save_dir "experiments.yidi/biholo/f1_psi${psi}_phi${phi}/" \
																		 --save_name $layers \
																		                     \
																		 --max_epochs 400 \
																		 --loss_func "weighted_MAPE" 
        done
    done
done
for psi in -1 0 1; do
    for alpha in -1 0 1; do
        for layers in "50_50_1" "70_70_70_1" ; do
				    python biholoNN_train.py --seed 1234 \
					    											 --n_pairs 100000 \
						            						 --batch_size 5000 \
												          	 --function "f2" \
																	   --psi $psi \
                                     --alpha $alpha \
                                                    \
																		 --layers $layers \
																		 --load_model "experiments.yidi/biholo/f0_psi0.5/$layers" \
																		 --save_dir "experiments.yidi/biholo/f2_psi${psi}_alpha${alpha}/" \
																		 --save_name $layers \
																		                     \
																		 --max_epochs 400 \
																		 --loss_func "weighted_MAPE" 
        done
    done
done
