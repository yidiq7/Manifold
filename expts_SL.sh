#!/bin/bash

for seed in 17 1729
do for psi in 0 0.5 1 2.5 5 7.5 10
	do
	echo "psi = $psi"
	python3 biholoNN_train_SL.py seed psi 
	done
done

