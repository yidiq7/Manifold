#!/bin/bash

psi = 0.5

for seed in 17 1729
do for phi in 0.0 0.5 1.0 2.0 2.5 3.0 4.0 5.0
	do
	echo "psi = $psi ; phi = $phi"
	python3 biholoNN_train_SL_f1.py seed psi phi
	done
done

