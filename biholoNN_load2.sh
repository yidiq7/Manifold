for d in experiments.mrd/scanpsi500d/*/ ; do
    python biholoNN_load.py "$d" 100_500_100_1_seed1234
done
for seed in 1234 320 235 ; do
for d in experiments.mrd/scanpsi500c/*$seed/ ; do
    python biholoNN_load.py "$d" 100_500_100_1_seed$seed
done
done
for d in experiments.mrd/scanphi500{b,c}/*/ ; do
    python biholoNN_load.py "$d" 100_500_100_1_seed1729
done
