for d in experiments.mrd/scanpsi1000/*/ ; do
    python biholoNN_load.py "$d" 100_1000_100_1_seed1234
done
