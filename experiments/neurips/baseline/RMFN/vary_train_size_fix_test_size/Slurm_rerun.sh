#!/bin/bash

# Define the base directory
base_dir="/home/aaa208/scratch/PANORAMIA/outputs/neurips/attacks/baseline/RMFN/vary_train_size_fix_test_size/"

filename="experiments/neurips/baseline/RMFN/vary_train_size_fix_test_size/run_args.sh"

# Define ranges for var1, var2, and var3
var1_range=(1000 5000 10000 15000 20000 25000 29000)  # Example range for var1
var2_range=(10 11 12 13 14)  # Example range for var2
var3_range=(0 1 2 3 4)  # Example range for var3

# Loop through var1, var2, and var3 values
for var1 in "${var1_range[@]}"; do
    for var2 in "${var2_range[@]}"; do
        for var3 in "${var3_range[@]}"; do
            # Construct the directory path
            directory="n_train_${var1}/mix/mia_seed_${var2}/train_seed_${var3}"

            # Check if the file exists
            if [ ! -f "${base_dir}${directory}/test_preds.npy" ]; then
                # echo "${base_dir}${directory}"
                rm -r "${base_dir}${directory}/"
                sbatch --exclude=cdr2550,cdr2591 $filename "baseline" $var1 "mix" $var2 $var3
            fi
        done
    done
done
