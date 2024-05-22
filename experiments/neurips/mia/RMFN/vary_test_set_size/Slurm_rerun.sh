#!/bin/bash

# Define the base directory
base_dir="/home/aaa208/scratch/PANORAMIA/outputs/neurips/attacks/mia/RMFN/vary_test_size/"

filename="experiments/neurips/mia/RMFN/vary_test_set_size/run_args.sh"

# Define ranges for var1, var2, and var3
var1_range=("checkpoint-12000" "checkpoint-25000" "checkpoint-50000")
var2_range=(1000 5000 10000 13000 15000)  # Example range for var1
var3_range=(10 11 12 13 14)  # Example range for var2
var4_range=(0 1 2 3 4)  # Example range for var3

# Loop through var1, var2, and var3 values
for var1 in "${var1_range[@]}"; do
    for var2 in "${var2_range[@]}"; do
        for var3 in "${var3_range[@]}"; do
            for var4 in "${var4_range[@]}"; do
                # Construct the directory path
                directory="${var1}/n_test_${var2}/mix/mia_seed_${var3}/train_seed_${var4}"

                # Check if the file exists
                if [ ! -f "${base_dir}${directory}/test_preds.npy" ]; then
                    echo "${base_dir}${directory}"
                    # rm -r "${base_dir}${directory}/"
                    # sbatch --exclude=cdr2550,cdr2591 $filename "mia" $var2 "mix" $var1 $var3 $var4
                fi
            done
        done
    done
done
