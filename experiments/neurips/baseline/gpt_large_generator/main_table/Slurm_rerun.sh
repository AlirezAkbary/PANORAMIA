#!/bin/bash

# Define the base directory
base_dir="/home/aaa208/scratch/PANORAMIA/outputs/neurips/attacks/baseline/gpt_large_generator/main_table/n_train_20000/"

filename="experiments/neurips/baseline/gpt_large_generator/main_table/run_args.sh"

# Define ranges for var1, var2, and var3
var1_range=(10000)  # Example range for var1
var2_range=(10 11 12 13 14)  # Example range for var2
var3_range=(0 1 2 3 4)  # Example range for var3

# Loop through var1, var2, and var3 values
for var1 in "${var1_range[@]}"; do
    for var2 in "${var2_range[@]}"; do
        for var3 in "${var3_range[@]}"; do
            # Construct the directory path
            directory="n_test_${var1}/mix/mia_seed_${var2}/train_seed_${var3}"

            # Check if the file exists
            if [ ! -f "${base_dir}${directory}/test_preds.npy" ]; then
                # echo "${base_dir}${directory}"
                rm -r "${base_dir}${directory}/"
                sbatch --exclude=cdr2550,cdr2591 $filename "baseline" 20000 $var1 "mix" $var2 $var3
            fi
        done
    done
done
