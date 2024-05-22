#!/bin/bash

# Define the base directory
base_dir="/home/aaa208/scratch/PANORAMIA/outputs/neurips/attacks/baseline/RMFN/vary_test_size"

# Define ranges for var1, var2, and var3
var1_range=(1000 5000 10000 13000 15000)  # Example range for var1
var2_range=(10 11 12 13 14)  # Example range for var2
var3_range=(0 1 2 3 4)  # Example range for var3

# Loop through var1, var2, and var3 values
for var1 in "${var1_range[@]}"; do
    for var2 in "${var2_range[@]}"; do
        for var3 in "${var3_range[@]}"; do
            # Construct the directory path
            directory="/n_test_${var1}/mix/mia_seed_${var2}/train_seed_${var3}"

            # Get the creation time of the directory
            creation_time=$(stat -c %z "${base_dir}${directory}")

            # Print the creation time
            echo "Creation time of ${base_dir}${directory}: $creation_time"
        done
    done
done
