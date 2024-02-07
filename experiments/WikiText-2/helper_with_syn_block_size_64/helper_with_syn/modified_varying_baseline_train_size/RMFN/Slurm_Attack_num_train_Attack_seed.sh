#!/bin/bash

filename="experiments/WikiText-2/helper_with_syn_block_size_64/helper_with_syn/modified_varying_baseline_train_size/RMFN/run_args.sh"

for attack_main in "baseline" "mia"
do
    for attack_num_train in 100 300 500 750 1000 1250 1500 1750 2000
    do
        for seed_num in 0 1 2 3 4 5 6 7 8 9
        do
            sbatch --exclude=cdr2550,cdr2591 $filename $seed_num $attack_num_train $attack_main
        done
    done
done