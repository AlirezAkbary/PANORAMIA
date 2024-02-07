#!/bin/bash

filename="experiments/WikiText-2/target_with_aux/mia_with_helper/RMFN/e50/run_args.sh"

for attack_main in "mia"
do
    for attack_num_train in 6000
    do
        for seed_num in 0 1 2 3 4 5 6 7 8 9
        do
            sbatch --exclude=cdr2550,cdr2591 $filename $seed_num $attack_num_train $attack_main
        done
    done
done