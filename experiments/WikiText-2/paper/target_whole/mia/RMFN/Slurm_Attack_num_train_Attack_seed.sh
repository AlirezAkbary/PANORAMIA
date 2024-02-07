#!/bin/bash

filename="experiments/WikiText-2/paper/target_whole/mia/RMFN/run_args.sh"

for attack_main in "mia"
do
    for attack_num_train in 6000
    do
        for target_checkpoint in "checkpoint-2500" 
        do
            for seed_num in 0 1 2 3 4 
            do
                sbatch --exclude=cdr2550,cdr2591 $filename $seed_num $attack_num_train $attack_main $target_checkpoint
            done
        done
    done
done