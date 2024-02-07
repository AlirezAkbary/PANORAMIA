#!/bin/bash

filename="experiments/WikiText-2/target_on_first_90_percent/baseline/RMRN/check_6000/run_args.sh"

for attack_main in "baseline"
do
    for attack_num_train in 6000
    do
        for game_seed_num in 56
        do
            for seed_num in 0 1 2 3 4
            do
                sbatch --exclude=cdr2550,cdr2591 $filename $seed_num $attack_num_train $attack_main $game_seed_num
            done
        done
    done
done