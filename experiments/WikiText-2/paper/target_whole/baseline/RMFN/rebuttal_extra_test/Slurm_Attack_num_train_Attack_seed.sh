#!/bin/bash

filename="experiments/WikiText-2/paper/target_whole/baseline/RMFN/rebuttal_extra_test/run_args.sh"

for attack_main in "baseline"
do
    for attack_num_train in 6000
    do
        for game_seed_num in 30
        do
            for seed_num in 0 1 2 3 4
            do
                for extra_m in 22000 24000
                do
                    sbatch --exclude=cdr2550,cdr2591 $filename $seed_num $attack_num_train $attack_main $game_seed_num $extra_m
                done
            done
        done
    done
done