#!/bin/bash

filename="experiments/WikiText-2/paper/target_whole/baseline/RMFN/rebuttal_extra_test/vary_m/run_args.sh"

for attack_main in "baseline"
do
    for attack_num_train in 6000
    do
        for game_seed_num in 30
        do
            for seed_num in 0
            do
                for m in 1000 2000 4000 6000 8000 10000
                do
                    sbatch --exclude=cdr2550,cdr2591 $filename $seed_num $attack_num_train $attack_main $game_seed_num $m
                done
            done
        done
    done
done