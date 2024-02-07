#!/bin/bash

filename="experiments/WikiText-2/paper/target_whole/plateau/RMFN/run_args.sh"

for attack_main in "baseline"
do
    for attack_num_train in 100 1000 2000 3000 4000 6000 9000 12000 14000
    do
        for game_seed_num in 30
        do
            for seed_num in 0 1 2 3 4
            do
                sbatch --exclude=cdr2550,cdr2591 $filename $seed_num $attack_num_train $attack_main $game_seed_num 
            done
        done
    done
done