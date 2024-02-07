#!/bin/bash

filename="experiments/WikiText-2/paper/target_with_canaries/baseline/RMFMFN/run_args.sh"

for attack_main in "baseline"
do
    for attack_num_train in 6000
    do
        for game_seed_num in 30
        do
            for which_helper in "mix"
            do
                for seed_num in 0 1 2 3 4
                do
                    sbatch --exclude=cdr2550,cdr2591 $filename $seed_num $attack_num_train $attack_main $game_seed_num $which_helper
                done
            done
        done
    done
done