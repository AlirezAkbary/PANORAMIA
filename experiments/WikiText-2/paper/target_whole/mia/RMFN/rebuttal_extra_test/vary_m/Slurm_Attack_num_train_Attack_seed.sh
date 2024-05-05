#!/bin/bash

filename="experiments/WikiText-2/paper/target_whole/mia/RMFN/rebuttal_extra_test/vary_m/run_args.sh"

for attack_main in "mia"
do
    for attack_num_train in 6000
    do
        for target_checkpoint in "checkpoint-2500" "checkpoint-10000" "checkpoint-20000"
        do
            for seed_num in 0 1 2 3 4 
            do
                for m in 1000 2000 4000 6000 8000 10000
                do
                    sbatch --exclude=cdr2550,cdr2591 $filename $seed_num $attack_num_train $attack_main $target_checkpoint $m
                done
            done
        done
    done
done