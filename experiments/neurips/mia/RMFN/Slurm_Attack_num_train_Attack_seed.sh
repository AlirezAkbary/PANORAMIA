#!/bin/bash

filename="experiments/neurips/mia/RMFN/run_args.sh"

for attack_main in "mia"
do
    for attack_num_train in 20000
    do
        for which_helper in "mix" 
        do
            for target_checkpoint in "checkpoint-50000" 
            do
                for mia_seed_num in 10 11 12 13 14
                do
                    for training_seed_num in 0 1 2 3 4 
                    do 
                        sbatch --exclude=cdr2550,cdr2591 $filename $attack_main $attack_num_train $which_helper $target_checkpoint $mia_seed_num $training_seed_num
                    done
                done
            done
        done  
    done
done