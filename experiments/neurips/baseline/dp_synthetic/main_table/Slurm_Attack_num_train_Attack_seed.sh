#!/bin/bash

filename="experiments/neurips/baseline/dp_synthetic/main_table/run_args.sh"

for attack_main in "baseline"
do
    for attack_num_train in 20000
    do
        for attack_num_test in 19500
        do    
            for which_helper in "mix" 
            do
                for mia_seed_num in 10 11 12 13 14
                do
                    for training_seed_num in 0 1 2 3 4 
                    do 
                        sbatch --exclude=cdr2550,cdr2591 $filename $attack_main $attack_num_train $attack_num_test $which_helper $mia_seed_num $training_seed_num
                    done
                done
            done  
        done
    done
done