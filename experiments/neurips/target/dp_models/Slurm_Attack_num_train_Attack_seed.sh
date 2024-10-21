#!/bin/bash

filename="experiments/neurips/target/dp_models/run_dp_target_args.sh"

for epsilon in 1 3 10
do
    for clip_norm in 0.1
    do
        for lr in 0.001
        do
            for gradient_accumulation_steps in 256
            do
                for num_epoch in 20
                do
                    sbatch $filename $epsilon $clip_norm $lr $gradient_accumulation_steps $num_epoch
                done
            done  
        done
    done
done