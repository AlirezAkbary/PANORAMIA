#!/bin/bash

filename="experiments/neurips/dp_gpt_large_generator/run_dp_target_args.sh"

for epsilon in 3
do
    for clip_norm in 0.1 1
    do
        for lr in 0.001 0.0001
        do
            for gradient_accumulation_steps in 64 256
            do
                for num_epoch in 20 40
                do
                    if [ "$epsilon" -eq 3 ] && [ "$(echo "$clip_norm == 0.1" | bc)" -eq 1 ] && [ "$(echo "$lr == 0.001" | bc)" -eq 1 ] && [ "$gradient_accumulation_steps" -eq 64 ] && [ "$num_epoch" -eq 20 ]; then
                        continue
                    fi
                    # echo $epsilon $clip_norm $lr $gradient_accumulation_steps $num_epoch
                    sbatch --exclude=cdr2550,cdr2591 $filename $epsilon $clip_norm $lr $gradient_accumulation_steps $num_epoch
                done
            done  
        done
    done
done