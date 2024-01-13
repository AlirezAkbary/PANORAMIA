#!/bin/bash

filename="experiments/WikiText-2/helper_with_syn_block_size_64/helper_with_real/RMFN/run_args.sh"

for seed_num in 0 1 2 3 4 5 6 7 8 9
do
    sbatch $filename $seed_num
done