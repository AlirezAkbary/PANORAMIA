#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:4
#SBATCH --mem=32000M
#SBATCH --time=0-30:00
#SBATCH --account=def-t55wang


module load StdEnv/2023 arrow/15.0.1 rust/1.76.0 python scipy-stack

source ../../test-priv/test-priv-env/bin/activate

epsilon=$1
clip_norm=$2
lr=$3
gradient_accumulation_steps=$4
num_epoch=$5


log_dir="/home/aaa208/scratch/PANORAMIA/experiments/neurips/dp_gpt_large_generator/epsilon_$epsilon/clip_norm_$clip_norm/lr_$lr/gradient_accumulation_steps_$gradient_accumulation_steps/num_epoch_$num_epoch"
output_dir="/home/aaa208/scratch/PANORAMIA/outputs/neurips/dp_gpt_large_generator/saved_model/epsilon_$epsilon/clip_norm_$clip_norm/lr_$lr/gradient_accumulation_steps_$gradient_accumulation_steps/num_epoch_$num_epoch"

run_name="epsilon_$epsilon/clip_norm_$clip_norm/lr_$lr/gradient_accumulation_steps_$gradient_accumulation_steps/num_epoch_$num_epoch"


python -m src.main  --base_log_dir $log_dir \
                    --base_project_name "PANORAMIA-neurips-dp_gpt_large_generator" \
                    --dataset_path "EleutherAI/wikitext_document_level" \
                    --dataset_name "wikitext-103-raw-v1" \
                    --dataset_data_split_percentage 16 \
                    --dataset_validation_size 0.1 \
                    --dataset_test_size 0.1 \
                    --dataset_num_chunks_keep 50 \
                    --dataset_seed 8 \
                    --dataset_do_shuffle \
                    --dataset_pretrained_model_name_or_path "gpt2" \
                    --dataset_block_size 64 \
                    --dataset_generator_train_percent 35 \
                    --dataset_prompt_sampling_percent 15 \
                    --dataset_helper_model_percent 100 \
                    --dataset_helper_model_train_data_mode "syn" \
                    --dataset_syn_audit_percent 45 \
                    --dataset_mia_num_train 6000 \
                    --dataset_mia_num_val 1000 \
                    --dataset_mia_num_test 10000 \
                    --dataset_audit_mode "RMFN" \
                    --dataset_game_seed 30 \
                    --generator_train_pretrained_model_name_or_path "gpt2-large" \
                    --generator_train_saving_dir $output_dir \
                    --generator_train_run_name $run_name \
                    --generator_train_seed 42 \
                    --generator_train_train_with_dp \
                    --generator_train_optimization_per_device_batch_size 16 \
                    --generator_train_optimization_epoch $num_epoch \
                    --generator_train_optimization_learning_rate $lr \
                    --generator_train_optimization_weight_decay 0.0 \
                    --generator_train_optimization_warmup_steps 100 \
                    --generator_train_optimization_gradient_accumulation_steps $gradient_accumulation_steps \
                    --generator_train_dp_per_example_max_grad_norm $clip_norm \
                    --generator_train_dp_target_epsilon $epsilon