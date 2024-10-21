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


log_dir="/home/aaa208/scratch/PANORAMIA/experiments/neurips/target/dp_models/epsilon_$epsilon/clip_norm_$clip_norm/lr_$lr/gradient_accumulation_steps_$gradient_accumulation_steps/num_epoch_$num_epoch"
output_dir="/home/aaa208/scratch/PANORAMIA/outputs/neurips/audit_model/saved_model/dp_models/epsilon_$epsilon/clip_norm_$clip_norm/lr_$lr/gradient_accumulation_steps_$gradient_accumulation_steps/num_epoch_$num_epoch"

run_name="epsilon_$epsilon/clip_norm_$clip_norm/lr_$lr/gradient_accumulation_steps_$gradient_accumulation_steps/num_epoch_$num_epoch"


python -m src.main  --base_log_dir $log_dir \
                    --base_project_name "PANORAMIA-neurips-dp_target_models" \
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
                    --generator_train_pretrained_model_name_or_path "gpt2" \
                    --generator_train_saving_dir "/home/aaa208/scratch/PANORAMIA/outputs/WikiText-2/generator/shuffle_wt_2_test/saved_model/checkpoint-768/" \
                    --generator_train_run_name "generator-fine-tune-paper-target_whole" \
                    --generator_train_seed 42 \
                    --generator_train_optimization_per_device_batch_size 64 \
                    --generator_train_optimization_epoch 40 \
                    --generator_train_optimization_learning_rate 2e-05 \
                    --generator_train_optimization_weight_decay 0.01 \
                    --generator_train_optimization_warmup_steps 100 \
                    --generator_generation_saving_dir "/home/aaa208/scratch/PANORAMIA/outputs/WikiText-2/generator/shuffle_wt_2_test/saved_synthetic_data/" \
                    --generator_generation_syn_file_name "syn_data.csv" \
                    --generator_generation_seed 42 \
                    --generator_generation_parameters_batch_size 64 \
                    --generator_generation_parameters_prompt_sequence_length 64 \
                    --generator_generation_parameters_max_length 128 \
                    --generator_generation_parameters_top_k 200 \
                    --generator_generation_parameters_top_p 1 \
                    --generator_generation_parameters_temperature 1 \
                    --generator_generation_parameters_num_return_sequences 5 \
                    --audit_target_pretrained_model_name_or_path "gpt2" \
                    --audit_target_saving_dir $output_dir \
                    --audit_target_seed 42 \
                    --audit_target_run_name $run_name \
                    --audit_target_embedding_type "loss_seq" \
                    --audit_target_train_with_DP \
                    --audit_target_optimization_learning_rate $lr \
                    --audit_target_optimization_weight_decay 0.0 \
                    --audit_target_optimization_warmup_steps 100 \
                    --audit_target_optimization_batch_size 16 \
                    --audit_target_optimization_epoch $num_epoch \
                    --audit_target_optimization_gradient_accumulation_steps  $gradient_accumulation_steps \
                    --audit_target_dp_target_epsilon $epsilon \
                    --audit_target_dp_per_example_max_grad_norm $clip_norm 
