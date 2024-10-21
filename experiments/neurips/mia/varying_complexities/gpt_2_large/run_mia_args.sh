#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32000M
#SBATCH --time=0-10:00
#SBATCH --account=def-t55wang


module load StdEnv/2023 arrow/15.0.1 rust/1.76.0 python scipy-stack

source ../../test-priv/test-priv-env/bin/activate


attack_main=$1
attack_num_train=$2
attack_num_test=$3
which_helper=$4
target_checkpoint=$5
mia_seed_num=$6
training_seed_num=$7


log_dir="/home/aaa208/scratch/PANORAMIA/experiments/neurips/mia/varying_complexities/gpt_2_large/$target_checkpoint/n_train_$attack_num_train/n_test_$attack_num_test/$which_helper/mia_seed_$mia_seed_num/train_seed_$training_seed_num/"
output_dir="/home/aaa208/scratch/PANORAMIA/outputs/neurips/attacks/mia/varying_complexities/gpt_2_large/$target_checkpoint/n_train_$attack_num_train/n_test_$attack_num_test/$which_helper/mia_seed_$mia_seed_num/train_seed_$training_seed_num/"

audit_mode="RMFN_fixed_test"
                      
python -m src.main  --base_log_dir $log_dir \
                    --base_project_name "PANORAMIA-neurips-mia-varying_complexities_gpt2_large-$target_checkpoint-$which_helper" \
                    --base_attack_main $attack_main \
                    --dataset_path "EleutherAI/wikitext_document_level" \
                    --dataset_name "wikitext-103-raw-v1" \
                    --dataset_data_split_percentage 16 \
                    --dataset_validation_size 0.1 \
                    --dataset_test_size 0.1 \
                    --dataset_num_chunks_keep 50 \
                    --dataset_path_to_synthetic_data "/home/aaa208/scratch/PANORAMIA/outputs/neurips/generator/saved_synthetic_data/syn_data.csv" \
                    --dataset_synthetic_text_column_name "text" \
                    --dataset_seed 8 \
                    --dataset_do_shuffle \
                    --dataset_pretrained_model_name_or_path "gpt2" \
                    --dataset_block_size 64 \
                    --dataset_generator_train_percent 35 \
                    --dataset_prompt_sampling_percent 15 \
                    --dataset_helper_model_percent 100 \
                    --dataset_helper_model_train_data_mode "syn" \
                    --dataset_syn_audit_percent 45 \
                    --dataset_mia_num_train $attack_num_train \
                    --dataset_mia_num_val 1000 \
                    --dataset_mia_num_test $attack_num_test \
                    --dataset_mia_seed $mia_seed_num\
                    --dataset_audit_mode $audit_mode \
                    --dataset_game_seed $mia_seed_num \
                    --generator_train_pretrained_model_name_or_path "gpt2" \
                    --generator_train_saving_dir "/home/aaa208/scratch/PANORAMIA/outputs/neurips/generator/saved_model/checkpoint-2784/" \
                    --generator_train_run_name "generator-fine-tune-paper-target_whole" \
                    --generator_train_seed 42 \
                    --generator_train_optimization_per_device_batch_size 64 \
                    --generator_train_optimization_epoch 40 \
                    --generator_train_optimization_learning_rate 2e-05 \
                    --generator_train_optimization_weight_decay 0.01 \
                    --generator_train_optimization_warmup_steps 100 \
                    --generator_generation_saving_dir "/home/aaa208/scratch/PANORAMIA/outputs/neurips/generator/saved_synthetic_data/" \
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
                    --audit_target_saving_dir "/home/aaa208/scratch/PANORAMIA/outputs/neurips/audit_model/saved_model/gpt-2-large/epoch_200/$target_checkpoint/" \
                    --audit_target_seed 42 \
                    --audit_target_run_name "target_epoch_100_block_size_64" \
                    --audit_target_embedding_type "loss_seq" \
                    --audit_target_optimization_learning_rate 2e-05 \
                    --audit_target_optimization_weight_decay 0.01 \
                    --audit_target_optimization_warmup_steps 100 \
                    --audit_target_optimization_batch_size 64 \
                    --audit_target_optimization_epoch 100 \
                    --audit_target_optimization_save_strategy "no" \
                    --audit_helper_pretrained_model_name_or_path "gpt2" \
                    --audit_helper_saving_dir "/home/aaa208/scratch/PANORAMIA/outputs/neurips/audit_model/helper_with_syn/saved_model/epoch_60/checkpoint-2646/" \
                    --audit_helper_seed 42 \
                    --audit_helper_run_name "helper_with_syn_helper_percent_100" \
                    --audit_helper_embedding_type "loss_seq" \
                    --audit_helper_optimization_learning_rate 2e-05 \
                    --audit_helper_optimization_weight_decay 0.01 \
                    --audit_helper_optimization_warmup_steps 100 \
                    --audit_helper_optimization_batch_size 64 \
                    --audit_helper_optimization_epoch 40 \
                    --audit_helper_optimization_save_strategy "epoch" \
                    --audit_helper_optimization_load_best_model_at_end \
                    --audit_helper_optimization_save_total_limit 1 \
                    --attack_mia_net_type "$which_helper" \
                    --attack_mia_distinguisher_type "GPT2Distinguisher" \
                    --attack_mia_run_name "RMFN_main_table" \
                    --attack_mia_training_args_seed $training_seed_num \
                    --attack_mia_training_args_output_dir $output_dir \
                    --attack_mia_training_args_max_steps 10000 \
                    --attack_mia_training_args_batch_size 64 \
                    --attack_mia_training_args_warmup_steps 500 \
                    --attack_mia_training_args_weight_decay 0.01 \
                    --attack_mia_training_args_learning_rate 3e-05 \
                    --attack_mia_training_args_reg_coef 0 \
                    --attack_mia_training_args_phase1_max_steps 2500 \
                    --attack_mia_training_args_phase1_batch_size 64 \
                    --attack_mia_training_args_phase1_learning_rate 0.003 \
                    --attack_mia_training_args_phase1_reg_coef 1 \
                    --attack_mia_training_args_max_fpr 0.1 \
                    --attack_mia_training_args_evaluate_every_n_steps 100 \
                    --attack_mia_training_args_metric_for_best_model "eps"

nvidia-smi

deactivate
