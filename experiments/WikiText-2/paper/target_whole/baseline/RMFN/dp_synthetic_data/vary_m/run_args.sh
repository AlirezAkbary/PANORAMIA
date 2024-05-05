#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32000M
#SBATCH --time=0-00:30
#SBATCH --account=def-t55wang

module load StdEnv/2023 arrow/15.0.1 rust/1.76.0 python scipy-stack

source ../../test-priv/test-priv-env/bin/activate


attack_seed_num=$1
attack_num_train=$2
attack_main=$3
game_seed_num=$4
m=$5

                                    
python -m src.main  --base_log_dir "/home/aaa208/scratch/PANORAMIA/experiments/WikiText-2/paper/target_whole/baseline/RMFN/dp_synthetic_data/game_seed_$game_seed_num/rebuttal_extra_test/" \
                    --base_project_name "PANORAMIA-Wiki_Text_2-paper-target_whole-baseline-RMFN-dp_synthetic_data" \
                    --base_attack_main $attack_main \
                    --dataset_path "wikitext" \
                    --dataset_name "wikitext-2-raw-v1" \
                    --dataset_path_to_synthetic_data "/home/aaa208/scratch/PANORAMIA/outputs/WikiText-2/generator/paper/target_whole/saved_dp_synthetic_data/syn_data.csv" \
                    --dataset_synthetic_text_column_name "text" \
                    --dataset_seed 8 \
                    --dataset_do_shuffle \
                    --dataset_pretrained_model_name_or_path "gpt2" \
                    --dataset_block_size 64 \
                    --dataset_generator_train_percent 34 \
                    --dataset_prompt_sampling_percent 18 \
                    --dataset_target_model_percent 76 \
                    --dataset_helper_model_percent 100 \
                    --dataset_helper_model_train_data_mode "syn" \
                    --dataset_syn_audit_percent 30 \
                    --dataset_mia_num_train $attack_num_train \
                    --dataset_mia_num_val 1000 \
                    --dataset_mia_num_test $m \
                    --dataset_audit_mode "RMFN" \
                    --dataset_game_seed $game_seed_num \
                    --dataset_include_auxilary \
                    --generator_train_pretrained_model_name_or_path "gpt2" \
                    --generator_train_saving_dir "/home/aaa208/scratch/PANORAMIA/outputs/WikiText-2/generator/paper/target_whole/dp_fine_tune/" \
                    --generator_train_run_name "generator-fine-tune-paper-target_whole" \
                    --generator_train_seed 42 \
                    --generator_train_optimization_per_device_batch_size 64 \
                    --generator_train_optimization_epoch 40 \
                    --generator_train_optimization_learning_rate 2e-05 \
                    --generator_train_optimization_weight_decay 0.01 \
                    --generator_train_optimization_warmup_steps 100 \
                    --generator_generation_saving_dir "/home/aaa208/scratch/PANORAMIA/outputs/WikiText-2/generator/paper/target_whole/saved_dp_synthetic_data/" \
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
                    --audit_target_saving_dir "/home/aaa208/scratch/PANORAMIA/outputs/WikiText-2/audit_model/paper/target_whole/saved_model/epoch_100/epoch_100/checkpoint-2500/" \
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
                    --audit_helper_saving_dir "/home/aaa208/scratch/PANORAMIA/outputs/WikiText-2/audit_model/paper/target_whole/helper_with_syn/dp_synthetic_data/epoch_40/checkpoint-2265/" \
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
                    --attack_baseline_net_type "mix" \
                    --attack_baseline_distinguisher_type "GPT2Distinguisher" \
                    --attack_baseline_run_name "RMFN" \
                    --attack_baseline_training_args_seed $attack_seed_num \
                    --attack_baseline_training_args_output_dir "/scratch/aaa208/PANORAMIA/outputs/WikiText-2/attacks/paper/target_whole/baseline/RMFN/dp_synthetic_data/\game_seed_$game_seed_num/" \
                    --attack_baseline_training_args_which_test "test_$m" \
                    --attack_baseline_training_args_max_steps 4000 \
                    --attack_baseline_training_args_batch_size 64 \
                    --attack_baseline_training_args_warmup_steps 500 \
                    --attack_baseline_training_args_weight_decay 0.01 \
                    --attack_baseline_training_args_learning_rate 3e-05 \
                    --attack_baseline_training_args_reg_coef 0 \
                    --attack_baseline_training_args_phase1_max_steps 2000 \
                    --attack_baseline_training_args_phase1_batch_size 64 \
                    --attack_baseline_training_args_phase1_learning_rate 0.003 \
                    --attack_baseline_training_args_phase1_reg_coef 1 \
                    --attack_baseline_training_args_logging_steps 10 \
                    --attack_baseline_training_args_save_strategy "no" \
                    --attack_baseline_training_args_evaluation_strategy "epoch" \
                    --attack_baseline_training_args_overwrite_output_dir \
                    --attack_baseline_training_args_max_fpr 0.1 \
                    --attack_baseline_training_args_evaluate_every_n_steps 50 \
                    --attack_baseline_training_args_metric_for_best_model "auc"

nvidia-smi

deactivate
