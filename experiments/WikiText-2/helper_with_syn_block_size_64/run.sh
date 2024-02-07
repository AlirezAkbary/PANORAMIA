#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32000M
#SBATCH --time=0-2:00
#SBATCH --account=def-t55wang


module load gcc/9.3.0 arrow/10.0.1 python scipy-stack

source panoramia_venv/bin/activate


python -m src.main --use_yml_config --path_yml_config /home/aaa208/projects/def-t55wang/aaa208/PANORAMIA/PANORAMIA/experiments/WikiText-2/helper_with_syn_block_size_64/helper_with_syn.yaml

nvidia-smi

deactivate
