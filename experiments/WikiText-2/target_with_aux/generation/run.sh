#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:4
#SBATCH --mem=32000M
#SBATCH --time=0-5:00
#SBATCH --account=def-t55wang


module load gcc/9.3.0 arrow/10.0.1 python scipy-stack

source panoramia_venv/bin/activate


python -m src.main --use_yml_config --path_yml_config experiments/WikiText-2/target_with_aux/generation/generation.yaml

nvidia-smi

deactivate
