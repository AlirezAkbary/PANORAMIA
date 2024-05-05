#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:4
#SBATCH --mem=32000M
#SBATCH --time=0-1:00
#SBATCH --account=def-t55wang

module load StdEnv/2023 arrow/15.0.1 rust/1.76.0 python scipy-stack

source ../../test-priv/test-priv-env/bin/activate

python -m src.main --use_yml_config --path_yml_config experiments/WikiText-2/paper/target_whole/generator_generation/dp_generator/extra_syn_data/generator_generation.yaml

nvidia-smi

deactivate
