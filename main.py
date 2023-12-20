import logging
import os
import sys

import yaml
from easydict import EasyDict
from transformers import AutoModelForCausalLM

from arguments import init_args
from src.datasets.datamodule import PANORAMIADataModule
from src.generator.train import fine_tune_generator
from src.generator.generate import generate_synthetic_samples
from src.audit_model.train import train_audit_model

def main(config: EasyDict):
    """
    runs the whole pipeline of PANORAMIA
    """
    # instantiating a data module of PANORAMIA
    dm = PANORAMIADataModule(
        **config.dataset
    )
    
    

    # --------------------
    # Part 1. Generative Model Training/Loading
    # --------------------

    # Train if the generative model is not provided
    if os.path.exists(config.generator.train.saving_dir):
        logging.info(f"Loading the generator model from {config.generator.train.saving_dir} ...")
        generator_model = AutoModelForCausalLM.from_pretrained(config.generator.train.saving_dir)
    else:
        generator_model = fine_tune_generator(config, dm)

    
    # --------------------
    # Part 2. Generate/Load Synthetic Samples
    # --------------------

    if not os.path.exists(config.generator.generation.saving_dir):
        generate_synthetic_samples(config, dm, generator_model)

    
    # Handling the synthetic dataset in data module
    dm.setup_synthetic_dataset()


    # del the generator model from memory
    del generator_model

    # --------------------
    # Part 3. Train/Load Audit Model
    # --------------------

    # train_audit_model()

    # --------------------
    # Part 4. MIA/Baseline Attack
    # --------------------

    # attack()






if __name__ == "__main__":
    # argument settings
    args = init_args()

    # determine to load from a yaml config or the arguments
    if args.use_yml_config:
        # read from yaml
        with open(args.path_yml_config, 'r') as stream:
            config = yaml.safe_load(stream)
    else:
        # TODO
        raise NotImplementedError
    
    # convert normal dictionary of config to attribute dictionary
    config = EasyDict(config)

    # Setup logging directory
    os.makedirs(config.base.log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(
            config.base.log_dir,
            "output.log"
        ),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main(config)


    
    


    