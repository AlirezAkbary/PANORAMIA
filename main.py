import logging
import os
import sys

import yaml
from easydict import EasyDict

from arguments import init_args
from src.datasets.datamodule import PANORAMIADataModule
from src.generator.train import fine_tune_generator

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
    # Train if synthetic data doesn't already exist
    if (config.dataset.path_to_synthetic_data_dir is None):
        fine_tune_generator(config, dm)


    # --------------------
    # Part 2. Generate/Load Synthetic Samples
    # --------------------

    # generate_synthetic()


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


    
    


    