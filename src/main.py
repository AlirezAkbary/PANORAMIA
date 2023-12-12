import logging
import os

import yaml
from easydict import EasyDict

from src.arguments import init_args
from src.datasets.datamodule import PANORAMIADataModule

def main(config: EasyDict):
    """
    runs the whole pipeline of PANORAMIA
    """
    ...



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
    os.makedirs(config.log_dir, exist_ok=False)
    logging.basicConfig(
        filename=os.path.join(
            config.log_dir,
            "output.log"
        ),
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    main(config)


    

    
    


    