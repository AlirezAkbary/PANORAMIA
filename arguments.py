import argparse

def add_load_args(parser):
    parser.add_argument("--use_yml_config", action='store_true', default=False, help='load config from yaml')
    parser.add_argument("--path_yml_config", type=str, help='Path to experiment yaml config')
    return parser

def add_base_args(parser):
    parser.add_argument("--log_dir", type=str, help='Path to where the log file would be saved')
    return parser

def add_dataset_args(parser):
    #parser.add_argument("", type)
    return parser

def add_generator_args(parser):
    return parser

def init_args():
    parser = argparse.ArgumentParser()
    parser = add_load_args(parser)
    parser = add_base_args(parser)
    parser = add_dataset_args(parser)
    parser = add_generator_args(parser)
    args = parser.parse_args()
    return args


def args_to_attr_dict(args):
    """
    convert the args.Namespace object into a nested dictionary
    """
    ...