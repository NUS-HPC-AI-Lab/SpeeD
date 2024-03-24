import os
import warnings

from omegaconf import OmegaConf

from .common_utils import is_main_process


def init_working_space(path):
    # check exist
    if is_main_process():
        if not os.path.exists(path):
            print("\033[31m create working directory: {} \033[0m".format(path))
            os.makedirs(path)
        else:
            # check empty
            if len(os.listdir(path)) == 0:
                print("\033[31m working directory: {} \033[0m".format(path))
            else:
                warnings.warn("\033[31m working directory: {} is not empty \033[0m".format(path))


def save_config(config, path):
    if is_main_process():
        config_name = "config.yaml"
        with open(os.path.join(path, config_name), "w") as f:
            OmegaConf.save(config, f)
            print("\033[31m save config to {} \033[0m".format(os.path.join(path, config_name)))
