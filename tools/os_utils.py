import json
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


def read_prompt_file_to_list(prompt_file):
    # if is json
    if prompt_file.endswith(".json"):
        prompt_list = json.load(open(prompt_file, "r"))
    elif prompt_file.endswith(".txt"):
        with open(prompt_file, "r") as f:
            prompt_list = f.readlines()
    return prompt_list
