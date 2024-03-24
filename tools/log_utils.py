import builtins
import os
import warnings
from abc import abstractmethod

import wandb

from .common_utils import instantiate_from_config, is_main_process


def print(*args, **kwargs):
    """
    Print function that only prints from rank 0.
    """
    if is_main_process():
        builtins.print(*args, **kwargs)


class Base_log(object):
    name = "base_log"
    turn = False
    enable_flag = "enable_base_log"

    def __init__(self, config, **kwargs):
        if is_main_process():
            if self.enable_flag in config:
                flag = config[self.enable_flag]
                if flag is not None:
                    if isinstance(flag, bool):
                        self.turn = flag
                    else:
                        warnings.warn(f"{self.name} flag is not a bool")

            if self.turn == True:
                print("The experiment enable {}".format(self.name))
                self.set_logger(config, **kwargs)

    @abstractmethod
    def set_logger(self, config, **kwargs):
        pass

    def log(self, **kwargs):
        if is_main_process():
            if self.turn == True:
                self.update(**kwargs)

    @abstractmethod
    def update(self, **kwargs):
        raise NotImplementedError


class Tensorboard_log(Base_log):
    name = "tensorboard_log"
    enable_flag = "enable_tensorboard_log"
    turn = True

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def set_logger(self, config, path, **kwargs):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(path)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.writer.add_scalar(k, v)


class Wandb_log(Base_log):
    name = "wandb_log"
    enable_flag = "enable_wandb_log"
    turn = False

    def __init__(self, config, **kwargs):
        super(Wandb_log, self).__init__(config)

    def set_logger(self, config, **kwargs):
        # os.environ['WANDB_API_KEY']= config.wandb.api_key
        if "wandb_api_key" in config and config.wandb_api_key is not None:
            os.environ["WANDB_API_KEY"] = config.wandb_api_key
        instantiate_from_config(**config.wandb)

    def update(self, **kwargs):
        wandb.log(kwargs)
