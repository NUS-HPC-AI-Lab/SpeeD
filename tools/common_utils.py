import random

import numpy as np
import torch


def seed_all(random_seed):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True


import torch.distributed as dist


# check if main rank
def is_main_process():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    else:
        return True


import hydra


def instantiate_from_config(**kwargs):
    return hydra.utils.instantiate(kwargs)
