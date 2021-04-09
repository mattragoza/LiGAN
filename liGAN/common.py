import time
import numpy as np
import torch
import molgrid


def get_unique_seed():
    # wait some fraction of a ms to avoid overlap
    time.sleep((time.time() % 1) / 1000)
    # return current time in ns mod the max allowed seed
    return int(time.time()*1e6) % 4294967295 # 2**32 - 1


def set_random_seed(random_seed=None):
    if random_seed is None:
        random_seed = get_unique_seed()
    print('Setting random seed to', random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    molgrid.set_random_seed(random_seed)
