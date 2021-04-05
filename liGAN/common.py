import numpy as np
import torch
import molgrid


def set_random_seed(random_seed):
    numpy.random.seed(random_seed)
    torch.manual_seed(random_seed)
    molgrid.set_random_seed(random_seed)
