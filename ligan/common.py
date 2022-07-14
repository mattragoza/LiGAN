import time, random
import rdkit # import before molgrid to avoid RuntimeWarning
import numpy as np
import torch, molgrid


def get_unique_seed():
    # wait some fraction of a ms to avoid overlap
    time.sleep((time.time() % 1) / 1000)
    # return current time in ns mod the max allowed seed
    return int(time.time()*1e6) % 4294967295 # 2**32 - 1


def set_random_seed(random_seed=None):
    if random_seed is None:
        random_seed = get_unique_seed()
    print('Setting random seed to', random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    molgrid.set_random_seed(random_seed)


def catch_exception(func=None, exc_type=Exception, default=np.nan):
    '''
    Wrap a function in a try-except block and
    return a default value when a specific type
    of Exception is raised.

    Can be used in two different ways:

    1) Simple decorator
        wrapped = catch_exception(func, exc_type, default)

    2) Decorator factory (must use keywords args)
        @catch_exception(exc_type=..., default=...)
        def func(*args, **kwargs):
            ...
    '''
    if func is None: # use as a decorator factory
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except exc_type:            
                    return default
            return wrapper
        return decorator

    else: # use as a simple decorator
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exc_type:            
                return default
        return wrapper
