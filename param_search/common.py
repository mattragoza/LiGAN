import os
from collections import Iterable


def read_file(file_):
    with open(file_, 'r') as f:
        return f.read()


def write_file(file_, buf):
    with open(file_, 'w') as f:
        f.write(buf)


def get_terminal_size():
    with os.popen('stty size') as p:
        return map(int, p.read().split())


def non_string_iterable(obj):
    '''
    Check whether obj is a non-string iterable.
    '''
    return not isinstance(obj, str) and isinstance(obj, Iterable)


def as_non_string_iterable(obj):
    '''
    Put obj in a list if it's a string or not iterable.
    '''
    return obj if non_string_iterable(obj) else [obj]
