import os


def read_file(file_):
    with open(file_, 'r') as f:
        return f.read()


def write_file(file_, buf):
    with open(file_, 'w') as f:
        f.write(buf)


def get_terminal_size():
    with os.popen('stty size') as p:
        return map(int, p.read().split())
