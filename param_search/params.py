import re, ast
import itertools
from collections import OrderedDict

from .common import read_file, write_file, as_non_string_iterable


class Params(OrderedDict):
    '''
    Params represents a single point in a ParamSpace,
    a particular assignment of values to parameters.
    '''
    space = None

    @classmethod
    def from_file(cls, params_file):
        return cls(*read_params(params_file).item())


class ParamSpace(OrderedDict):
    '''
    ParamSpace defines ranges of possible values for a set
    of parameters. Iterating over the space produces elements
    from the Cartesian product of the parameter ranges.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Params = type('Params', (Params,), dict(space=self))

    @classmethod
    def from_file(cls, params_file):
        return cls(*read_params(params_file).items())

    def __setitem__(self, key, value):
        super().__setitem__(key, as_non_string_iterable(value))

    def __iter__(self):
        params = self.keys()
        for values in itertools.product(*self.values()):
            yield self.Params(zip(params, values))

    def __len__(self):
        values = self.values()
        if values:
            n = 1
            for value in values:
                n *= len(value)
            return n
        else:
            return 0


def parse_params(buf, line_start='', converter=ast.literal_eval):
    '''
    Parse lines in buf as param = value pairs, filtering by an
    optional line_start pattern. After parsing, a converter
    function is applied to param values.
    '''
    params = OrderedDict()
    line_pat = r'^{}(\S+)\s*=\s*(.+)$'.format(line_start)
    for p, v in re.findall(line_pat, buf, re.MULTILINE):
        params[p] = converter(v)
    return params


def format_params(params, line_start='', converter=repr):
    '''
    Serialize params as param = value lines with an optional
    line_start string. Before formatting, a converter function
    is applies to param values.
    '''
    lines = []
    for p, v in params.items():
        lines.append('{}{} = {}\n'.format(line_start, p, converter(v)))
    return ''.join(lines)


def read_params(params_file, line_start='', converter=ast.literal_eval):
    '''
    Read lines from params_file as param = value pairs.
    '''
    buf = read_file(params_file)
    return parse_params(buf, line_start, converter)


def write_params(params_file, params):
    '''
    Write params to params_file as param = value lines.
    '''
    buf = format_params(params)
    write_file(params_file, buf)
