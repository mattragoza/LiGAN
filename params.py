import re, ast
import itertools
import collections


def non_string_iterable(obj):
    '''
    Check whether obj is a non-string iterable.
    '''
    iterable = isinstance(obj, collections.Iterable)
    string = isinstance(obj, str)
    return not string and iterable


def as_non_string_iterable(obj):
    '''
    Return obj in a list if it's a string or not iterable.
    '''
    return obj if non_string_iterable(obj) else [obj]


def read_file(file_):
    with open(file_, 'r') as f:
        return f.read()


def write_file(file_, buf):
    with open(file_, 'w') as f:
        f.write(buf)


def parse_params(buf, line_start='', converter=ast.literal_eval):
    '''
    Parse lines in buf as param = value pairs, filtering by an
    optional line_start pattern. After parsing, a converter
    function is applied to param values.
    '''
    params = collections.OrderedDict()
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


class Params(collections.OrderedDict):
    space = None

    def __str__(self):
        return self.name

    def __repr__(self):
        return repr(str(self))

    def flatten(self, scope='', sep='.'):
        param = re.sub(r'_params$', '_name', scope)
        items = [(param, str(self))]
        for param, value in self.items():
            if scope:
                param = scope + sep + param
            if isinstance(value, Params):
                items.extend(value.flatten(param).items())
            else:
                items.append((param, value))
        name = self.name
        self = Params(items)
        self.name = name
        return self


class ParamSpace(object):

    def __init__(self, arg=None, format=None, **kw_dims):

        self.dims = collections.OrderedDict()
        self.format = format

        if arg is not None:

            if isinstance(arg, ParamSpace):
                arg_dims = arg.dims
                if self.format is None:
                    self.format = arg.format
            elif isinstance(arg, str):
                arg_dims = read_params(arg)
            else:
                arg_dims = collections.OrderedDict(arg)

            for p, v in arg_dims.items():
                self.dims[p] = as_non_string_iterable(v)

        for p, v in kw_dims.items():
            self.dims[p] = as_non_string_iterable(v)

        class _Params(Params):
            space = self
            def __init__(self, *args, **kwargs):
                Params.__init__(self, *args, **kwargs)
                self.name = self.space.format_params(self)

        self.Params = _Params

    def format_params(self, params):
        if self.format is None:
            format = '_'.join('{' + p + '}' for p, v in self.dims.items() if len(v) > 1).format
        else:
            format = self.format
        return format(**params)

    def __repr__(self):
        return repr(self.dims).replace('OrderedDict', 'ParamSpace')

    def __getitem__(self, param):
        return self.dims[param]

    def __setitem__(self, param, value):
        self.dims[param] = as_non_string_iterable(value)

    def __iter__(self):
        params = self.dims.keys()
        for values in itertools.product(*self.dims.values()):
            yield self.Params(zip(params, values))

    def __len__(self):
        values = self.dims.values()
        if values:
            n = 1
            for v in values:
                n *= len(v)
            return n
        else:
            return 0

    def flatten(self, scope='', sep='.'):
        for params in self:
            yield params.flatten(scope, sep)
