import os
import contextlib
import tempfile
from caffe.proto import caffe_pb2
from google.protobuf import text_format, message
import caffe
from caffe import TRAIN, TEST


def read_prototxt(param, prototxt_file):
    with open(prototxt_file, 'r') as f:
        text_format.Merge(f.read(), param)


def from_prototxt(param_type, prototxt_file):
    param = param_type()
    read_prototxt(param, prototxt_file)
    return param


def write_prototxt(param, prototxt_file):
    with open(prototxt_file, 'w') as f:
        f.write(str(param))


@contextlib.contextmanager
def temp_prototxt(param):
    _, prototxt_file = tempfile.mkstemp()
    write_prototxt(param, prototxt_file)
    yield prototxt_file
    os.remove(prototxt_file)


def update_param(param_, *args, **kwargs):
    for key, value in kwargs.items():
        if isinstance(value, list):
            update_param(getattr(param_, key), *value)
        elif isinstance(value, dict):
            update_param(getattr(param_, key), **value)
        elif isinstance(value, message.Message):
            getattr(param_, key).CopyFrom(value)
        else:
            setattr(param_, key, value)
    for i, value in enumerate(args):
        if i == len(param_):
            try:
                param_.add()
            except AttributeError:
                param_.append(value)
                continue
        if isinstance(value, list):
            update_param(param_[i], *value)
        elif isinstance(value, dict):
            update_param(param_[i], **value)
        elif isinstance(value, message.Message):
            param_[i].CopyFrom(value)
        else:
            param_[i] = value


# can't inherit from protobuf message, so just add methods to the generated classes
for name, cls in caffe_pb2.__dict__.iteritems():
    if isinstance(cls, type) and issubclass(cls, message.Message):
        cls.from_prototxt = classmethod(from_prototxt)
        cls.to_prototxt = write_prototxt
        cls.temp_prototxt = temp_prototxt
        cls.update = update_param
        cls.__init__ = update_param
        globals()[name] = cls


class Net(caffe.Net):

    @classmethod
    def from_param(cls, net_param=None, phase=None, **kwargs):
        net_param = net_param or NetParameter()
        net_param.update(**kwargs)
        with net_param.temp_prototxt() as model_file:
            return cls(model_file, phase)


class Solver(caffe._caffe.Solver):

    @classmethod
    def from_param(cls, solver_param=None, **kwargs):
        solver_param = solver_param or SolverParameter()
        solver_param.update(**kwargs)
        with solver_param.temp_prototxt() as solver_file:
            return getattr(caffe, '{}Solver'.format(solver_param.type))(solver_file)
