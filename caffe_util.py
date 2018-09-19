import os
import contextlib
import tempfile
from caffe.proto import caffe_pb2
from google.protobuf import text_format, message
import caffe


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


def set_molgrid_data_source(net_param, data_file, data_root, phase=None):
    for layer_param in net_param.layer:
        if layer_param.type == 'MolGridData':
            data_param = layer_param.molgrid_data_param
            if phase is None:
                data_param.source = data_file
                data_param.root_folder = data_root
            elif layer_param.include[0].phase == phase:
                data_param.source = data_file
                data_param.root_folder = data_root


def get_molgrid_data_param(net_param, phase=None):
    for layer_param in net_param.layer:
        if layer_param.type == 'MolGridData':
            data_param = layer_param.molgrid_data_param
            if phase is None:
                return data_param
            elif layer_param.include[0].phase == phase:
                return data_param


# can't inherit from protobuf message, so just add methods to the generated classes
for name, cls in caffe_pb2.__dict__.iteritems():
    if isinstance(cls, type) and issubclass(cls, message.Message):
        cls.from_prototxt = classmethod(from_prototxt)
        cls.to_prototxt = write_prototxt
        cls.temp_prototxt = temp_prototxt
        cls.update = update_param
        cls.__init__ = update_param
        globals()[name] = cls
        if issubclass(cls, caffe_pb2.NetParameter):
            cls.set_molgrid_data_source = set_molgrid_data_source
            cls.get_molgrid_data_param = get_molgrid_data_param


class Net(caffe.Net):

    @classmethod
    def from_param(cls, net_param=None, weights_file=None, phase=-1, **kwargs):
        net_param = net_param or NetParameter()
        net_param.update(**kwargs)
        with net_param.temp_prototxt() as model_file:
            return cls(network_file=model_file, weights=weights_file, phase=phase)

    @classmethod
    def from_spec(cls, net_spec, *args, **kwargs):
        return Net.from_param(net_spec.to_proto(), *args, **kwargs)

    def get_n_params(self):
        n_params = 0
        for key, param_blobs in self.params.iteritems():
            for param_blob in param_blobs:
                n_params += param_blob.data.size
        return n_params

    def get_n_data(self):
        n_data = 0
        for key, data_blob in self.blobs.iteritems():
            n_data += data_blob.data.size
        return n_data

    def get_size(self):
        return 2*(self.get_n_params() + self.get_n_data())*4


class Solver(caffe._caffe.Solver):

    @classmethod
    def from_param(cls, solver_param=None, **kwargs):
        solver_param = solver_param or SolverParameter()
        solver_param.update(**kwargs)
        with solver_param.temp_prototxt() as solver_file:
            return getattr(caffe, '{}Solver'.format(solver_param.type))(solver_file)
