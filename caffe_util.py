import sys, os
import contextlib
import tempfile
from caffe.proto import caffe_pb2
from google.protobuf import text_format, message
from google.protobuf.descriptor import FieldDescriptor
import caffe
from caffe.proto import caffe_pb2


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


def is_non_string_iterable(value):
    # repeated containers don't have an __iter__
    # attribute, but can still get an iterator
    try:
        return not isinstance(value, str) and iter(value)
    except TypeError:
        return False


def assign_index_param(param, index, value):

    if index > len(param):
        raise IndexError("can't assign to param index greater than length")

    if index == len(param): # append value
        try: # composite value
            param.add()
        except AttributeError:
            param.append(value)
            return

    if isinstance(value, message.Message):
        param[index].CopyFrom(value)

    elif isinstance(value, dict):
        update_composite_param(param[index], **value)

    elif isinstance(value, list):
        update_repeated_param(param[index], *value)

    else:
        param[index] = value


def update_repeated_param(param, *args):

    for i, value in enumerate(args):
        assign_index_param(param, i, value)


def assign_field_param(param, key, value):

    # allow assigning a single value to repeated fields
    if is_repeated_field(param, key):
        if isinstance(value, dict) or not is_non_string_iterable(value):
            value = [value]

    if isinstance(value, message.Message):
        getattr(param, key).CopyFrom(value)

    elif isinstance(value, dict):
        update_composite_param(getattr(param, key), **value)

    elif is_non_string_iterable(value):
        update_repeated_param(getattr(param, key), *value)

    else:
        setattr(param, key, value)


def update_composite_param(param, *args, **kwargs):

    for i, value in enumerate(args):
        key = get_field_name_by_index(param, i)
        if key in kwargs:
            raise TypeError(
                type(self).__name__ + ' got multiple values for ' + repr(key)
            )
        kwargs[key] = value

    for key, value in kwargs.items():
        assign_field_param(param, key, value)


# protobuf message fields use enums to specify their type
field_type_enum = {}  # enum_val -> type_name
for key, val in vars(FieldDescriptor).items():
    if key.startswith('TYPE_'):
        field_type_enum[val] = key


def is_optional_field(param, key):
    label = getattr(type(param), key).DESCRIPTOR.label
    return label == FieldDescriptor.LABEL_OPTIONAL


def is_repeated_field(param, key):
    label = getattr(type(param), key).DESCRIPTOR.label
    return label == FieldDescriptor.LABEL_REPEATED


def is_required_field(param, key):
    label = getattr(type(param), key).DESCRIPTOR.label
    return label == FieldDescriptor.LABEL_REQUIRED


def get_field_name_by_index(param, i):
    return param.DESCRIPTOR.fields[i].name


def get_message_docstring(msg):
    '''
    Create a nicely formatted docstring for a
    protocol buffer message specifying the
    label, type, and name of each field.
    '''
    max_len = 0
    for field in msg.fields:
        if field.message_type is None:
            field_type = field_type_enum[field.type]
        else:
            field_type = field.message_type.name
        if len(field_type) > max_len:
            max_len = len(field_type)

    lines = []
    for field in msg.fields:
        if field.label == field.LABEL_OPTIONAL:
            s = 'optional '
        elif field.label == field.LABEL_REPEATED:
            s = 'repeated '
        elif field.label == field.LABEL_REQUIRED:
            s = 'required '
        if field.message_type is None:
            s += field_type_enum[field.type].ljust(max_len)
        else:
            s += field.message_type.name.ljust(max_len)
        lines.append(s + ' ' + field.name)

    return '\n'.join(lines)


def get_param_repr(param):
    type_name, hex_id = type(param).__name__, hex(id(param))
    return '<{} object at {}>'.format(type_name, hex_id)


# we can't subclass protobuf messages,
# so add convenience methods to generated classes
# and import them to the module level
for msg_name in caffe_pb2.DESCRIPTOR.message_types_by_name:
    cls = getattr(caffe_pb2, msg_name)
    cls.__init__ = update_composite_param
    cls.__setitem__ = assign_field_param
    cls.from_prototxt = classmethod(from_prototxt)
    cls.to_prototxt = write_prototxt
    cls.temp_prototxt = temp_prototxt
    cls.__doc__ = get_message_docstring(cls.DESCRIPTOR)
    cls.__repr__ = get_param_repr
    globals()[msg_name] = cls


class CaffeNet(caffe.Net):

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
        for layer_name, param_blobs in self.params.items():
            for param_blob in param_blobs:
                n_params += param_blob.data.size
        return n_params

    def get_n_activs(self):
        n_activs = 0
        for blob_name, activ_blob in self.blobs.items():
            n_activs += activ_blob.data.size
        return n_activs

    def get_approx_size(self):
        return 2*(self.get_n_params() + self.get_n_activs())*4

    def get_min_width(self):
        min_width = float('inf')
        min_width_name = None
        for blob_name, activ_blob in self.blobs.items():
            if '_latent_mean' in blob_name:
                width = activ_blob.data.size // activ_blob.shape[0]
                if width < min_width:
                    min_width = width
                    min_width_name = blob_name
        return min_width


# problem: I want to extend Solver and all its subclasses
# (e.g. SGDSolver, AdamSolver) with the same functionality
# using a single CaffeSolver definition

# solution: dynamically change superclass of dynamically
# created, instance-specific CaffeSolver subclasses
# so that other CaffeSolver instances are not affected

class CaffeSolver(caffe._caffe.Solver):

    def __new__(cls, *args, **kwargs):
        class CaffeSolver(cls):
            pass
        return super().__new__(CaffeSolver)

    def __init__(
        self,
        param=None,
        state=None,
        scaffold=False,
        **kwargs,
    ):
        # can't overwrite Solver.param
        self.param_ = param or SolverParameter()

        for key, value in kwargs.items():
            assign_field_param(self.param_, key, value)

        if scaffold:
            self.scaffold(state)

    def scaffold(self, state=None, weights=None):

        solver_type = getattr(
            caffe, self.param_.type + 'Solver'
        )
        type(self).__bases__ += (solver_type,)

        with temp_prototxt(self.param_) as temp_file:
            super().__init__(temp_file)

        if state:
            self.restore(state)

        if weights:
            self.net.copy_From(weights)
