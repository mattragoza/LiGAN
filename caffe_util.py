import os, copy
import contextlib
import tempfile
from collections import namedtuple, OrderedDict
from caffe.proto import caffe_pb2
from google.protobuf import text_format, message
from google.protobuf.descriptor import FieldDescriptor
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


def assign_index_param(param, index, value):

    if index == len(param): # append

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

    if isinstance(value, message.Message):
        getattr(param, key).CopyFrom(value)

    elif isinstance(value, dict):
        update_composite_param(getattr(param, key), **value)

    elif hasattr(value, '__len__'):
        update_repeated_param(getattr(param, key), *value)

    else:
        setattr(param, key, value)


def update_composite_param(param, **kwargs):

    for key, value in kwargs.items():
        try:
            assign_field_param(param, key, value)
        except AttributeError:
            print(type(param), repr(key), type(value))
            raise


# protobuf message fields use an enum to specify their type
# we don't have access to it, so recreate it with this hack
field_type_enum = {}
for type_name, i in vars(FieldDescriptor).items():
    if type_name.startswith('TYPE_'):
        field_type_enum[i] = type_name


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
    globals()[msg_name] = cls


class CaffeNode(object):

    # TODO
    # - allow lazy evaluation of net using graph
    # - allow multiple top-most blobs
    # - handle split layers somehow
    # - allow in-place layer calls
    # - allow pythonic usage/magic methods
    # - infer n_tops from layer type
    # - tuplify blob shapes where possible
    # - more readable node names

    def __init__(self, name=None):
        self.name = name or hex(id(self))
        self.bottoms = []
        self.tops = []
        self.net = None

    def has_scaffold(self):
        return self.net and self.net.has_scaffold()

    def add_to_net(self, net):
        for bottom in self.bottoms:
            bottom.add_to_net(net)
        self.net = net

    def find_in_net(self):
        for bottom in self.bottoms:
            bottom.find_in_net()

    def scaffold(self):
        net = CaffeNet(scaffold=False)
        self.add_to_net(net)
        net.scaffold()
        self.find_in_net()
        return net


class CaffeBlob(CaffeNode):

    def __init__(self, name=None):
        super().__init__(name=name)

    def find_in_net(self):
        super().find_in_net()
        self.blob = self.net.blobs[self.name]

    def shape(self):
        return tuple(self.blob.shape)

    def data(self):
        return self.blob.data

    def diff(self):
        return self.blob.diff

    def __add__(self, other):
        return Eltwise(Eltwise.param_type.SUM)(self, other)

    def __sub__(self, other):
        return Eltwise(Eltwise.param_type.SUM, coeff=[1, -1])(self, other)

    def __mul__(self, other):
        return Eltwise(Eltwise.param_type.PROD)(self, other)

    def sum(self, axis=0):
        return Reduction(Reduction.param_type.SUM, axis=axis)(self)


class CaffeLayer(CaffeNode):

    # the protobuf message type that stores the params
    # that control what function the layer applies
    param_type = NotImplemented

    # the field name in LayerParameter that stores a
    # protobuf message of the above type (param_type)
    param_name = NotImplemented

    def __init__(self, *args, **kwargs):
        '''
        Create a layer prototype where the params specific
        to the layer type (e.g. convolution_param) are set
        from args and kwargs.
        '''
        super().__init__()

        # TODO determine n_tops from layer params
        # how many blobs to produce from __call__
        self.n_tops = kwargs.pop('n_tops', 1)

        if self.param_type:
            self.param = self.param_type()

            for i, val in enumerate(args):
                key = self.param.DESCRIPTOR.fields[i].name
                if key in kwargs:
                    raise TypeError(
                        type(self).__name__ + 
                        ' got multiple values for param ' + 
                        key
                    )
                kwargs[key] = val

            update_composite_param(self.param, **kwargs)

        elif args or kwargs:
            raise TypeError(
                type(self).__name__ + ' has no params'
            )

    @classmethod
    def from_param(cls, param, blobs=None):

        assert isinstance(param, LayerParameter)
        if blobs is None:
            blobs = {}

        # get the subclass from layer type
        cls = globals()[param.type]

        # create the layer prototype
        layer = cls(**dict(
            (k.name, v) for k,v in getattr(
                param, cls.param_name
            ).ListFields()
        ))

        # if present, set name, bottoms, and tops
        if param.name:
            layer.name = param.name

        for b in param.bottom:
            if b not in blobs:
                blobs[b] = CaffeBlob(b)
            layer.add_bottom(blobs[b])

        for t in param.top:
            if t not in blobs:
                blobs[t] = CaffeBlob(t)
            layer.add_top(blobs[t])

        return layer

    def __call__(self, *args, name=None):
        '''
        Calling a CaffeLayer on input CaffeBlobs
        returns output CaffeBlobs that result from
        applying the function defined by the layer
        to the input blobs.
        '''
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            args = tuple(args[0])

        assert all(isinstance(a, CaffeBlob) for a in args)

        # copy the layer prototype
        layer_type = type(self)
        layer = layer_type(n_tops=self.n_tops)

        if self.param_type:
            layer.param.CopyFrom(self.param)

        if name:
            layer.name = name

        # apply the layer to the provided bottom blobs
        for bottom in args:
            layer.add_bottom(bottom)

        # create and return top blobs based on n_tops
        if layer.n_tops == 0:
            return

        elif layer.n_tops == 1:
            layer.add_top(CaffeBlob())
            return layer.tops[0]

        else:
            for i in range(layer.n_tops):
                layer.add_top(CaffeBlob())
            return layer.tops

    def add_bottom(self, bottom):
        assert isinstance(bottom, CaffeBlob)
        self.bottoms.append(bottom)
        bottom.tops.append(self)

    def add_top(self, top):
        assert isinstance(top, CaffeBlob)
        self.tops.append(top)
        top.bottoms.append(self)

    def add_to_net(self, net):
        super().add_to_net(net)
        net.add_layer(self)

    def find_in_net(self):
        super().find_in_net()
        self.blobs = self.net.layer_dict[self.name].blobs

    def blobs(self):
        return list(self.blobs)

    def __ror__(self, other):
        '''
        Can use "|" to pipe CaffeLayer together.
        '''
        return self.__call__(other)

    def __str__(self):
        param_repr = '\n' + repr(self.param).rstrip()
        param_repr = param_repr.replace('\n', '\n  ') + '\n'
        return type(self).__name__ + '(' + param_repr + ')'

    @classmethod
    def make_subclass(cls, layer_name):
        '''
        Dynamically subclass CaffeLayer by using
        the layer name to look up which param
        fields to fill out in layer parameter.
        '''
        # get the param type that stores the layer params
        param_type = param_type_map.get(layer_name, None)
        param_name = param_name_map.get(param_type, None)
        return type(
            layer_name,
            (cls,),
            dict(
                param_name=param_name,
                param_type=param_type,
            )
        )


# map from param types to param names (in layer param)
param_name_map = {}
for field in LayerParameter.DESCRIPTOR.fields:
    if field.name.endswith('_param'):
        param_type_name = getattr(
            LayerParameter, field.name
        ).DESCRIPTOR.message_type.name
        param_type = globals()[param_type_name]
        param_name_map[param_type] = field.name


# map from layer names to param types
param_type_map = {}
for layer_name in caffe.layer_type_list():
    param_type_name = layer_name + 'Parameter'
    if param_type_name in globals():
        param_type_map[layer_name] = globals()[param_type_name]


# special cases
param_type_map['Deconvolution'] = ConvolutionParameter
param_type_map['EuclideanLoss'] = LossParameter
param_type_map['SigmoidCrossEntropyLoss'] = LossParameter
param_type_map['MultinomialLogisticLoss'] = LossParameter
param_type_map['SoftmaxWithLoss'] = SoftmaxParameter
param_type_map['SoftmaxWithNoisyLabelLoss'] = SoftmaxParameter
param_type_map['RNN'] = RecurrentParameter
param_type_map['LSTM'] = RecurrentParameter


# dynamically subclass CaffeLayer for each caffe layer type
for layer_name in caffe.layer_type_list():
    globals()[layer_name] = CaffeLayer.make_subclass(layer_name)


class CaffeNet(caffe.Net):

    def __init__(
        self, 
        param=None,
        weights=None,
        phase=caffe.TEST,
        scaffold=False,
        **kwargs
    ):
        self.param = param or NetParameter()

        for key, value in kwargs.items():
            caffe.net_spec.assign_proto(self.param, key, value)

        # can't overwrite Net.layers or Net.blobs
        self.layers_ = OrderedDict()
        self.blobs_ = OrderedDict()

        for layer_param in self.param.layer:
            layer = CaffeLayer.from_param(layer_param, self.blobs_)
            self.layers_[layer.name] = layer

        if scaffold:
            self.scaffold(weights, phase)

    @classmethod
    def from_prototxt(cls, model_file):
        return cls(param=NetParameter.from_prototxt(model_file))

    def add_layer(self, layer):
        assert isinstance(layer, CaffeLayer)
        if layer.name not in self.layers:
            param = LayerParameter()
            param.name = layer.name
            param.bottom.extend([b.name for b in layer.bottoms])
            param.top.extend([t.name for t in layer.tops])
            if layer.param_type:
                assign_field_param(param, layer.param_name, layer.param)
            self.param.layer.append(param)
            self.layers_[layer.name] = layer

    def scaffold(self, weights=None, phase=caffe.TEST):

        with temp_prototxt(self.param) as temp_file:
            super().__init__(
                network_file=temp_file,
                weights=weights,
                phase=phase,
            )

        for b, blob in list(self.blobs_.items()):
            n_tops = len(blob.tops)

            if n_tops > 1: # insert split layer

                prev_tops, blob.tops = blob.tops, []
                split_name = '{}_{}_0_split'.format(
                       blob.name, blob.name
                )
                split_blobs = Split(n_tops=n_tops)(
                    blob, name=split_name
                )
                assert len(split_blobs) == n_tops

                for i, split_blob in enumerate(split_blobs):

                    split_blob.name = split_name + '_' + str(i)
                    top = prev_tops[i]

                    for j, bottom in enumerate(top.bottoms):
                        if bottom is blob:
                            break
                    assert bottom is blob

                    top.bottoms[j] = split_blob
                    split_blob.tops.append(top)
                    self.blobs_[split_blob.name] = split_blob

                self.add_layer(split_blob.bottoms[0])

    def has_scaffold(self):
        try:
            self.blobs
            return True
        except:
            return False

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
# using a single CaffeSolver 

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
        **kwargs
    ):
        # can't overwrite Solver.param
        self.param_ = param or SolverParameter()

        for key, value in kwargs.items():
            caffe.net_spec.assign_proto(self.param_, key, value)

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
            self.net.copy_from(weights)


class CaffeSubNet(object):

    def __init__(self, net, start, end):
        self.net = net
        self.start = start
        self.end = end

    def forward(self, input=None):
        if input is not None:
            self.net.blobs[self.start].data[...] = input
        self.net.forward(start=self.start, end=self.end)

    def backward(self, gradient=None):
        if gradient is not None:
            self.net.blobs[self.end].diff[...] = gradient
        self.net.backward(start=self.start, end=self.end)
