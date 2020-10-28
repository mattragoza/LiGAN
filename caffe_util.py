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
    '''
    A node in a Caffe computation graph, which
    can either be a CaffeBlob or a CaffeLayer.
    '''
    # TODO
    # - allow lazy evaluation of net using graph
    # - allow multiple top-most blobs
    # - allow in-place layer calls
    # - infer n_tops from layer type
    # - tuplify blob shapes where possible
    # - more readable automatic node names

    def __init__(self, name=None):
        self.name = name or hex(id(self))
        self.bottoms = []
        self.tops = []
        self.net = None

    def add_bottom(self, bottom):
        self.bottoms.append(bottom)

    def add_top(self, top):
        self.tops.append(top)

    def replace_bottom(self, old, new):
        for i, bottom in enumerate(self.bottoms):
            if bottom is old:
                break
        assert bottom is old
        self.bottoms[i] = new

    def set_net(self, net):
        self.net = net

    def has_scaffold(self):
        return self.net and self.net.has_scaffold()

    def find_in_net(self):
        pass

    def scaffold(self):
        net = CaffeNet(scaffold=False)
        self.add_to_net(net) # TODO traverse graph
        net.scaffold()
        self.find_in_net() # TODO traverse graph
        return net


class CaffeBlob(CaffeNode):
    '''
    A caffe blob of data and associated gradient, which
    can be the input and/or output of a CaffeLayer.
    '''
    def __init__(self, name=None):
        super().__init__(name=name)

    def add_bottom(self, bottom):
        assert isinstance(bottom, CaffeLayer)
        super().add_bottom(bottom)

    def add_top(self, top):
        assert isinstance(top, CaffeLayer)
        super().add_top(top)

    def find_in_net(self):
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
    '''
    A caffe function that is applied to some number of
    CaffeBlobs and produces CaffeBlobs as output.
    '''
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

        self.loss_weight = None

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
            blobs[b].add_top(layer)

        for t in param.top:
            if t not in blobs:
                blobs[t] = CaffeBlob(t)
            layer.add_top(blobs[t])
            blobs[t].add_bottom(layer)

        if param.loss_weight:
            layer.loss_weight = param.loss_weight[0]

        return layer

    def __call__(self, *args, name=None, loss_weight=0):
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

        if loss_weight:
            layer.loss_weight = loss_weight

        # apply the layer to the provided bottom blobs
        for bottom in args:
            layer.add_bottom(bottom)

        # create and return top blobs based on n_tops
        if layer.n_tops == 0:
            return

        elif layer.n_tops == 1:
            top = CaffeBlob()
            top.add_bottom(layer)
            layer.add_top(top)
            return top

        else:
            for i in range(layer.n_tops):
                top = CaffeBlob()
                top.add_bottom(layer)
                layer.add_top(top)

            return layer.tops

    def add_bottom(self, bottom):
        assert isinstance(bottom, CaffeBlob)
        super().add_bottom(bottom)

    def add_top(self, top):
        assert isinstance(top, CaffeBlob)
        super().add_top(top)

    def is_in_place(self):
        return len(set(self.bottoms + self.tops)) == 1

    def find_in_net(self):
        self.blobs = self.net.layer_dict[self.name].blobs

    def blobs(self):
        return list(self.blobs)

    def __ror__(self, other):
        '''
        Can use "|" to pipe CaffeLayers together.
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
        param = param or NetParameter()
        update_composite_param(param, **kwargs)

        # can't overwrite Net.layers or Net.blobs
        self.layers_ = OrderedDict()
        self.blobs_ = OrderedDict()

        # construct graph of net from layer params
        for layer_param in param.layer:
            layer = CaffeLayer.from_param(layer_param, self.blobs_)
            self.add_layer(layer)
            layer.set_net(self)
            for top in layer.tops:
                top.set_net(self)

        if scaffold:
            self.scaffold(weights, phase)

    @classmethod
    def from_prototxt(cls, model_file):
        return cls(param=NetParameter.from_prototxt(model_file))

    def add_layer(self, layer):
        assert isinstance(layer, CaffeLayer)
        assert layer.name not in self.layers_
        self.layers_[layer.name] = layer

    def get_param(self):
        '''
        Return a NetParameter from the CaffeLayer list.
        '''
        param = NetParameter()
        for layer in self.layers_.values():

            layer_param = LayerParameter()
            layer_param.name = layer.name
            layer_param.type = type(layer).__name__

            for bottom in layer.bottoms:
                layer_param.bottom.append(bottom.name)

            for top in layer.tops:
                layer_param.top.append(top.name)

            if layer.loss_weight is not None:
                layer_param.loss_weight.append(layer.loss_weight)

            if layer.param_type and layer.param.ListFields():
                assign_field_param(
                    layer_param, layer.param_name, layer.param
                )

            param.layer.append(layer_param)

        return param

    def scaffold(self, weights=None, train=False):
        '''
        Initialize the underlying layers and blobs
        in caffe and link them to the graph. This
        method inserts split layers as needed.
        '''
        # get NetParameter from graph
        param = self.get_param()
        phase = caffe.TRAIN if train else caffe.TEST

        # initalize Net superclass from NetParameter
        with temp_prototxt(param) as temp_file:
            super().__init__(
                network_file=temp_file,
                weights=weights,
                phase=phase,
            )

        # insert split layers into graph
        self.insert_splits()

        print(self.get_param())

        # link graph to Net scaffold
        for layer in self.layers_.values():
            layer.find_in_net()

        for blob in self.blobs_.values():
            blob.find_in_net()

    def insert_splits(self):
        '''
        Insert split layers after blobs with more than
        one top layer that are not applied in-place.
        '''
        for blob in list(self.blobs_.values()):

            in_place_tops = []
            not_in_place_tops = []
            for top in blob.tops:
                if top.is_in_place():
                    in_place_tops.append(top)
                else:
                    not_in_place_tops.append(top)

            if len(not_in_place_tops) > 1: # insert split

                blob.tops = in_place_tops
                split_name = '{}_{}_0_split'.format(blob.name, blob.name)
                split = Split(n_tops=len(not_in_place_tops))
                split_blobs = split(blob, name=split_name)

                for i, split_blob in enumerate(split_blobs):
                    split_blob.name = split_name + '_' + str(i)
                    not_in_place_tops[i].replace_bottom(blob, split_blob)
                    split_blob.add_top(not_in_place_tops[i])

                split = split_blob.bottoms[0]
                self.add_layer(split)
                split.set_net(self)
                for top in split.tops:
                    self.blobs_[top.name] = top
                    top.set_net(self)

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
