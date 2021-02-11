import sys, os
import contextlib
import tempfile
from collections import namedtuple, OrderedDict
from google.protobuf import text_format, message
from google.protobuf.descriptor import FieldDescriptor
import caffe
from caffe.proto import caffe_pb2
from caffe.draw import draw_net_to_file


def read_prototxt(param, prototxt_file):
    with open(prototxt_file, 'r') as f:
        text_format.Merge(f.read(), param)


def from_prototxt_str(param_type, prototxt_str):
    param = param_type()
    text_format.Merge(prototxt_str, param)
    return param

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


# create a mapping from protobuf message types
# that store layer type-specific parameters
# to the LayerParameter field name of that type
# e.g.
#   LayerParameter.input_param stores an InputParameter,
#   so field_name_map[InputParameter] == 'input_param'
field_name_map = {}
for field in LayerParameter.DESCRIPTOR.fields:
    if field.name.endswith('_param'):
        param_type_name = getattr(
            LayerParameter, field.name
        ).DESCRIPTOR.message_type.name
        param_type = globals()[param_type_name]
        field_name_map[param_type] = field.name


# create a mapping from layer type names
# to protobuf message types that store
# parameters specific to that layer type
# e.g.
#   Input layers use InputParameter
#   so param_type_map['Input'] = InputParameter
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


class CaffeNode(object):
    '''
    A node in a Caffe computation graph, which
    can either be a CaffeBlob or a CaffeLayer.
    '''
    # TODO
    # - force explicit split layers by not
    #   allowing blobs to have > 1 top or bottom
    # - lazy/partial evaluation of graph
    # - create param through graph traversal
    # - infer n_tops from layer type
    # - tuplify blob shapes where possible
    # - more readable automatic node names

    def __init__(self, name=None):
        self.name = name or hex(id(self))
        self.bottoms = []
        self.tops = []

    def add_bottom(self, bottom):
        self.bottoms.append(bottom)

    def add_top(self, top):
        self.tops.append(top)

    def replace_bottom(self, old, new):
        self.bottoms[self.bottoms.index(old)] = new

    def replace_top(self, old, new):
        self.tops[self.tops.index(old)] = new

    def add_to_net(self, net):
        raise NotImplementedError

    def find_in_net(self, net):
        raise NotImplementedError

    def has_scaffold(self):
        raise NotImplementedError

    def to_param(self):
        raise NotImplementedError

    def apply(self, func, down=False, up=False, visited=None):
        '''
        Recursively apply func to bottom nodes, self,
        and top nodes, in that order, depending on
        whether down and/or up flags are set.
        '''
        if visited is None:
            visited = set()
        visited.add(self)

        if down: # apply recursively to bottoms
            for bottom in self.bottoms:
                if bottom not in visited:
                    bottom.apply(func, down, up, visited)
        func(self)

        if up: # apply recursively to tops
            for top in self.tops:
                if top not in visited:
                    top.apply(func, down, up, visited)

    def scaffold(self, *args, **kwargs):
        '''
        Traverse the graph creating a NetParameter,
        then create a CaffeNet from the NetParameter,
        then find the graph nodes in the CaffeNet.
        '''
        net = CaffeNet(force_backward=True)
        self.apply(net.add_node, down=True, up=True)
        net.scaffold(*args, **kwargs)
        self.apply(net.find_node, down=True, up=True)
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

    def add_to_net(self, net):
        pass

    def find_in_net(self, net):
        self.net = net
        self.blob = net.blobs[self.name]

    def has_scaffold(self):
        return hasattr(self, 'blob')

    @property
    def shape(self):
        return tuple(self.blob.shape)

    @property
    def data(self):
        return self.blob.data

    @property
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

    def reshape(self, shape):
        return Reshape(shape=dict(dim=list(shape)))(self)


class CaffeLayer(CaffeNode):
    '''
    A caffe function that is applied to some number of
    CaffeBlobs and produces CaffeBlobs as output.
    '''
    # the protobuf message type that stores the params
    # that control what function the layer applies
    param_type = NotImplemented

    # the field name in LayerParameter that stores a
    # protobuf message of the above type
    field_name = NotImplemented

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
        self.in_place = kwargs.pop('in_place', False)
        assert not self.in_place or self.n_tops == 1

        self.param = None
        if self.param_type:

            self.param = self.param_type()
            update_composite_param(self.param, *args, **kwargs)

        elif args or kwargs:
            raise TypeError(
                type(self).__name__ + ' has no params'
            )

        self.loss_weight = None
        self.lr_mult = None
        self.decay_mult = None

    @classmethod
    def from_param(cls, param, blobs=None):
        '''
        Construct a CaffeLayer from a LayerParameter,
        optionally checking/updating the provided
        blobs dict before/after creating CaffeBlobs.
        '''
        assert isinstance(param, LayerParameter)
        if blobs is None:
            blobs = {}

        # get the subclass from layer type
        cls = globals()[param.type]

        # create the layer prototype
        layer = cls(**dict(
            (k.name, v) for k,v in getattr(
                param, cls.field_name
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

        if param.param:
            layer.lr_mult = param.param[0].lr_mult
            layer.decay_mult = param.param[0].decay_mult

        return layer

    def to_param(self):
        '''
        Return the CaffeLayer as a LayerParameter.
        '''
        param = LayerParameter()
        param.name = self.name
        param.type = type(self).__name__

        for bottom in self.bottoms:
            param.bottom.append(bottom.name)

        for top in self.tops:
            param.top.append(top.name)

        if self.loss_weight is not None:
            param.loss_weight.append(self.loss_weight)

        if self.lr_mult is not None:
            param.param.add()
            param.param[0].lr_mult = self.lr_mult
            param.param[0].decay_mult = self.decay_mult

        if self.param_type and self.param.ListFields():
            assign_field_param(
                param, self.field_name, self.param
            )

        return param

    def __call__(self, *args, name=None, loss_weight=0):
        '''
        Calling a CaffeLayer on input CaffeBlobs
        returns output CaffeBlobs that result from
        applying the function defined by the layer
        to the input blobs.
        '''
        if len(args) == 1 and is_non_string_iterable(args[0]):
            args = tuple(args[0])

        assert not self.in_place or (len(args) == 1 and self.n_tops == 1)
        assert all(isinstance(a, CaffeBlob) for a in args)

        if name:
            self.name = name

        # apply the layer to the provided bottom blobs
        for bottom in args:
            self.add_bottom(bottom)
            bottom.add_top(self)

        # create and return top blobs based on n_tops
        if self.n_tops == 0:
            return

        elif self.n_tops == 1:
            top = bottom if self.in_place else CaffeBlob()
            top.add_bottom(self)
            self.add_top(top)
            return top

        else:
            for i in range(self.n_tops):
                top = CaffeBlob()
                top.add_bottom(self)
                self.add_top(top)

            return self.tops

    def add_bottom(self, bottom):
        assert isinstance(bottom, CaffeBlob)
        super().add_bottom(bottom)

    def add_top(self, top):
        assert isinstance(top, CaffeBlob)
        super().add_top(top)

    def is_in_place(self):
        return len(set(self.bottoms + self.tops)) == 1

    def add_to_net(self, net):
        net.add_layer(self)

    def find_in_net(self, net):
        self.net = net
        self.layer = net.layer_dict[self.name]

    def has_scaffold(self):
        return hasattr(self, 'layer')

    @property
    def blobs(self):
        return list(self.layer.blobs)

    def forward(self):
        self.net.forward(start=self.name, end=self.name)

    def backward(self):
        self.net.backward(start=self.name, end=self.name)

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
        field_name = field_name_map.get(param_type, None)
        return type(
            layer_name,
            (cls,),
            dict(
                field_name=field_name,
                param_type=param_type,
            )
        )


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

        self.force_backward = param.force_backward

        if scaffold:
            self.scaffold(weights, phase)

    @classmethod
    def from_prototxt(cls, model_file):
        return cls(param=NetParameter.from_prototxt(model_file))

    def add_node(self, node):
        assert isinstance(node, CaffeNode)
        node.add_to_net(self)

    def add_layer(self, layer):
        assert isinstance(layer, CaffeLayer)
        self.layers_[layer.name] = layer

    def find_node(self, node):
        assert isinstance(node, CaffeNode)
        node.find_in_net(self)

    def to_param(self):
        '''
        Convert the CaffeNet to a NetParameter.
        '''
        param = NetParameter()
        param.force_backward = self.force_backward

        for layer in self.layers_.values():
            layer_param = layer.to_param()
            param.layer.append(layer_param)

        return param

    def scaffold(self, weights=None, train=False):
        '''
        Initialize the underlying layers and blobs
        in caffe and link them to the graph. This
        method inserts split layers as needed.
        '''
        # get NetParameter from graph
        param = self.to_param()
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

        # link graph to Net scaffold
        for layer in self.layers_.values():
            layer.find_in_net(self)

        for blob in self.blobs_.values():
            blob.find_in_net(self)

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

    def print_norms(self):
        print('data_norm diff_norm blob_name')
        for b in self.blobs:
            data_norm = np.linalg.norm(self.blobs[b].data)
            diff_norm = np.linalg.norm(self.blobs[b].diff)
            print('{:9.2f} {:9.2f} {}'.format(
                data_norm, diff_norm, b  
            ))

    def draw(self, im_file, train=False, rank_dir='BT'):
        param = self.to_param()
        phase = caffe.TRAIN if train else caffe.TEST
        draw_net_to_file(param, im_file, rank_dir, phase)


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
        **kwargs
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
            self.net.copy_from(weights)


class CaffeSubNet(object):
    '''
    An inclusive range of layers in a CaffeNet.
    '''
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
        self.net.backward(start=self.end, end=self.start)


# dynamically subclass CaffeLayer for each caffe layer type
for layer_name in caffe.layer_type_list():
    globals()[layer_name] = CaffeLayer.make_subclass(layer_name)

