from __future__ import print_function, division
import sys, os, re, argparse, itertools, ast
from collections import OrderedDict
import numpy as np
import pandas as pd
import caffe

import io, name_formats, params
import molgrid
import caffe_util as cu
from benchmark import benchmark_net


# mapping of pool_types to PoolingParam.pool enum values
pool_type_map = dict(
    a=cu.Pooling.param_type.AVE,
    m=cu.Pooling.param_type.MAX,
    s=cu.Pooling.param_type.STOCHASTIC,
)


class Module(object):

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Sequential(Module):

    def __init__(self, *args):
        self.modules = []
        for arg in args:
            self.add_module(arg)

    def add_module(self, module):
        self.modules.append(module)

    def __call__(self, bottom):
        for module in self.modules:
            top = module(bottom)
            bottom = top
        return top

    def forward(self):
        for module in self.modules:
            module.forward()

    def backward(self):
        for module in reversed(self.modules):
            module.backward()


class ConvReLU(Sequential):

    def __init__(self, n_input, n_output, kernel_size, relu_leak):

        conv = cu.Convolution(
            num_output=n_output,
            kernel_size=kernel_size,
            pad=kernel_size//2,
            weight_filler=dict(type='xavier'),
        )
        relu = cu.ReLU(
            negative_slope=relu_leak,
            in_place=True,
        )
        super().__init__(conv, relu)


class ConvBlock(Sequential):

    def __init__(
        self,
        n_convs,
        n_input,
        n_output,
        kernel_size,
        relu_leak,
        dense_net=False,
    ):
        if dense_net:
            raise NotImplementedError('TODO densely-connected')

        super().__init__()
        for i in range(n_convs):
            conv_relu = ConvReLU(n_input, n_output, kernel_size, relu_leak)
            n_input = n_output
            self.add_module(conv_relu)


class DeconvReLU(Sequential):

    def __init__(self, n_input, n_output, kernel_size, relu_leak):

        deconv = cu.Deconvolution(
            num_output=n_output,
            kernel_size=kernel_size,
            pad=kernel_size//2,
            weight_filler=dict(type='xavier'),
        )
        relu = cu.ReLU(
            negative_slope=relu_leak,
            in_place=True,
        )
        super().__init__(deconv, relu)


class DeconvBlock(Sequential):

    def __init__(
        self,
        n_deconvs,
        n_input, 
        n_output,
        kernel_size,
        relu_leak,
        dense_net=False,
    ):
        if dense_net:
            raise NotImplementedError('TODO densely-connected')

        super().__init__()
        for i in range(n_deconvs):
            deconv_relu = DeconvReLU(
                n_input, n_output, kernel_size, relu_leak
            )
            n_input = n_output
            self.add_module(deconv_relu)


class Pooling(Sequential):

    def __init__(self, n_input, pool_type, pool_factor):

        if pool_type in pool_type_map:

            pool = cu.Pooling(
                pool=pool_type_map[pool_type],
                kernel_size=pool_factor,
                stride=pool_factor,
            )

        elif pool_type == 'c': # strided convolution

            pool = cu.Convolution(
                num_output=n_input,
                group=n_input,
                kernel_size=pool_factor,
                stride=pool_factor,
                weight_filler=dict(type='xavier'),
                engine=cu.Convolution.param_type.CAFFE,
            )

        else:
            raise ValueError('unknown pool_type ' + repr(pool_type))

        super().__init__(pool)


class Unpooling(Sequential):

    def __init__(self, n_input, unpool_type, unpool_factor):

        if unpool_type == 'n': # nearest-neighbor

            unpool = cu.Deconvolution(
                num_output=n_input,
                group=n_input,
                kernel_size=unpool_factor,
                stride=unpool_factor,
                weight_filler=dict(type='constant', value=1),
                bias_term=False,
                engine=cu.Convolution.param_type.CAFFE,
            )
            unpool.lr_mult = 0
            unpool.decay_mult = 0

        elif unpool_type == 'c': # strided deconvolution

            unpool = cu.Deconvolution(
                num_output=n_input,
                group=n_input,
                kernel_size=unpool_factor,
                stride=unpool_factor,
                weight_filler=dict(type='xavier'),
                engine=cu.Convolution.param_type.CAFFE,
            )

        else:
            raise ValueError('unknown unpool_type ' + repr(unpool_type))

        super().__init__(unpool)


class FcReshape(Sequential):

    def __init__(self, n_input, out_shape, relu_leak):
        fc = cu.InnerProduct(
            num_output=np.prod(out_shape),
            weight_filler=dict(type='xavier'),
        )
        relu = cu.ReLU(negative_slope=relu_leak, in_place=True)
        reshape = cu.Reshape(dict(dim=[-1] + list(out_shape)))
        super().__init__(fc, relu, reshape)


class Encoder(Sequential):

    # TODO reimplement the following:
    # - self-attention
    # - densely-connected
    # - batch discrimination
    # - fully-convolutional
    
    def __init__(
        self,
        n_channels,
        grid_dim,
        n_filters,
        width_factor,
        n_levels,
        conv_per_level,
        kernel_size,
        relu_leak,
        pool_type,
        pool_factor,
        n_output,
        init_conv_pool=False,
    ):
        super().__init__()

        # track changing dimensions
        self.n_channels = n_channels
        self.grid_dim = grid_dim

        if init_conv_pool:
            self.add_conv(n_filters, kernel_size, relu_leak)
            self.add_pool(pool_type, pool_factor)

        for i in range(n_levels):

            if i > 0: # pool between conv blocks
                self.add_pool(pool_type, pool_factor)
                n_filters *= width_factor

            self.add_conv_block(
                conv_per_level, n_filters, kernel_size, relu_leak
            )

        self.add_fc(n_output)

    def add_conv(self, n_filters, kernel_size, relu_leak):
        conv = ConvReLU(self.n_channels, n_filters, kernel_size, relu_leak)
        self.add_module(conv)
        self.n_channels = n_filters

    def add_pool(self, pool_type, pool_factor):
        pool = Pooling(self.n_channels, pool_type, pool_factor)
        self.add_module(pool)
        self.grid_dim //= pool_factor

    def add_conv_block(self, n_convs, n_filters, kernel_size, relu_leak):
        conv_block = ConvBlock(
            n_convs, self.n_channels, n_filters, kernel_size, relu_leak
        )
        self.add_module(conv_block)
        self.n_channels = n_filters

    def add_fc(self, n_output):
        fc = cu.InnerProduct(n_output, weight_filler=dict(type='xavier'))
        self.add_module(fc)


class Decoder(Sequential):

    # TODO re-implement the following:
    # - self-attention
    # - densely-connected
    # - fully-convolutional
    # - gaussian output

    def __init__(
        self,
        n_input,
        grid_dim,
        n_channels,
        width_factor,
        n_levels,
        deconv_per_level,
        kernel_size,
        relu_leak,
        unpool_type,
        unpool_factor,
        n_output,
        final_unpool=False,
    ):
        super().__init__()

        # first fc layer maps to initial grid shape
        self.add_fc_reshape(n_input, n_channels, grid_dim, relu_leak)
        n_filters = n_channels

        for i in reversed(range(n_levels)):

            if i + 1 < n_levels: # unpool between deconv blocks
                self.add_unpool(unpool_type, unpool_factor)
                n_filters //= width_factor

            self.add_deconv_block(
                deconv_per_level, n_filters, kernel_size, relu_leak
            )

        if final_unpool:
            self.add_unpool(unpool_type, unpool_factor)

        # final deconv maps to correct n_output channels
        self.add_deconv(n_output, kernel_size, relu_leak)

    def add_fc_reshape(self, n_input, n_channels, grid_dim, relu_leak):
        out_shape = (n_channels,) + (grid_dim,)*3
        fc_reshape = FcReshape(n_input, out_shape, relu_leak)
        self.add_module(fc_reshape)
        self.n_channels = n_channels
        self.grid_dim = grid_dim

    def add_unpool(self, unpool_type, unpool_factor):
        unpool = Unpooling(self.n_channels, unpool_type, unpool_factor)
        self.add_module(unpool)
        self.grid_dim *= unpool_factor

    def add_deconv(self, n_filters, kernel_size, relu_leak):
        deconv = DeconvReLU(
            self.n_channels, n_filters, kernel_size, relu_leak
        )
        self.add_module(deconv)
        self.n_channels = n_filters

    def add_deconv_block(self, n_deconvs, n_filters, kernel_size, relu_leak):
        deconv_block = DeconvBlock(
            n_deconvs, self.n_channels, n_filters, kernel_size, relu_leak
        )
        self.add_module(deconv_block)
        self.n_channels = n_filters


class EncoderDecoder(Sequential):

    def __init__(
        self,
        n_channels=19,
        grid_dim=48,
        n_filters=32,
        width_factor=2,
        n_levels=4,
        conv_per_level=3,
        kernel_size=3,
        relu_leak=0.1,
        pool_factor=2,
        pool_type='a',
        unpool_type='n',
        n_latent=1024,
        init_conv_pool=False,
    ):
        self.encoder = Encoder(
            n_channels=n_channels,
            grid_dim=grid_dim,
            n_filters=n_filters,
            width_factor=width_factor,
            n_levels=n_levels,
            conv_per_level=conv_per_level,
            kernel_size=kernel_size,
            relu_leak=relu_leak,
            pool_type=pool_type,
            pool_factor=pool_factor,
            n_output=n_latent,
            init_conv_pool=init_conv_pool,
        )

        self.decoder = Decoder(
            n_input=n_latent,
            grid_dim=self.encoder.grid_dim,
            n_channels=self.encoder.n_channels,
            width_factor=width_factor,
            n_levels=n_levels,
            deconv_per_level=conv_per_level,
            kernel_size=kernel_size,
            relu_leak=relu_leak,
            unpool_type=unpool_type,
            unpool_factor=pool_factor,
            n_output=n_channels,
            final_unpool=init_conv_pool,
        )
        super().__init__(self.encoder, self.decoder)



def write_model(model_file, net_param, model_params={}):
    '''
    Write net_param to model_file. If params dict is given,
    write the params as comments in the header of model_file.
    '''
    buf = ''
    if model_params:
        buf += params.format_params(model_params, '# ')
    buf += str(net_param)
    io.write_file(model_file, buf)


def write_models(model_dir, param_space, scaffold=False, n_benchmark=0, verbose=False):
    '''
    Write a model in model_dir for every set of params in param_space.
    '''
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if scaffold or n_benchmark > 0:
        df = pd.DataFrame(index=range(len(param_space)))

    model_names = []
    for i, model_params in enumerate(param_space):

        model_file = os.path.join(model_dir, '{}.model'.format(model_params.name))

        if verbose:
            print('CREATING MODEL ' + str(i))
            print('model_file = ' + model_file)
            print('model_name = ' + model_params.name)
            print('model_params = \n' + params.format_params(model_params, '  '), end='')

        net_param = make_model(verbose=verbose, **model_params)
        write_model(model_file, net_param, model_params)
        model_names.append(model_params.name)

        if scaffold or n_benchmark > 0:
            df.loc[i, 'model_file'] = model_file
            net = caffe_util.CaffeNet.from_prototxt(model_file)
            net.scaffold()
            result = benchmark_net(net, n=n_benchmark)
            for key, value in result.mean().items():
                df.loc[i, key] = value
            if verbose:
                print(df.loc[i])

        print(model_file)

    if scaffold or n_benchmark > 0:
        print(df)

    print(model_names)


def parse_params(buf, line_start='', delim='=', prefix='', converter=ast.literal_eval):
    '''
    Parse lines in buf as param = value pairs, filtering by an
    optional line_start pattern. After parsing, a converter
    function is applied to param values.
    '''
    params = OrderedDict()
    line_pat = r'^{}(\S+)\s*{}\s*(.+)$'.format(line_start, delim)
    for p, v in re.findall(line_pat, buf, re.MULTILINE):
        params[prefix+p] = converter(v)
    return params





def read_params_from_model(model_file, prefix=''):
    '''
    Read lines starting with # in model_file as param = value pairs.
    '''
    buf = read_file(model_file)
    return parse_params(buf, line_start=r'#\s*', prefix=prefix)


def read_param_space(params_file):
    '''
    Read lines from params_file as param = values pairs,
    where all values are converted to lists.
    '''
    buf = read_file(params_file)
    converter = lambda v: as_list(ast.literal_eval(v))
    return parse_params(buf, converter=converter)


def read_params_from_solver(solver_file, prefix=''):
    buf = read_file(solver_file)
    return parse_params(buf, delim=':', prefix=prefix)


def as_list(value):
    '''
    Return value as a list if it's not one already.
    '''
    return value if isinstance(value, list) else [value]


def param_space_product(param_space):
    '''
    Iterate over the Cartesian product of values in param_space.
    Expects a dict mapping params to lists of possible values.
    Produces dicts mapping params to specific values.
    '''
    for values in itertools.product(*param_space.itervalues()):
        yield OrderedDict(itertools.izip(param_space.iterkeys(), values))


def percent_index(lst, pct):
    return lst[int(pct*len(lst))]


def param_space_latin_hypercube(n, param_space): #TODO fix this
    for sample in pyDOE.lhs(len(param_space), n):
        values = map(percent_index, zip(param_space, sample))
        yield dict(zip(param_space, value))


def standardize_model_type(model_type):
    if model_type == 'data':
        return '-'
    if model_type == 'disc':
        return '_d-'
    m = re.match(r'(_)?(v)?(a|c)', model_type)
    if m:
        model_type = model_type.replace('a', 'd-d')
        model_type = model_type.replace('c', 'r-l')
    return model_type


def parse_model_type(model_type):
    model_type = standardize_model_type(model_type)
    enc_pat = r'(v)?(d|r|l)'
    dec_pat = r'(d|r|l|y)'
    pat = r'(_)?(?P<enc>({})*)-(?P<dec>({})*)'.format(enc_pat, dec_pat)
    m = re.match(pat, model_type)
    try:
        has_data = not m.group(1)
        map_ = dict(r='rec', l='lig', d='data')
        encoders = [(bool(v), map_[e]) for v,e in re.findall(enc_pat, m.group('enc'))]
        decoders = [map_[d] for d in re.findall(dec_pat, m.group('dec'))]
        return has_data, encoders, decoders
    except AttributeError:
        raise Exception('could not parse encode_type {} with pattern {}'.format(encode_type, pat))


def format_model_type(has_data, encoders, decoders):
    encode_str = ''.join(v+e for v,e in encoders)
    decode_str = ''.join(decoders)
    return '{}{}-{}'.format(('_', '')[has_data], encode_str, decode_str)


def least_prime_factor(n):
    return next(i for i in range(2, n+1) if n%i == 0)


def make_model(
        model_type='data',
        data_dim=24,
        resolution=0.5,
        data_options='',
        n_levels=0,
        conv_per_level=0,
        arch_options='',
        n_filters=32,
        width_factor=2,
        n_latent=0,
        loss_types='',
        batch_size=16,
        conv_kernel_size=3,
        latent_kernel_size=None,
        pool_type='a',
        unpool_type='n',
        growth_rate=16,
        rec_map='my_rec_map',
        lig_map='my_lig_map',
        rec_molcache='',
        lig_molcache='',
        loss_weight_L1=1.0,
        loss_weight_L2=1.0,
        loss_weight_KL=1.0,
        loss_weight_log=1.0,
        loss_weight_wass=1.0,
        verbose=False
    ):

    molgrid_data, encoders, decoders = parse_model_type(model_type)

    use_covalent_radius = 'c' in data_options
    binary_atoms = 'b' in data_options
    fixed_radius = 'f' in data_options

    leaky_relu = 'l' in arch_options
    gaussian_output = 'g' in arch_options
    sigmoid_output = 's' in arch_options
    self_attention = 'a' in arch_options
    batch_disc = 'b' in arch_options
    dense_net = 'd' in arch_options
    init_conv_pool = 'i' in arch_options
    fully_conv = 'c' in arch_options

    assert len(encoders) == 0 or n_latent > 0
    assert len(decoders) <= 1
    assert pool_type in ['c', 'm', 'a']
    assert unpool_type in ['c', 'n']
    assert conv_kernel_size%2 == 1
    assert not latent_kernel_size or latent_kernel_size%2 == 1

    # determine number of rec and lig channels
    n_channels = dict()
    n_channels['rec'] = molgrid.FileMappedGninaTyper(rec_map).num_types()
    n_channels['lig'] = molgrid.FileMappedGninaTyper(lig_map).num_types()
    n_channels['data'] = n_channels['rec'] + n_channels['lig']

    net = caffe.NetSpec()

    # input
    if molgrid_data:

        net.data, net.label, net.aff = caffe.layers.MolGridData(ntop=3,
            include=dict(phase=caffe.TRAIN),
            source='TRAINFILE',
            root_folder='DATA_ROOT',
            has_affinity=True,
            batch_size=batch_size,
            dimension=(data_dim - 1)*resolution,
            resolution=resolution,
            binary_occupancy=binary_atoms,
            fixed_radius=fixed_radius and np.sqrt(3)*resolution/2 + 1e-6,
            shuffle=True,
            balanced=False,
            random_rotation=True,
            random_translate=2.0,
            radius_multiple=1.5,
            use_covalent_radius=use_covalent_radius,
            recmap=rec_map,
            ligmap=lig_map,
            recmolcache=rec_molcache,
            ligmolcache=lig_molcache,
        )

        net._ = caffe.layers.MolGridData(ntop=0, name='data', top=['data', 'label', 'aff'],
            include=dict(phase=caffe.TEST),
            source='TESTFILE',
            root_folder='DATA_ROOT',
            has_affinity=True,
            batch_size=batch_size,
            dimension=(data_dim - 1)*resolution,
            resolution=resolution,
            binary_occupancy=binary_atoms,
            fixed_radius=fixed_radius and np.sqrt(3)*resolution/2 + 1e-6,
            shuffle=False,
            balanced=False,
            random_rotation=False,
            random_translate=0.0,
            radius_multiple=1.5,
            use_covalent_radius=use_covalent_radius,
            recmap=rec_map,
            ligmap=lig_map,
            recmolcache=rec_molcache,
            ligmolcache=lig_molcache,
        )

        net.rec, net.lig = caffe.layers.Slice(net.data, ntop=2, name='slice_rec_lig',
                                              axis=1, slice_point=n_channels['rec'])

    else: # no molgrid_data layers, just input blobs
        net.rec = caffe.layers.Input(shape=dict(dim=[batch_size, n_channels['rec']] + [data_dim]*3))
        net.lig = caffe.layers.Input(shape=dict(dim=[batch_size, n_channels['lig']] + [data_dim]*3))
        net.data = caffe.layers.Concat(net.rec, net.lig, axis=1)

        if not decoders: # discriminative model, so need label input blob
            net.label = caffe.layers.Input(shape=dict(dim=[batch_size, n_latent]))

    # encoder(s)
    encoder_tops = []
    for variational, enc in encoders:

        curr_top = net[enc]
        curr_dim = data_dim
        curr_n_filters = n_channels[enc]
        next_n_filters = n_filters
        pool_factors = []

        if init_conv_pool: # initial conv and pooling

            conv = '{}_enc_init_conv'.format(enc)
            net[conv] = caffe.layers.Convolution(curr_top,
                    num_output=next_n_filters,
                    weight_filler=dict(type='xavier'),
                    kernel_size=conv_kernel_size,
                    pad=conv_kernel_size//2)

            curr_top = net[conv]
            curr_n_filters = next_n_filters

            relu = '{}_relu'.format(conv)
            net[relu] = caffe.layers.ReLU(curr_top,
                negative_slope=0.1*leaky_relu,
                in_place=True)

            pool = '{}_enc_init_pool'.format(enc)
            pool_factor = least_prime_factor(curr_dim)
            pool_factors.append(pool_factor)
            net[pool] = caffe.layers.Pooling(curr_top,
                    pool=caffe.params.Pooling.AVE,
                    kernel_size=pool_factor,
                    stride=pool_factor)

            curr_top = net[pool]
            curr_dim = int(curr_dim//pool_factor)

        for i in range(n_levels):

            if i > 0: # pool between convolution blocks

                assert curr_dim > 1, 'nothing to pool at level {}'.format(i)

                pool = '{}_enc_level{}_pool'.format(enc, i)
                pool_factor = least_prime_factor(curr_dim)
                pool_factors.append(pool_factor)

                if pool_type == 'c': # convolution with stride

                    net[pool] = caffe.layers.Convolution(curr_top,
                        num_output=curr_n_filters,
                        group=curr_n_filters,
                        weight_filler=dict(type='xavier'),
                        kernel_size=pool_factor,
                        stride=pool_factor,
                        engine=caffe.params.Convolution.CAFFE)

                elif pool_type == 'm': # max pooling

                    net[pool] = caffe.layers.Pooling(curr_top,
                        pool=caffe.params.Pooling.MAX,
                        kernel_size=pool_factor,
                        stride=pool_factor)

                elif pool_type == 'a': # average pooling

                    net[pool] = caffe.layers.Pooling(curr_top,
                        pool=caffe.params.Pooling.AVE,
                        kernel_size=pool_factor,
                        stride=pool_factor)

                curr_top = net[pool]
                curr_dim = int(curr_dim//pool_factor)
                next_n_filters = int(width_factor*curr_n_filters)

            if self_attention and i == 1:

                att = '{}_enc_level{}_att'.format(enc, i)
                att_f = '{}_f'.format(att)
                net[att_f] = caffe.layers.Convolution(curr_top,
                    num_output=curr_n_filters//8,
                    weight_filler=dict(type='xavier'),
                    kernel_size=1)

                att_g = '{}_g'.format(att)
                net[att_g] = caffe.layers.Convolution(curr_top,
                    num_output=curr_n_filters//8,
                    weight_filler=dict(type='xavier'),
                    kernel_size=1)

                att_s = '{}_s'.format(att)
                net[att_s] = caffe.layers.MatMul(net[att_f], net[att_g], transpose_a=True)

                att_B = '{}_B'.format(att)
                net[att_B] = caffe.layers.Softmax(net[att_s], axis=2)

                att_h = '{}_h'.format(att)
                net[att_h] = caffe.layers.Convolution(curr_top,
                    num_output=curr_n_filters,
                    weight_filler=dict(type='xavier'),
                    kernel_size=1)

                att_o = '{}_o'.format(att)
                net[att_o] = caffe.layers.MatMul(net[att_h], net[att_B], transpose_b=True)

                att_o_reshape = '{}_o_reshape'.format(att)
                net[att_o_reshape] = caffe.layers.Reshape(net[att_o],
                    shape=dict(dim=[batch_size, curr_n_filters] + [curr_dim]*3))

                curr_top = net[att_o_reshape]

            for j in range(conv_per_level): # convolutions

                conv = '{}_enc_level{}_conv{}'.format(enc, i, j)
                net[conv] = caffe.layers.Convolution(curr_top,
                    num_output=next_n_filters,
                    weight_filler=dict(type='xavier'),
                    kernel_size=conv_kernel_size,
                    pad=conv_kernel_size//2)

                if dense_net:
                    concat_tops = [curr_top, net[conv]]

                curr_top = net[conv]
                curr_n_filters = next_n_filters

                relu = '{}_relu'.format(conv)
                net[relu] = caffe.layers.ReLU(curr_top,
                    negative_slope=0.1*leaky_relu,
                    in_place=True)

                if dense_net:

                    concat = '{}_concat'.format(conv)
                    net[concat] = caffe.layers.Concat(*concat_tops, axis=1)

                    curr_top = net[concat]
                    curr_n_filters += next_n_filters

            if dense_net: # bottleneck conv

                conv = '{}_enc_level{}_bottleneck'.format(enc, i)
                next_n_filters = int(curr_n_filters//2)                 #TODO implement bottleneck_factor
                net[conv] = caffe.layers.Convolution(curr_top,
                    num_output=next_n_filters,
                    weight_filler=dict(type='xavier'),
                    kernel_size=1,
                    pad=0)

                curr_top = net[conv]
                curr_n_filters = next_n_filters

        if batch_disc:

            bd_f = '{}_enc_bd_f'.format(enc)
            net[bd_f] = caffe.layers.Reshape(curr_top,
                shape=dict(dim=[batch_size, 1, curr_n_filters*curr_dim**3]))

            bd_f_tile = '{}_tile'.format(bd_f)
            net[bd_f_tile] = caffe.layers.Tile(net[bd_f], axis=1, tiles=batch_size)

            bd_f_T = '{}_T'.format(bd_f)
            net[bd_f_T] = caffe.layers.Reshape(net[bd_f],
                shape=dict(dim=[1, batch_size, curr_n_filters*curr_dim**3]))

            bd_f_T_tile = '{}_tile'.format(bd_f_T)
            net[bd_f_T_tile] = caffe.layers.Tile(net[bd_f_T], axis=0, tiles=batch_size)

            bd_f_diff = '{}_diff'.format(bd_f)
            net[bd_f_diff] = caffe.layers.Eltwise(net[bd_f_tile], net[bd_f_T_tile],
                operation=caffe.params.Eltwise.SUM,
                coeff=[1, -1])

            bd_f_diff2 = '{}2'.format(bd_f_diff)
            net[bd_f_diff2] = caffe.layers.Eltwise(net[bd_f_diff], net[bd_f_diff],
                operation=caffe.params.Eltwise.PROD)

            bd_f_ssd = '{}_ssd'.format(bd_f)
            net[bd_f_ssd] = caffe.layers.Convolution(net[bd_f_diff2],
                param=dict(lr_mult=0, decay_mult=0),
                convolution_param=dict(
                    num_output=1,
                    weight_filler=dict(type='constant', value=1),
                    bias_term=False,
                    kernel_size=[1],
                    engine=caffe.params.Convolution.CAFFE))

            bd_f_ssd_reshape = '{}_reshape'.format(bd_f_ssd)
            net[bd_f_ssd_reshape] = caffe.layers.Reshape(net[bd_f_ssd],
                shape=dict(dim=[batch_size, curr_n_filters] + [curr_dim]*3))

            bd_o = '{}_bd_o'.format(enc)
            net[bd_o] = caffe.layers.Concat(curr_top, net[bd_f_ssd_reshape], axis=1)

            curr_top = net[bd_o]

        # latent space
        if variational:

            if fully_conv: # convolutional latent variables

                mean = '{}_latent_mean'.format(enc)
                net[mean] = caffe.layers.Convolution(curr_top,
                    num_output=n_latent,
                    weight_filler=dict(type='xavier'),
                    kernel_size=latent_kernel_size,
                    pad=latent_kernel_size//2)

                log_std = '{}_latent_log_std'.format(enc)
                net[log_std] = caffe.layers.Convolution(curr_top,
                    num_output=n_latent,
                    weight_filler=dict(type='xavier'),
                    kernel_size=latent_kernel_size,
                    pad=latent_kernel_size//2)

            else:

                mean = '{}_latent_mean'.format(enc)
                net[mean] = caffe.layers.InnerProduct(curr_top,
                    num_output=n_latent,
                    weight_filler=dict(type='xavier'))

                log_std = '{}_latent_log_std'.format(enc)
                net[log_std] = caffe.layers.InnerProduct(curr_top,
                    num_output=n_latent,
                    weight_filler=dict(type='xavier'))

            std = '{}_latent_std'.format(enc)
            net[std] = caffe.layers.Exp(net[log_std])

            noise = '{}_latent_noise'.format(enc)
            noise_shape = [batch_size, n_latent]
            if fully_conv:
                noise_shape += [1]
            net[noise] = caffe.layers.DummyData(
                data_filler=dict(type='gaussian'),
                shape=dict(dim=noise_shape))
            noise_top = net[noise]

            if fully_conv: # broadcast noise sample along spatial axes

                noise_tile = '{}_latent_noise_tile'.format(enc)
                net[noise_tile] = caffe.layers.Tile(net[noise], axis=2, tiles=curr_dim**3)

                noise_reshape = '{}_latent_noise_reshape'.format(enc)
                net[noise_reshape] = caffe.layers.Reshape(net[noise_tile],
                    shape=dict(dim=[batch_size, n_latent, curr_dim, curr_dim, curr_dim]))

                noise_top = net[noise_reshape]

            std_noise = '{}_latent_std_noise'.format(enc)
            net[std_noise] = caffe.layers.Eltwise(noise_top, net[std],
                operation=caffe.params.Eltwise.PROD)

            sample = '{}_latent_sample'.format(enc)
            net[sample] = caffe.layers.Eltwise(net[std_noise], net[mean],
                operation=caffe.params.Eltwise.SUM)

            curr_top = net[sample]

            # K-L divergence

            mean2 = '{}_latent_mean2'.format(enc)
            net[mean2] = caffe.layers.Eltwise(net[mean], net[mean],
                operation=caffe.params.Eltwise.PROD)

            var = '{}_latent_var'.format(enc)
            net[var] = caffe.layers.Eltwise(net[std], net[std],
                operation=caffe.params.Eltwise.PROD)

            one = '{}_latent_one'.format(enc)
            one_shape = [batch_size, n_latent]
            if fully_conv:
                one_shape += [curr_dim]*3
            net[one] = caffe.layers.DummyData(
                data_filler=dict(type='constant', value=1),
                shape=dict(dim=one_shape))

            kldiv_term = '{}_latent_kldiv_term_sum'.format(enc)
            net[kldiv_term] = caffe.layers.Eltwise(net[one], net[log_std], net[mean2], net[var],
                operation=caffe.params.Eltwise.SUM,
                coeff=[-0.5, -1.0, 0.5, 0.5])

            kldiv_batch = '{}_latent_kldiv_batch_sum'.format(enc)
            net[kldiv_batch] = caffe.layers.Reduction(net[kldiv_term],
                operation=caffe.params.Reduction.SUM)

            kldiv_loss = 'kldiv_loss'
            net[kldiv_loss] = caffe.layers.Power(net[kldiv_batch],
                scale=1.0/batch_size, loss_weight=loss_weight_KL)

        else:

            if fully_conv:
                conv = '{}_latent_conv'.format(enc)
                net[conv] = caffe.layers.Convolution(curr_top,
                    num_output=n_latent,
                    weight_filler=dict(type='xavier'),
                    kernel_size=latent_kernel_size,
                    pad=latent_kernel_size//2)
                curr_top = net[conv]

            else:
                fc = '{}_latent_fc'.format(enc)
                net[fc] = caffe.layers.InnerProduct(curr_top,
                    num_output=n_latent,
                    weight_filler=dict(type='xavier'))
                curr_top = net[fc]

        encoder_tops.append(curr_top)

    if len(encoder_tops) > 1: # concat latent vectors

        net.latent_concat = caffe.layers.Concat(*encoder_tops, axis=1)
        curr_top = net.latent_concat

    if decoders: # decoder(s)

        dec_init_dim = curr_dim
        dec_init_n_filters = curr_n_filters
        decoder_tops = []

        for dec in decoders:

            label_top = net[dec]
            label_n_filters = n_channels[dec]
            next_n_filters = dec_init_n_filters if conv_per_level else n_channels[dec]

            if not fully_conv:

                fc = '{}_dec_fc'.format(dec)
                net[fc] = caffe.layers.InnerProduct(curr_top,
                    num_output=next_n_filters*dec_init_dim**3,
                    weight_filler=dict(type='xavier'))

                relu = '{}_relu'.format(fc)
                net[relu] = caffe.layers.ReLU(net[fc],
                    negative_slope=0.1*leaky_relu,
                    in_place=True)

                reshape = '{}_reshape'.format(fc)
                net[reshape] = caffe.layers.Reshape(net[fc],
                    shape=dict(dim=[batch_size, next_n_filters] + [dec_init_dim]*3))

                curr_top = net[reshape]
                curr_n_filters = dec_init_n_filters
                curr_dim = dec_init_dim

            for i in reversed(range(n_levels)):

                if i < n_levels-1: # upsample between convolution blocks

                    unpool = '{}_dec_level{}_unpool'.format(dec, i)
                    pool_factor = pool_factors.pop(-1)

                    if unpool_type == 'c': # deconvolution with stride

                        net[unpool] = caffe.layers.Deconvolution(curr_top,
                            convolution_param=dict(
                                num_output=curr_n_filters,
                                group=curr_n_filters,
                                weight_filler=dict(type='xavier'),
                                kernel_size=pool_factor,
                                stride=pool_factor,
                                engine=caffe.params.Convolution.CAFFE))

                    elif unpool_type == 'n': # nearest-neighbor interpolation

                        net[unpool] = caffe.layers.Deconvolution(curr_top,
                            param=dict(lr_mult=0, decay_mult=0),
                            convolution_param=dict(
                                num_output=curr_n_filters,
                                group=curr_n_filters,
                                weight_filler=dict(type='constant', value=1),
                                bias_term=False,
                                kernel_size=pool_factor,
                                stride=pool_factor,
                                engine=caffe.params.Convolution.CAFFE))

                    curr_top = net[unpool]
                    curr_dim = int(pool_factor*curr_dim)
                    next_n_filters = int(curr_n_filters//width_factor)

                for j in range(conv_per_level): # convolutions

                    deconv = '{}_dec_level{}_deconv{}'.format(dec, i, j)

                    # final convolution has to produce the desired number of output channels
                    last_conv = (i == 0) and (j+1 == conv_per_level) and not (dense_net or init_conv_pool)
                    if last_conv:
                        next_n_filters = label_n_filters

                    net[deconv] = caffe.layers.Deconvolution(curr_top,
                        convolution_param=dict(
                            num_output=next_n_filters,
                            weight_filler=dict(type='xavier'),
                            kernel_size=conv_kernel_size,
                            pad=1))

                    if dense_net:
                        concat_tops = [curr_top, net[deconv]]

                    curr_top = net[deconv]
                    curr_n_filters = next_n_filters

                    relu = '{}_relu'.format(deconv)
                    net[relu] = caffe.layers.ReLU(curr_top,
                        negative_slope=0.1*leaky_relu,
                        in_place=True)

                    if dense_net:

                        concat = '{}_concat'.format(deconv)
                        net[concat] = caffe.layers.Concat(*concat_tops, axis=1)

                        curr_top = net[concat]
                        curr_n_filters += next_n_filters

                if dense_net: # bottleneck conv

                    conv = '{}_dec_level{}_bottleneck'.format(dec, i)

                    last_conv = (i == 0) and not init_conv_pool
                    if last_conv:
                        next_n_filters = label_n_filters
                    else:
                        next_n_filters = int(curr_n_filters//2)         #TODO implement bottleneck_factor

                    net[conv] = caffe.layers.Deconvolution(curr_top,
                        convolution_param=dict(
                            num_output=next_n_filters,
                            weight_filler=dict(type='xavier'),
                            kernel_size=1,
                            pad=0))

                    curr_top = net[conv]
                    curr_n_filters = next_n_filters

                if self_attention and i == 1:

                    att = '{}_dec_level{}_att'.format(dec, i)
                    att_f = '{}_f'.format(att)
                    net[att_f] = caffe.layers.Convolution(curr_top,
                        num_output=curr_n_filters//8,
                        weight_filler=dict(type='xavier'),
                        kernel_size=1)

                    att_g = '{}_g'.format(att)
                    net[att_g] = caffe.layers.Convolution(curr_top,
                        num_output=curr_n_filters//8,
                        weight_filler=dict(type='xavier'),
                        kernel_size=1)

                    att_s = '{}_s'.format(att)
                    net[att_s] = caffe.layers.MatMul(net[att_f], net[att_g], transpose_a=True)

                    att_B = '{}_B'.format(att)
                    net[att_B] = caffe.layers.Softmax(net[att_s], axis=2)

                    att_h = '{}_h'.format(att)
                    net[att_h] = caffe.layers.Convolution(curr_top,
                        num_output=curr_n_filters,
                        weight_filler=dict(type='xavier'),
                        kernel_size=1)

                    att_o = '{}_o'.format(att)
                    net[att_o] = caffe.layers.MatMul(net[att_h], net[att_B], transpose_b=True)

                    att_o_reshape = '{}_o_reshape'.format(att)
                    net[att_o_reshape] = caffe.layers.Reshape(net[att_o],
                        shape=dict(dim=[batch_size, curr_n_filters] + [curr_dim]*3))

                    curr_top = net[att_o_reshape]

            if init_conv_pool: # final upsample and deconv

                unpool = '{}_dec_final_unpool'.format(dec) 
                pool_factor = pool_factors.pop(-1)
                net[unpool] = caffe.layers.Deconvolution(curr_top,
                    param=dict(lr_mult=0, decay_mult=0),
                    convolution_param=dict(
                        num_output=curr_n_filters,
                        group=curr_n_filters,
                        weight_filler=dict(type='constant', value=1),
                        bias_term=False,
                        kernel_size=pool_factor,
                        stride=pool_factor,
                        engine=caffe.params.Convolution.CAFFE))

                curr_top = net[unpool]
                curr_dim = int(pool_factor*curr_dim)

                deconv = '{}_dec_final_deconv'.format(dec)
                next_n_filters = label_n_filters
                net[deconv] = caffe.layers.Deconvolution(curr_top,
                convolution_param=dict(
                    num_output=next_n_filters,
                    weight_filler=dict(type='xavier'),
                    kernel_size=conv_kernel_size,
                    pad=1))

                curr_top = net[deconv]
                curr_n_filters = next_n_filters

                relu = '{}_relu'.format(deconv)
                net[relu] = caffe.layers.ReLU(curr_top,
                    negative_slope=0.1*leaky_relu,
                    in_place=True)

            # output
            if gaussian_output:

                gauss_kernel_size = 7
                conv = '{}_dec_gauss_conv'.format(dec)
                net[conv] = caffe.layers.Convolution(curr_top,
                    param=dict(lr_mult=0, decay_mult=0),
                    num_output=label_n_filters,
                    group=label_n_filters,
                    weight_filler=dict(type='constant', value=0), # fill from saved weights
                    bias_term=False,
                    kernel_size=gauss_kernel_size,
                    pad=gauss_kernel_size//2,
                    engine=caffe.params.Convolution.CAFFE)

                curr_top = net[conv]

            # separate output blob from the one used for gen loss is needed for GAN backprop
            if sigmoid_output:
                gen = '{}_gen'.format(dec)
                net[gen] = caffe.layers.Sigmoid(curr_top)
            else:
                gen = '{}_gen'.format(dec)
                net[gen] = caffe.layers.Power(curr_top)

    elif loss_types: # discriminative model
        label_top = net.label

        # output
        if sigmoid_output:
            if n_latent > 1:
                net.output = caffe.layers.Softmax(curr_top)
            else:
                net.output = caffe.layers.Sigmoid(curr_top)
        else:
            net.output = caffe.layers.Power(curr_top)

    # loss
    if 'e' in loss_types:

        net.L2_loss = caffe.layers.EuclideanLoss(
            curr_top,
            label_top,
            loss_weight=loss_weight_L2
        )

    if 'a' in loss_types:

        net.diff = caffe.layers.Eltwise(
            curr_top,
            label_top,
            operation=caffe.params.Eltwise.SUM,
            coeff=[1.0, -1.0]
        )
        net.abs_sum = caffe.layers.Reduction(
            net.diff,
            operation=caffe.params.Reduction.ASUM
        )
        net.L1_loss = caffe.layers.Power(
            net.abs_sum,
            scale=1.0/batch_size,
            loss_weight=loss_weight_L1
        )

    if 'f' in loss_types:

        fit = '{}_gen_fit'.format(dec)
        net[fit] = caffe.layers.Python(
            curr_top,
            module='generate',
            layer='AtomFittingLayer',
            param_str=str(dict(
                resolution=resolution,
                use_covalent_radius=True,
                gninatypes_file='/net/pulsar/home/koes/mtr22/gan/data/O_2_0_0.gninatypes'
            ))
        )
        net.fit_L2_loss = caffe.layers.EuclideanLoss(
            curr_top,
            net[fit],
            loss_weight=1.0
        )

    if 'F' in loss_types:

        fit = '{}_gen_fit'.format(dec)
        net[fit] = caffe.layers.Python(curr_top,
            module='layers',
            layer='AtomFittingLayer',
            param_str=str(dict(
                resolution=resolution,
                use_covalent_radius=True
            ))
        )
        net.fit_L2_loss = caffe.layers.EuclideanLoss(
            curr_top,
            net[fit],
            loss_weight=1.0
        )

    if 'c' in loss_types:

        net.chan_L2_loss = caffe.layers.Python(
            curr_top,
            label_top,
            module='layers',
            layer='ChannelEuclideanLossLayer',
            loss_weight=1.0
        )

    if 'm' in loss_types:

        net.mask_L2_loss = caffe.layers.Python(
            curr_top,
            label_top,
            model='layers',
            layer='MaskedEuclideanLossLayer',
            loss_weight=0.0
        )

    if 'x' in loss_types:

        if n_latent > 1 and not decoders:
            net.log_loss = caffe.layers.SoftmaxWithLoss(
                curr_top,
                label_top,
                loss_weight=loss_weight_log
            )
        else:
            net.log_loss = caffe.layers.SigmoidCrossEntropyLoss(
                curr_top,
                label_top,
                loss_weight=loss_weight_log
            )

    if 'w' in loss_types:

        net.wass_sign = caffe.layers.Power(
            label_top,
            scale=-2,
            shift=1
        )
        net.wass_prod = caffe.layers.Eltwise(
            net.wass_sign,
            curr_top,
            operation=caffe.params.Eltwise.PROD
        )
        net.wass_loss = caffe.layers.Reduction(
            net.wass_prod,
            operation=caffe.params.Reduction.MEAN,
            loss_weight=loss_weight_wass
        )

    if verbose:
        print('iterating over dict of net top blobs and layers')
        for k, v in net.tops.items():
            print('top name = ' + k)
            try:
                print('layer params = ' + repr(v.fn.params))
            except AttributeError:
                print('layer params = ' + repr(v.params))

    return net.to_proto()


def parse_version(version_str):
    return tuple(map(int, version_str.split('.'))) if version_str else None


def get_last_value(ord_dict):
    return ord_dict[sorted(ord_dict.keys())[-1]]


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Create model prototxt files from model params')
    parser.add_argument('params_file', help='file defining model params or dimensions of param space')
    parser.add_argument('-o', '--out_dir', required=True, help='common output directory for model files')
    parser.add_argument('-n', '--model_name', help='custom model name format')
    parser.add_argument('-m', '--model_type', default=None, help='model type, for default model name format (e.g. data, gen, or disc)')
    parser.add_argument('-v', '--version', default=None, help='version, for default model name format (e.g. 13, default most recent)')
    parser.add_argument('--scaffold', default=False, action='store_true', help='attempt to scaffold models in Caffe and estimate memory usage')
    parser.add_argument('--benchmark', default=0, type=int, help='benchmark N forward-backward pass times and actual memory usage')
    parser.add_argument('--verbose', default=False, action='store_true', help='print out more info for debugging prototxt creation')
    parser.add_argument('--gpu', default=False, action='store_true', help='if benchmarking, use the GPU')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    if not args.model_name and not args.model_type:
        raise ValueError('must specify a custom model name format or a model type to use the default name format')

    if not args.model_name: # use a default name format

        if args.version is None: # use latest version
            if args.model_type == 'data':
                args.version = '11'
            elif args.model_type == 'gen':
                args.version = '13'
            elif args.model_type == 'disc':
                args.version = '11'

        args.model_name = NAME_FORMATS[args.model_type][args.version]

    if args.gpu:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    param_space = params.ParamSpace(args.params_file, format=args.model_name.format)
    write_models(args.out_dir, param_space, args.scaffold, args.benchmark, args.verbose)


if __name__ == '__main__':
    main(sys.argv[1:])
