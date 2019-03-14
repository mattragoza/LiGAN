from __future__ import print_function, division
import sys, os, re, argparse, itertools, ast
from collections import OrderedDict
import numpy as np
import caffe

import caffe_util


# format strings for mapping model params to unique names
NAME_FORMATS = dict(
    data=OrderedDict({
        '11': 'data_{data_dim:d}_{resolution:g}{data_options}'
    }),
    gen=OrderedDict({
        '110': '{encode_type}e11_{data_dim:d}_0',
        '11': '{encode_type}e11_{data_dim:d}_{n_levels:d}_{conv_per_level:d}_{n_filters:d}_{pool_type}_{unpool_type}',
        '12': '{encode_type}e12_{data_dim:d}_{resolution:g}_{n_levels:d}_{conv_per_level:d}_{n_filters:d}_{width_factor:d}_{loss_types}',
        '13': '{encode_type}e13_{data_dim:d}_{resolution:g}{data_options}_{n_levels:d}_{conv_per_level:d}{arch_options}_{n_filters:d}_{width_factor:d}_{n_latent:d}_{loss_types}'
    }),
    disc=OrderedDict({
        '00': 'disc{arch_options}',
        '01': 'disc_{data_dim:d}_{n_levels:d}_{conv_per_level:d}{arch_options}_{n_filters:d}_{width_factor:d}_in',
        '11': 'd11_{data_dim:d}_{n_levels:d}_{conv_per_level:d}{arch_options}_{n_filters:d}_{width_factor:d}_{loss_types}',
    }),
    solver=OrderedDict({
        '10': '{solver_name}',
        '11': '{solver_name}_{gen_train_iter:d}_{disc_train_iter:d}_{train_options}_{instance_noise:g}',
        '12': '{solver_name}_{gen_train_iter:d}_{disc_train_iter:d}_{train_options}_{instance_noise:g}_{loss_weight:g}_{loss_weight_decay:g}',
    }),
)


def write_file(file_, buf):
    with open(file_, 'w') as f:
        f.write(buf)


def write_model(model_file, net_param, params={}):
    '''
    Write net_param to model_file. If params dict is given,
    write the params as comments in the header of model_file.
    '''
    buf = ''.join('# {} = {}\n'.format(p, repr(v)) for p, v in params.items())
    buf += str(net_param)
    write_file(model_file, buf)


def read_file(file_):
    with open(file_, 'r') as f:
        return f.read()


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


def read_params(params_file):
    '''
    Read lines from params_file as param = value pairs.
    '''
    buf = read_file(params_file)
    return parse_params(buf)


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


def parse_name(name, name_format, prefix=''):
    pattern = '^' + name_format.replace('{',   r'(?P<') \
                               .replace(':d}', r'>\d+)') \
                               .replace(':f}', r'>[-+]?(\d*\.\d+|\d+)') \
                               .replace(':g}', r'>[-+]?(\d*\.\d+|\d+)(e[-+]?\d+)?)') \
                               .replace('}',   r'>.*)') + '$'
    try:
        return OrderedDict((prefix+p, v) for p,v in re.match(pattern, name).groupdict().items())
    except AttributeError:
        raise Exception('failed to parse {} with format {}'.format(repr(name), name_format))


def parse_gan_name(gan_model_name):

    print(gan_model_name)

    m = re.match(r'^(.+)_([^_]+e(\d+).+)_((d(isc|(\d+)).*))$', gan_model_name)
    params = dict(
        solver_name=m.group(1),
        gen_model_name=m.group(2),
        gen_model_version=m.group(3),
        disc_model_name=m.group(4),
        disc_model_version='00' if m.group(4) in {'disc', 'disc2'} else \
            ('01' if m.group(5) == 'disc' else m.group(6))
    )
    try:
        params.update(parse_name(params['solver_name'], NAME_FORMATS['solver']['12'], 'job_params.'))
    except:
        try:
            params.update(parse_name(params['solver_name'], NAME_FORMATS['solver']['11'], 'job_params.'))
        except:
            params.update(parse_name(params['solver_name'], NAME_FORMATS['solver']['10'], 'job_params.'))
    try:
        solver_file = 'solvers/{}.solver'.format(params['job_params.solver_name'])
        params.update(**read_params_from_solver(solver_file, 'job_params.solver_params.'))
    except IOError:
        params.update(**read_params_from_solver('solvers/adam0.solver', 'job_params.solver_params.'))
    params.update(parse_name(params['gen_model_name'], NAME_FORMATS['gen'][params['gen_model_version']], 'job_params.gen_model_params.'))
    params.update(parse_name(params['disc_model_name'], NAME_FORMATS['disc'][params['disc_model_version']], 'job_params.disc_model_params.'))
    params['job_params.gen_model_params.loss_types'] += 'g'
    return params


def parse_gen_name(gen_model_name):

    m = re.match(r'[^_]+e(\d+).+', gen_model_name)
    params = dict(gen_model_version=m.group(1))
    try:
        params.update(parse_name(gen_model_name, NAME_FORMATS['gen'][params['gen_model_version']], 'job_params.gen_model_params.'))
    except Exception as e:
        if params['gen_model_version'] == '11':
            try:
                params.update(parse_name(gen_model_name, NAME_FORMATS['gen']['110'], 'job_params.gen_model_params.'))
            except:
                raise e
        else:
            raise e
    params.update(**read_params_from_solver('solvers/adam0.solver', 'job_params.solver_params.'))
    return params


def parse_encode_type(encode_type):
    disc_pat = r'disc'
    m = re.match(disc_pat, encode_type)
    if m:
        encode_type = '_d-'
    old_pat = r'(_)?(v)?(a|c)'
    m = re.match(old_pat, encode_type)
    if m:
        encode_type = encode_type.replace('a', 'd-d')
        encode_type = encode_type.replace('c', 'r-l')
    encode_pat = r'(v)?(d|r|l)' 
    decode_pat = r'(d|r|l|y)'
    full_pat = r'(_)?(?P<enc>({})+)-(?P<dec>({})*)'.format(encode_pat, decode_pat)
    m = re.match(full_pat, encode_type)
    assert m, 'encode_type did not match pattern {}'.format(full_pat)
    molgrid_data = not m.group(1)
    map_ = dict(r='rec', l='lig', d='data')
    encoders = [(bool(v), map_[e]) for v,e in re.findall(encode_pat, m.group('enc'))]
    decoders = [map_[d] for d in re.findall(decode_pat, m.group('dec'))]
    return molgrid_data, encoders, decoders


def format_encode_type(molgrid_data, encoders, decoders):
    encode_str = ''.join(v+e for v,e in encoders)
    decode_str = ''.join(decoders)
    return '{}{}-{}'.format(('_', '')[molgrid_data], encode_str, decode_str)


def least_prime_factor(n):
    return next(i for i in range(2, n+1) if n%i == 0)


def make_model(encode_type, data_dim, resolution, data_options='', n_levels=0, conv_per_level=0,
               arch_options='', n_filters=32, width_factor=2, n_latent=None, loss_types='',
               batch_size=16, conv_kernel_size=3, pool_type='a', unpool_type='n', growth_rate=16):

    molgrid_data, encoders, decoders = parse_encode_type(encode_type)

    use_covalent_radius = 'c' in data_options
    binary_atoms = 'b' in data_options

    leaky_relu = 'l' in arch_options
    gaussian_output = 'g' in arch_options
    self_attention = 'a' in arch_options
    batch_disc = 'b' in arch_options
    dense_net = 'd' in arch_options      
    init_conv_pool = 'i' in arch_options

    assert len(decoders) <= 1
    assert pool_type in ['c', 'm', 'a']
    assert unpool_type in ['c', 'n']
    assert conv_kernel_size%2 == 1

    n_channels = dict(rec=16, lig=19, data=16+19)

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
            fixed_radius=binary_atoms * np.sqrt(3)*resolution/2 + 1e-6,
            shuffle=True,
            balanced=False,
            random_rotation=True,
            random_translate=2.0,
            radius_multiple=1.5,
            use_covalent_radius=use_covalent_radius)

        net._ = caffe.layers.MolGridData(ntop=0, name='data', top=['data', 'label', 'aff'],
            include=dict(phase=caffe.TEST),
            source='TESTFILE',
            root_folder='DATA_ROOT',
            has_affinity=True,
            batch_size=batch_size,
            dimension=(data_dim - 1)*resolution,
            resolution=resolution,
            binary_occupancy=binary_atoms,
            fixed_radius=binary_atoms * np.sqrt(3)*resolution/2 + 1e-6,
            shuffle=False,
            balanced=False,
            random_rotation=False,
            random_translate=0.0,
            radius_multiple=1.5,
            use_covalent_radius=use_covalent_radius)

        net.rec, net.lig = caffe.layers.Slice(net.data, ntop=2, name='slice_rec_lig',
                                              axis=1, slice_point=n_channels['rec'])

    else:
        net.rec = caffe.layers.Input(shape=dict(dim=[batch_size, n_channels['rec']] + [data_dim]*3))
        net.lig = caffe.layers.Input(shape=dict(dim=[batch_size, n_channels['lig']] + [data_dim]*3))

        net.data = caffe.layers.Concat(net.rec, net.lig, axis=1)

        if not decoders:
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

        # latent
        if n_latent is not None:

            if variational:

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
                net[noise] = caffe.layers.DummyData(
                    data_filler=dict(type='gaussian'),
                    shape=dict(dim=[batch_size, n_latent]))

                std_noise = '{}_latent_std_noise'.format(enc)
                net[std_noise] = caffe.layers.Eltwise(net[noise], net[std],
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
                net[one] = caffe.layers.DummyData(
                    data_filler=dict(type='constant', value=1),
                    shape=dict(dim=[batch_size, n_latent]))

                kldiv = '{}_latent_kldiv'.format(enc)
                net[kldiv] = caffe.layers.Eltwise(net[one], net[log_std], net[mean2], net[var],
                    operation=caffe.params.Eltwise.SUM,
                    coeff=[-0.5, -1.0, 0.5, 0.5])

                loss = 'kldiv_loss' # TODO handle multiple K-L divergence losses
                net[loss] = caffe.layers.Reduction(net[kldiv],
                    operation=caffe.params.Reduction.SUM,
                    loss_weight=1.0/batch_size)

            else:

                fc = '{}_latent_defc'.format(enc)
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
            if 'x' in loss_types:
                gen = '{}_gen'.format(dec)
                net[gen] = caffe.layers.Sigmoid(curr_top)
            else:
                gen = '{}_gen'.format(dec)
                net[gen] = caffe.layers.Power(curr_top)

    elif loss_types: # discriminative model
        label_top = net.label

        # output
        if n_latent > 1:
            net.output = caffe.layers.Softmax(curr_top)
        else:
            net.output = caffe.layers.Sigmoid(curr_top)

    # loss
    if 'e' in loss_types:
        net.L2_loss = caffe.layers.EuclideanLoss(curr_top, label_top, loss_weight=1.0)

    if 'a' in loss_types:

        net.diff = caffe.layers.Eltwise(curr_top, label_top,
            operation=caffe.params.Eltwise.SUM,
            coeff=[-1.0, 1.0])

        net.L1_loss = caffe.layers.Reduction(net.diff,
            operation=caffe.params.Reduction.ASUM,
            loss_weight=1.0)

    if 'f' in loss_types:

        fit = '{}_gen_fit'.format(dec)
        net[fit] = caffe.layers.Python(curr_top,
            module='generate',
            layer='AtomFittingLayer',
            param_str=str(dict(
                resolution=resolution,
                use_covalent_radius=True,
                gninatypes_file='/net/pulsar/home/koes/mtr22/gan/data/O_2_0_0.gninatypes')))

        net.fit_L2_loss = caffe.layers.EuclideanLoss(curr_top, net[fit], loss_weight=1.0)

    if 'F' in loss_types:

        fit = '{}_gen_fit'.format(dec)
        net[fit] = caffe.layers.Python(curr_top,
            module='layers',
            layer='AtomFittingLayer',
            param_str=str(dict(
                resolution=resolution,
                use_covalent_radius=True)))

        net.fit_L2_loss = caffe.layers.EuclideanLoss(curr_top, net[fit], loss_weight=1.0)

    if 'c' in loss_types:

        net.chan_L2_loss = caffe.layers.Python(curr_top, label_top,
            module='layers',
            layer='ChannelEuclideanLossLayer',
            loss_weight=1.0)

    if 'm' in loss_types:

        net.mask_L2_loss = caffe.layers.Python(curr_top, label_top,
            model='layers',
            layer='MaskedEuclideanLossLayer',
            loss_weight=0.0)

    if 'x' in loss_types:

        if n_latent > 1 and not decoders:
            net.log_loss = caffe.layers.SoftmaxWithLoss(curr_top, label_top, loss_weight=1.0)
        else:
            net.log_loss = caffe.layers.SigmoidCrossEntropyLoss(curr_top, label_top, loss_weight=1.0)

    if 'w' in loss_types:

        net.wass_sign = caffe.layers.Power(label_top, scale=-2, shift=1)
        net.wass_prod = caffe.layers.Eltwise(net.wass_sign, curr_top,
            operation=caffe.params.Eltwise.PROD)
        net.wass_loss = caffe.layers.Reduction(net.wass_prod,
            operation=caffe.params.Reduction.MEAN,
            loss_weight=1.0)

    return net.to_proto()


def parse_version(version_str):
    return tuple(map(int, version_str.split('.'))) if version_str else None


def get_last_value(ord_dict):
    return ord_dict[sorted(ord_dict.keys())[-1]]


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Create model prototxt files')
    parser.add_argument('params_file', help='file defining model params or dimensions of param space')
    parser.add_argument('-m', '--model_type', required=True, help='model name format type (data, gen, or disc)')
    parser.add_argument('-v', '--version', required=True, help='model name format version (e.g. 13, default most recent)')
    parser.add_argument('-s', '--scaffold', action='store_true', help='do Caffe model scaffolding')
    parser.add_argument('-o', '--out_prefix', default='models', help='common output prefix for model files')
    parser.add_argument('--gpu', default=False, action='store_true')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    if args.scaffold and args.gpu:
        caffe.set_mode_gpu()

    name_format = NAME_FORMATS[args.model_type][args.version]
    param_space = read_param_space(args.params_file)

    model_data = []
    for params in param_space_product(param_space):

        model_name = name_format.format(**params)
        model_file = os.path.join(args.out_prefix, model_name + '.model')
        net_param = make_model(**params)
        write_model(model_file, net_param, params)

        if args.scaffold:
            net = caffe_util.Net.from_param(net_param, phase=caffe.TRAIN)
            model_data.append((model_name, net.get_n_params(), net.get_size()))
        else:
            print(model_file)

    if args.scaffold:
        print('{:30}{:>12}{:>14}'.format('model_name', 'n_params', 'size'))
        for model_name, n_params, size in model_data:
            print('{:30}{:12d}{:10.2f} MiB'.format(model_name, n_params, size/2**20))


if __name__ == '__main__':
    main(sys.argv[1:])
