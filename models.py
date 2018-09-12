from __future__ import print_function, division
import sys
import os
import re
import argparse
import itertools
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

import caffe_util


SOLVER_NAME_FORMAT = '{solver_name}_{G_train_iter:d}_{D_train_iter:d}_{solver_options}_{instance_noise:.1f}_'

DISC_NAME_FORMAT = 'disc2_in{D_arch_options}'
OLD_DISC_NAME_FORMAT = 'disc_{data_dim:d}_{D_n_levels:d}_{D_conv_per_level:d}_{D_n_filters:d}_{D_width_factor:d}_in'

GEN_NAME_FORMATS = {
    (1, 1): '{encode_type}e11_{data_dim:d}_{G_n_levels:d}_{G_conv_per_level:d}' \
            + '_{G_n_filters:d}_{G_pool_type}_{G_depool_type}',

    (1, 2): '{encode_type}e12_{data_dim:d}_{resolution:.1f}_{G_n_levels:d}_{G_conv_per_level:d}' \
            + '_{G_n_filters:d}_{G_width_factor:d}_{G_loss_types}',

    (1, 3): '{encode_type}e13_{data_dim:d}_{resolution:.1f}_{n_levels:d}_{conv_per_level:d}' \
            + '{arch_options}_{n_filters:d}_{width_factor:d}_{n_latent:d}_{loss_types}',

    (1, 4): '{encode_type}e14_{data_dim:d}_{resolution:.1f}_{G_n_levels:d}_{G_conv_per_level:d}' \
            + '{G_arch_options}_{G_loss_types}'
}


GEN_SEARCH_SPACES = {
    (1, 1): dict(encode_type=['c', 'a'],
                 data_dim=[24],
                 resolution=[0.5],
                 n_levels=[1, 2, 3, 4, 5],
                 conv_per_level=[1, 2, 3],
                 n_filters=[16, 32, 64, 128],
                 width_factor=[1],
                 n_latent=[None],
                 loss_types=['e'],
                 pool_type=['c', 'm', 'a'],
                 depool_type=['c', 'n']),

    (1, 2): dict(encode_type=['c', 'a'],
                 data_dim=[24],
                 resolution=[0.5, 1.0],
                 n_levels=[2, 3],
                 conv_per_level=[2, 3],
                 n_filters=[16, 32, 64, 128],
                 width_factor=[1, 2, 3],
                 n_latent=[None],
                 loss_types=['e'],
                 pool_type=['a'],
                 depool_type=['n']),

    (1, 3): dict(encode_type=['r-l', '_r-l', 'vr-l', '_vr-l', 'rl-l', '_rl-l', 'rvl-l', '_rvl-l'],
                 data_dim=[12],
                 resolution=[0.5],
                 n_levels=[1],
                 conv_per_level=[1, 2, 3],
                 arch_options=[''],
                 n_filters=[8, 16],
                 width_factor=[1, 2],
                 n_latent=[128],
                 loss_types=['', 'e', 'a'])
}


def parse_encode_type(encode_type):
    old_pat = r'(_)?(v)?(a|c)'
    m = re.match(old_pat, encode_type)
    if m:
        encode_type = encode_type.replace('a', 'd-d')
        encode_type = encode_type.replace('c', 'r-l')
    encode_pat = r'(v)?(d|r|l)' 
    decode_pat = r'(d|r|l)'
    full_pat = r'(_)?(?P<enc>({})+)-(?P<dec>({})+)'.format(encode_pat, decode_pat)
    m = re.match(full_pat, encode_type)
    assert m, 'encode_type did not match pattern {}'.format(full_pat)
    molgrid_data = not m.group(1)
    encoders = re.findall(encode_pat, m.group('enc'))
    decoders = re.findall(decode_pat, m.group('dec'))
    return molgrid_data, encoders, decoders


def format_encode_type(molgrid_data, encoders, decoders):
    encode_str = ''.join(v+e for v,e in encoders)
    decode_str = ''.join(decoders)
    return '{}{}-{}'.format(('_', '')[molgrid_data], encode_str, decode_str)


def make_model(encode_type, data_dim, resolution, n_levels, conv_per_level, arch_options='',
               n_filters=32, width_factor=2, n_latent=1024, loss_types='', batch_size=50,
               conv_kernel_size=3, pool_type='a', depool_type='n'):

    molgrid_data, encoders, decoders = parse_encode_type(encode_type)
    leaky_relu = 'l' in arch_options

    assert pool_type in ['c', 'm', 'a']
    assert depool_type in ['c', 'n']

    dim = data_dim
    bsz = batch_size
    nc = dict(r=16, l=19, d=16+19)

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
            shuffle=True,
            balanced=False,
            random_rotation=True,
            random_translate=2.0,
            radius_multiple=1.5,
            use_covalent_radius=True)

        net._ = caffe.layers.MolGridData(ntop=0, name='data', top=['data', 'label', 'aff'],
            include=dict(phase=caffe.TEST),
            source='TESTFILE',
            root_folder='DATA_ROOT',
            has_affinity=True,
            batch_size=batch_size,
            dimension=(data_dim - 1)*resolution,
            resolution=resolution,
            shuffle=False,
            balanced=False,
            random_rotation=False,
            random_translate=0.0,
            radius_multiple=1.5,
            use_covalent_radius=True)

        net.no_label_aff = caffe.layers.Silence(net.label, net.aff, ntop=0)

        if 'r' in encode_type or 'l' in encode_type:
            net.rec, net.lig = caffe.layers.Slice(net.data, ntop=2, name='slice_rec_lig',
                                                  axis=1, slice_point=nc['r'])

    else:

        if 'd' in encode_type:
            net.data = caffe.layers.Input(shape=dict(dim=[bsz, nc['d'], dim, dim, dim]))

        if 'r' in encode_type:
            net.rec = caffe.layers.Input(shape=dict(dim=[bsz, nc['r'], dim, dim, dim]))

        if 'l' in encode_type:
            net.lig = caffe.layers.Input(shape=dict(dim=[bsz, nc['l'], dim, dim, dim]))

    # encoder(s)
    encoder_tops = []
    for variational, e in encoders:

        encoder_type = dict(d='data', r='rec', l='lig')[e]
        curr_top = net[encoder_type]
        curr_dim = data_dim
        curr_n_filters = nc[e]
        next_n_filters = n_filters

        pool_factors = []
        for i in range(n_levels):

            if i > 0: # pool before convolution

                assert curr_dim > 1, 'nothing to pool at level {}'.format(i)

                pool = '{}_level{}_pool'.format(encoder_type, i)
                for pool_factor in [2, 3, 5, curr_dim]:
                    if curr_dim % pool_factor == 0:
                        break
                pool_factors.append(pool_factor)

                if pool_type == 'c': # convolution with stride

                    net[pool] = caffe.layers.Convolution(curr_top,
                        num_output=curr_n_filters,
                        group=curr_n_filters,
                        weight_filler=dict(type='xavier'),
                        kernel_size=pool_factor,
                        stride=pool_factor)

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
            
            for j in range(conv_per_level): # convolutions

                conv = '{}_level{}_conv{}'.format(encoder_type, i, j)
                net[conv] = caffe.layers.Convolution(curr_top,
                    num_output=next_n_filters,
                    weight_filler=dict(type='xavier'),
                    kernel_size=conv_kernel_size,
                    pad=conv_kernel_size//2)

                relu = '{}_relu'.format(conv)
                net[relu] = caffe.layers.ReLU(net[conv],
                    negative_slope=0.1*leaky_relu,
                    in_place=True)

                curr_top = net[conv]
                curr_n_filters = next_n_filters

        # latent
        if n_latent is not None:
            
            if variational:

                mean = '{}_latent_mean'.format(encoder_type)
                net[mean] = caffe.layers.InnerProduct(curr_top,
                    num_output=n_latent,
                    weight_filler=dict(type='xavier'))

                log_std = '{}_latent_log_std'.format(encoder_type)
                net[log_std] = caffe.layers.InnerProduct(curr_top,
                    num_output=n_latent,
                    weight_filler=dict(type='xavier'))

                std = '{}_latent_std'.format(encoder_type)
                net[std] = caffe.layers.Exp(net[log_std])

                noise = '{}_latent_noise'.format(encoder_type)
                net[noise] = caffe.layers.DummyData(
                    data_filler=dict(type='gaussian'),
                    shape=dict(dim=[bsz, n_latent]))

                std_noise = '{}_latent_std_noise'.format(encoder_type)
                net[std_noise] = caffe.layers.Eltwise(net[noise], net[std],
                    operation=caffe.params.Eltwise.PROD)

                sample = '{}_latent_sample'.format(encoder_type)
                net[sample] = caffe.layers.Eltwise(net[std_noise], net[mean],
                    operation=caffe.params.Eltwise.SUM)

                curr_top = net[sample]

                # K-L divergence

                mean2 = '{}_latent_mean2'.format(encoder_type)
                net[mean2] = caffe.layers.Eltwise(net[mean], net[mean],
                    operation=caffe.params.Eltwise.PROD)

                var = '{}_latent_var'.format(encoder_type)
                net[var] = caffe.layers.Eltwise(net[std], net[std],
                    operation=caffe.params.Eltwise.PROD)

                one = '{}_latent_one'.format(encoder_type)
                net[one] = caffe.layers.DummyData(
                    data_filler=dict(type='constant', value=1),
                    shape=dict(dim=[bsz, n_latent]))

                kldiv = '{}_latent_kldiv'.format(encoder_type)
                net[kldiv] = caffe.layers.Eltwise(net[one], net[log_std], net[mean2], net[var],
                    operation=caffe.params.Eltwise.SUM,
                    coeff=[-0.5, -1.0, 0.5, 0.5])

                loss = 'kldiv_loss' # TODO handle multiple K-L divergence losses
                net[loss] = caffe.layers.Reduction(net[kldiv],
                    operation=caffe.params.Reduction.SUM,
                    loss_weight=1.0/bsz)

            else:

                fc = '{}_latent_fc'.format(encoder_type)
                net[fc] = caffe.layers.InnerProduct(curr_top,
                    num_output=n_latent,
                    weight_filler=dict(type='xavier'))

                curr_top = net[fc]

            encoder_tops.append(curr_top)

    if len(encoder_tops) > 1: # concat latent vectors

        net.latent_concat = caffe.layers.Concat(*encoder_tops, axis=1)
        curr_top = net.latent_concat

    # decoder(s)
    dec_init_dim = curr_dim
    dec_init_n_filters = curr_n_filters
    decoder_tops = []
    for d in decoders:

        decoder_type = dict(d='data', r='rec', l='lig')[d]
        label_top = net[decoder_type]
        label_n_filters = nc[d]

        fc = '{}_latent_defc'.format(decoder_type)
        net[fc] = caffe.layers.InnerProduct(curr_top,
            num_output=dec_init_n_filters*dec_init_dim**3,
            weight_filler=dict(type='xavier'))

        relu = '{}_relu'.format(fc)
        net[relu] = caffe.layers.ReLU(net[fc],
            negative_slope=0.1*leaky_relu,
            in_place=True)

        reshape = '{}_latent_reshape'.format(decoder_type)
        net[reshape] = caffe.layers.Reshape(net[fc],
            shape=dict(dim=[bsz, dec_init_n_filters] + [dec_init_dim]*3))

        curr_top = net[reshape]
        curr_n_filters = dec_init_n_filters
        curr_dim = dec_init_dim

        for i in reversed(range(n_levels)):

            if i < n_levels-1: # upsample before convolution

                unpool = '{}_level{}_unpool'.format(decoder_type, i)
                pool_factor = pool_factors.pop(-1)

                if depool_type == 'c': # deconvolution with stride

                    net[unpool] = caffe.layers.Deconvolution(curr_top,
                        convolution_param=dict(
                            num_output=curr_n_filters,
                            group=curr_n_filters,
                            weight_filler=dict(type='xavier'),
                            kernel_size=pool_factor,
                            stride=pool_factor))

                elif depool_type == 'n': # nearest-neighbor interpolation

                    net[unpool] = caffe.layers.Deconvolution(curr_top,
                        param=dict(lr_mult=0, decay_mult=0),
                        convolution_param=dict(
                            num_output=curr_n_filters,
                            group=curr_n_filters,
                            weight_filler=dict(type='constant', value=1),
                            bias_term=False,
                            kernel_size=pool_factor,
                            stride=pool_factor))

                curr_top = net[unpool]
                curr_dim = int(pool_factor*curr_dim)
                next_n_filters = int(curr_n_filters//width_factor)

            for j in range(conv_per_level): # convolutions

                last_conv = i == 0 and j+1 == conv_per_level
                if last_conv:
                    next_n_filters = label_n_filters

                deconv = '{}_level{}_deconv{}'.format(decoder_type, i, j)
                net[deconv] = caffe.layers.Deconvolution(curr_top,
                        convolution_param=dict(
                            num_output=next_n_filters,
                            weight_filler=dict(type='xavier'),
                            kernel_size=conv_kernel_size,
                            pad=1))

                relu = '{}_relu'.format(deconv)
                net[relu] = caffe.layers.ReLU(net[deconv],
                    negative_slope=0.1*leaky_relu*~last_conv,
                    in_place=True)

                curr_top = net[deconv]
                curr_n_filters = next_n_filters

        # output
        gen = '{}_gen'.format(decoder_type)
        net[gen] = caffe.layers.Power(curr_top)

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

        if 'c' in loss_types:

            net.chan_L2_loss = caffe.layers.Python(curr_top, label_top,
                model='channel_euclidean_loss_layer',
                layer='ChannelEuclideanLossLayer',
                loss_weight=1.0)

        if 'm' in loss_types:

            net.mask_L2_loss = caffe.layers.Python(curr_top, label_top,
                model='masked_euclidean_loss_layer',
                layer='MaskedEuclideanLossLayer',
                loss_weight=0.0)

    return net.to_proto()


def keyword_product(**kwargs):
    for values in itertools.product(*kwargs.itervalues()):
        yield dict(itertools.izip(kwargs.iterkeys(), values))


def percent_index(lst, pct):
    return lst[int(pct*len(lst))]


def orthogonal_samples(n, **kwargs):
    for sample in pyDOE.lhs(len(kwargs), n):
        values = map(percent_index, zip(kwargs, sample))
        yield dict(zip(kwargs, values))


def parse_version(version_str):
    return tuple(map(int, version_str.split('.')))


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', type=parse_version, required=True)
    parser.add_argument('-s', '--do_scaffold', action='store_true')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    model_data = []
    for kwargs in keyword_product(**GEN_SEARCH_SPACES[args.version]):
        model_name = GEN_NAME_FORMATS[args.version].format(**kwargs)
        model_file = os.path.join('models', model_name + '.model')
        net_param = make_model(**kwargs)
        net_param.to_prototxt(model_file)

        if args.do_scaffold:
            net = caffe_util.Net.from_param(net_param, phase=caffe.TRAIN)
            model_data.append((model_name, net.get_n_params(), net.get_size()))
        else:
            print(model_file)

    if args.do_scaffold:
        print('{:30}{:>12}{:>14}'.format('model_name', 'n_params', 'size'))
        for model_name, n_params, size in model_data:
            print('{:30}{:12d}{:10.2f} MiB'.format(model_name, n_params, size/2**20))


if __name__ == '__main__':
    main(sys.argv[1:])
