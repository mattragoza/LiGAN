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


SOLVER_NAME_FORMAT = '{solver_name}_{gen_train_iter:d}_{disc_train_iter:d}_{solver_options}_{instance_noise:.1f}_'

DISC_NAME_FORMAT = '_disc2_in{disc_options}'

GEN_NAME_FORMATS = {
    (1, 1): '{encode_type}e11_{data_dim:d}_{n_levels:d}_{conv_per_level:d}' \
            + '_{n_filters:d}_{pool_type}_{depool_type}',

    (1, 2): '{encode_type}e12_{data_dim:d}_{resolution:.1f}_{n_levels:d}_{conv_per_level:d}' \
            + '_{n_filters:d}_{width_factor:d}_{loss_types}',

    (1, 3): '{encode_type}e13_{data_dim:d}_{resolution:.1f}_{n_levels:d}_{conv_per_level:d}' \
            + '{gen_options}_{n_filters:d}_{width_factor:d}_{n_latent:d}_{loss_types}',

    (1, 4): '{encode_type}e14_{data_dim:d}_{resolution:.1f}_{n_levels:d}_{conv_per_level:d}' \
            + '{gen_options}_{loss_types}'
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

    (1, 3): dict(encode_type=['vr-l', '_vr-l', 'rl-l', '_rl-l', 'rvl-l', '_rvl-l'],
                 data_dim=[12],
                 resolution=[0.5],
                 n_levels=[1],
                 conv_per_level=[1, 2, 3],
                 gen_options=[''],
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


def make_model(encode_type, data_dim, resolution, n_levels, conv_per_level, gen_options='',
               n_filters=32, width_factor=2, n_latent=1024, loss_types='', batch_size=50,
               conv_kernel_size=3, pool_type='a', depool_type='n'):

    molgrid_data, encoders, decoders = parse_encode_type(encode_type)
    leaky_relu = 'l' in gen_options

    assert pool_type in ['c', 'm', 'a']
    assert depool_type in ['c', 'n']

    n_channels_map = dict(rec=16, lig=19, data=16+19)

    net = caffe_util.NetParameter()

    # input
    if molgrid_data:
        for training in [True, False]:

            data_layer = net.layer.add()
            data_layer.update(name='data',
                              type='MolGridData',
                              top=['data', 'label', 'aff'],
                              include=[dict(phase=caffe.TRAIN if training else caffe.TEST)])

            data_param = data_layer.molgrid_data_param
            data_param.update(source='TRAINFILE' if training else 'TESTFILE',
                              root_folder='DATA_ROOT',
                              has_affinity=True,
                              batch_size=batch_size,
                              dimension=(data_dim - 1)*resolution,
                              resolution=resolution,
                              shuffle=training,
                              balanced=False,
                              random_rotation=training,
                              random_translate=training*2.0,
                              radius_multiple=1.00,
                              use_covalent_radius=True)

        net.layer.add().update(name='no_label', type='Silence', bottom=['label'])
        net.layer.add().update(name='no_aff', type='Silence', bottom=['aff'])

        if 'r' in encode_type or 'l' in encode_type: # split rec and lig grids

            slice_layer = net.layer.add()
            slice_layer.update(name='slice_rec_lig',
                               type='Slice',
                               bottom=['data'],
                               top=['rec', 'lig'],
                               slice_param=dict(axis=1, slice_point=[n_channels_map['rec']]))
    else:
        if 'd' in encode_type:

            data_layer = net.layer.add()
            data_layer.update(name='data',
                              type='Input',
                              top=['data'])
            data_layer.input_param.shape.add(dim=[batch_size, n_channels_map['data'],
                                                  data_dim, data_dim, data_dim])

        if 'r' in encode_type:

            rec_layer = net.layer.add()
            rec_layer.update(name='rec',
                             type='Input',
                             top=['rec'])
            rec_layer.input_param.shape.add(dim=[batch_size, n_channels_map['rec'],
                                                 data_dim, data_dim, data_dim])

        if 'l' in encode_type:

            lig_layer = net.layer.add()
            lig_layer.update(name='lig',
                             type='Input',
                             top=['lig'])
            lig_layer.input_param.shape.add(dim=[batch_size, n_channels_map['lig'],
                                                 data_dim, data_dim, data_dim])

    # encoder(s)
    encoder_tops = []
    for variational, e in encoders:

        encoder_type = {'d':'data', 'r':'rec', 'l':'lig'}[e]
        curr_top = encoder_type
        curr_dim = data_dim
        curr_n_filters = n_channels_map[encoder_type]
        next_n_filters = n_filters
        pool_factors = []

        for i in range(n_levels):

            if i > 0: # pool before convolution

                assert curr_dim > 1, 'nothing to pool at level {}'.format(i)

                pool_name = '{}_level{}_pool'.format(encoder_type, i)
                pool_layer = net.layer.add()
                pool_layer.update(name=pool_name,
                                  bottom=[curr_top],
                                  top=[pool_name])

                if pool_type == 'c': # convolution with stride 2

                    pool_layer.type = 'Convolution'
                    pool_param = pool_layer.convolution_param
                    pool_param.update(num_output=curr_n_filters,
                                      group=curr_n_filters,
                                      weight_filler=dict(type='xavier'))

                elif pool_type == 'm': # max pooling

                    pool_layer.type = 'Pooling'
                    pool_param = pool_layer.pooling_param
                    pool_param.pool = caffe.params.Pooling.MAX

                elif pool_type == 'a': # average pooling

                    pool_layer.type = 'Pooling'
                    pool_param = pool_layer.pooling_param
                    pool_param.pool = caffe.params.Pooling.AVE

                for pool_factor in [2, 3, 5, curr_dim]:
                    if curr_dim % pool_factor == 0:
                        break
                pool_factors.append(pool_factor)
                pool_param.update(kernel_size=[pool_factor], stride=[pool_factor], pad=[0])

                curr_top = pool_name
                curr_dim = int(curr_dim//pool_factor)
                next_n_filters = int(width_factor*curr_n_filters)
            
            for j in range(conv_per_level): # convolutions

                conv_name = '{}_level{}_conv{}'.format(encoder_type, i, j)
                conv_layer = net.layer.add()
                conv_layer.update(name=conv_name,
                                  type='Convolution',
                                  bottom=[curr_top],
                                  top=[conv_name])

                conv_param = conv_layer.convolution_param
                conv_param.update(num_output=next_n_filters,
                                  kernel_size=[conv_kernel_size],
                                  stride=[1],
                                  pad=[conv_kernel_size//2],
                                  weight_filler=dict(type='xavier'))


                relu_name = '{}_level{}_relu{}'.format(encoder_type, i, j)
                relu_layer = net.layer.add()
                relu_layer.update(name=relu_name,
                                  type='ReLU',
                                  bottom=[conv_name],
                                  top=[conv_name])
                relu_layer.relu_param.negative_slope = 0.1 * leaky_relu
                relu_layer.relu_param.engine = caffe.params.ReLU.CAFFE

                curr_top = conv_name
                curr_n_filters = next_n_filters

        dec_init_dim = curr_dim
        dec_init_n_filters = curr_n_filters

        # latent
        if n_latent is not None:
            
            if variational:

                fc_name = '{}_latent_mean'.format(encoder_type)
                fc_layer = net.layer.add()
                fc_layer.update(name=fc_name,
                                type='InnerProduct',
                                bottom=[curr_top],
                                top=[fc_name])
                fc_param = fc_layer.inner_product_param
                fc_param.update(num_output=n_latent,
                                weight_filler=dict(type='xavier'))
                latent_mean = fc_name

                fc_name = '{}_latent_log_std'.format(encoder_type)
                fc_layer = net.layer.add()
                fc_layer.update(name=fc_name,
                                type='InnerProduct',
                                bottom=[curr_top],
                                top=[fc_name])
                fc_param = fc_layer.inner_product_param
                fc_param.update(num_output=n_latent,
                                weight_filler=dict(type='xavier'))
                latent_log_std = fc_name

                exp_name = '{}_latent_std'.format(encoder_type)
                exp_layer = net.layer.add()
                exp_layer.update(name=exp_name,
                                 type='Exp',
                                 bottom=[latent_log_std],
                                 top=[exp_name])
                latent_std = exp_name

                noise_name = '{}_latent_noise'.format(encoder_type)
                noise_layer = net.layer.add()
                noise_layer.update(name=noise_name,
                                   type='DummyData',
                                   top=[noise_name])
                noise_param = noise_layer.dummy_data_param
                noise_param.update(data_filler=[dict(type='gaussian')],
                                   shape=[dict(dim=[batch_size, n_latent])])
                latent_noise = noise_name

                mult_name = '{}_latent_std_noise'.format(encoder_type)
                mult_layer = net.layer.add()
                mult_layer.update(name=mult_name,
                                  type='Eltwise',
                                  bottom=[latent_noise, latent_std],
                                  top=[mult_name])
                mult_layer.eltwise_param.operation = caffe.params.Eltwise.PROD
                latent_std_noise = mult_name

                add_name = '{}_latent_sample'.format(encoder_type)
                add_layer = net.layer.add()
                add_layer.update(name=add_name,
                                 type='Eltwise',
                                 bottom=[latent_std_noise, latent_mean],
                                 top=[add_name])
                add_layer.eltwise_param.operation = caffe.params.Eltwise.SUM
                curr_top = add_name

                # KL-divergence
                mult_name = '{}_latent_mean2'.format(encoder_type)
                mult_layer = net.layer.add()
                mult_layer.update(name=mult_name,
                                  type='Eltwise',
                                  bottom=[latent_mean, latent_mean],
                                  top=[mult_name])
                mult_layer.eltwise_param.operation = caffe.params.Eltwise.PROD
                latent_mean2 = mult_name

                mult_name = '{}_latent_var'.format(encoder_type)
                mult_layer = net.layer.add()
                mult_layer.update(name=mult_name,
                                  type='Eltwise',
                                  bottom=[latent_std, latent_std],
                                  top=[mult_name])
                mult_layer.eltwise_param.operation = caffe.params.Eltwise.PROD
                latent_var = mult_name

                const_name = '{}_latent_one'.format(encoder_type)
                const_layer = net.layer.add()
                const_layer.update(name=const_name,
                                   type='DummyData',
                                   top=[const_name])
                const_param = const_layer.dummy_data_param
                const_param.update(data_filler=[dict(type='constant', value=1.0)],
                                   shape=[dict(dim=[batch_size, n_latent])])
                latent_one = const_name

                add_name = '{}_latent_kldiv'.format(encoder_type)
                add_layer = net.layer.add()
                add_layer.update(name=add_name,
                                 type='Eltwise',
                                 bottom=[latent_mean2, latent_var, latent_log_std, latent_one],
                                 top=[add_name])
                add_param = add_layer.eltwise_param
                add_param.update(operation=caffe.params.Eltwise.SUM,
                                 coeff=[0.5, 0.5, -1.0, -0.5])
                latent_kldiv = add_name

                sum_name = 'kldiv_loss' # TODO handle multiple Kldiv losses
                sum_layer = net.layer.add()
                sum_layer.update(name=sum_name,
                                 type='Reduction',
                                 bottom=[latent_kldiv],
                                 top=[sum_name],
                                 loss_weight=[1.0/batch_size])
                sum_layer.reduction_param.operation = caffe.params.Reduction.SUM

            else:

                fc_name = '{}_latent_fc'.format(encoder_type)
                fc_layer = net.layer.add()
                fc_layer.update(name=fc_name,
                                type='InnerProduct',
                                bottom=[curr_top],
                                top=[fc_name])
                fc_param = fc_layer.inner_product_param
                fc_param.update(num_output=n_latent,
                                weight_filler=dict(type='xavier'))
                curr_top = fc_name

            encoder_tops.append(curr_top)

    if len(encoder_tops) > 1: # concat latent vectors

        concat_name = 'latent_concat'
        concat_layer = net.layer.add()
        concat_layer.update(name=concat_name,
                            type='Concat',
                            bottom=encoder_tops,
                            top=[concat_name])
        concat_layer.concat_param.axis = 1
        curr_top = concat_name

    latent_top = curr_top

    # decoder(s)
    decoder_tops = []
    for d in decoders:

        decoder_type = {'d':'data', 'r':'rec', 'l':'lig'}[d]
        curr_top = latent_top
        label_top = decoder_type
        label_n_filters = n_channels_map[decoder_type]

        fc_name = '{}_latent_defc'.format(decoder_type)
        fc_layer = net.layer.add()
        fc_layer.update(name=fc_name,
                        type='InnerProduct',
                        bottom=[curr_top],
                        top=[fc_name])
        fc_param = fc_layer.inner_product_param
        fc_param.update(num_output=dec_init_n_filters*dec_init_dim**3,
                        weight_filler=dict(type='xavier'))

        relu_name = '{}_latent_derelu'.format(decoder_type)
        relu_layer = net.layer.add()
        relu_layer.update(name=relu_name,
                          type='ReLU',
                          bottom=[fc_name],
                          top=[fc_name])
        relu_layer.relu_param.negative_slope = 0.1 * leaky_relu
        relu_layer.relu_param.engine = caffe.params.ReLU.CAFFE
        curr_top = fc_name

        reshape_name = '{}_latent_reshape'.format(decoder_type)
        reshape_layer = net.layer.add()
        reshape_layer.update(name=reshape_name,
                             type='Reshape',
                             bottom=[curr_top],
                             top=[reshape_name])
        reshape_param = reshape_layer.reshape_param
        reshape_param.shape.update(dim=[batch_size, dec_init_n_filters,
                                        dec_init_dim, dec_init_dim, dec_init_dim])
        curr_top = reshape_name

        curr_n_filters = dec_init_n_filters
        curr_dim = dec_init_dim

        for i in reversed(range(n_levels)):

            if i < n_levels-1: # upsample before convolution

                depool_name = '{}_level{}_depool'.format(decoder_type, i)
                depool_layer = net.layer.add()
                depool_layer.update(name=depool_name,
                                    bottom=[curr_top],
                                    top=[depool_name])

                if depool_type == 'c': # deconvolution with stride 2

                    depool_layer.type = 'Deconvolution'
                    depool_param = depool_layer.convolution_param
                    depool_param.update(num_output=curr_n_filters,
                                        group=curr_n_filters,
                                        weight_filler=dict(type='xavier'),
                                        engine=caffe.params.Convolution.CAFFE)

                elif depool_type == 'n': # nearest-neighbor interpolation

                    depool_layer.type = 'Deconvolution'
                    depool_layer.update(param=[dict(lr_mult=0.0, decay_mult=0.0)])
                    depool_param = depool_layer.convolution_param
                    depool_param.update(num_output=curr_n_filters,
                                        group=curr_n_filters,
                                        weight_filler=dict(type='constant', value=1.0),
                                        bias_term=False,
                                        engine=caffe.params.Convolution.CAFFE)

                curr_top = depool_name
                
                pool_factor = pool_factors.pop(-1)
                depool_param.update(kernel_size=[pool_factor], stride=[pool_factor], pad=[0])
                curr_dim = int(pool_factor*curr_dim)
                next_n_filters = int(curr_n_filters//width_factor)

            for j in range(conv_per_level): # convolutions

                last_conv = i == 0 and j+1 == conv_per_level

                if last_conv:
                    next_n_filters = n_channels_map[label_top]

                deconv_name = '{}_level{}_deconv{}'.format(decoder_type, i, j)
                deconv_layer = net.layer.add()
                deconv_layer.update(name=deconv_name,
                                    type='Deconvolution',
                                    bottom=[curr_top],
                                    top=[deconv_name])

                deconv_param = deconv_layer.convolution_param
                deconv_param.update(num_output=next_n_filters,
                                    kernel_size=[conv_kernel_size],
                                    stride=[1],
                                    pad=[conv_kernel_size//2],
                                    weight_filler=dict(type='xavier'),
                                    engine=caffe.params.Convolution.CAFFE)

                derelu_name = '{}_level{}_derelu{}'.format(decoder_type, i, j)
                derelu_layer = net.layer.add()
                derelu_layer.update(name=derelu_name,
                                    type='ReLU',
                                    bottom=[deconv_name],
                                    top=[deconv_name])
                derelu_layer.relu_param.negative_slope = 0.1 * leaky_relu * ~last_conv

                curr_top = deconv_name
                curr_n_filters = next_n_filters

        pred_top = curr_top

        # output
        id_name = label_top + '_gen'
        id_layer = net.layer.add()
        id_layer.update(name=id_name,
                        type='Power',
                        bottom=[pred_top],
                        top=[id_name])

        # loss
        if 'e' in loss_types:

            loss_name = 'L2_loss'
            loss_layer = net.layer.add()
            loss_layer.update(name=loss_name,
                              type='EuclideanLoss',
                              bottom=[pred_top, label_top],
                              top=[loss_name],
                              loss_weight=[1.0])

        if 'a' in loss_types:

            diff_name = 'diff'
            diff_layer = net.layer.add()
            diff_layer.update(name=diff_name,
                              type='Eltwise',
                              bottom=[pred_top, label_top],
                              top=[diff_name])
            diff_param = diff_layer.eltwise_param
            diff_param.update(operation=caffe.params.Eltwise.SUM,
                              coeff=[-1.0, 1.0])

            loss_name = 'L1_loss'
            loss_layer = net.layer.add()
            loss_layer.update(name=loss_name,
                              type='Reduction',
                              bottom=[diff_name],
                              top=[loss_name],
                              loss_weight=[1.0])
            loss_param = loss_layer.reduction_param
            loss_param.update(operation=caffe.params.Reduction.ASUM)

        if 'c' in loss_types:

            loss_name = 'chan_L2_loss'
            loss_layer = net.layer.add()
            loss_layer.update(name=loss_name,
                              type='Python',
                              bottom=[pred_top, label_top],
                              top=[loss_name],
                              loss_weight=[1.0])
            loss_param = loss_layer.python_param
            loss_param.update(module='channel_euclidean_loss_layer',
                              layer='ChannelEuclideanLossLayer')

        if 'm' in loss_types:

            loss_name = 'mask_L2_loss'
            loss_layer = net.layer.add()
            loss_layer.update(name=loss_name,
                              type='Python',
                              bottom=[pred_top, label_top],
                              top=[loss_name],
                              loss_weight=[0.0])
            loss_param = loss_layer.python_param
            loss_param.update(module='masked_euclidean_loss_layer',
                              layer='MaskedEuclideanLossLayer')

    return net


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
