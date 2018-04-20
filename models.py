import itertools
import caffe_util
from caffe import TRAIN, TEST, params


def make_model(encode_type, data_dim, resolution, n_levels, conv_per_level,
               n_filters, width_factor, loss_types='', molgrid_data=True,
               batch_size=50, conv_kernel_size=3, pool_type='a', depool_type='n'):

    assert encode_type in ['a', 'c']
    assert pool_type in ['c', 'm', 'a']
    assert depool_type in ['c', 'n']

    n_rec_channels = 16
    n_lig_channels = 19
    n_channels = n_rec_channels + n_lig_channels

    net = caffe_util.NetParameter()

    # input
    if molgrid_data:
        for training in [True, False]:

            data_layer = net.layer.add()
            data_layer.update(name='data',
                              type='MolGridData',
                              top=['data', 'label', 'aff'],
                              include=[dict(phase=TRAIN if training else TEST)])

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
                              random_translate=training*2.0)

        net.layer.add().update(name='no_label', type='Silence', bottom=['label'])
        net.layer.add().update(name='no_aff', type='Silence', bottom=['aff'])

        if encode_type == 'c':

            slice_layer = net.layer.add()
            slice_layer.update(name='slice_rec_lig',
                               type='Slice',
                               bottom=['data'],
                               top=['rec', 'lig'],
                               slice_param=dict(axis=1, slice_point=[n_rec_channels]))
    else:
        if encode_type == 'a':

            data_layer = net.layer.add()
            data_layer.update(name='data',
                              type='Input',
                              top=['data'])
            data_layer.input_param.shape.update(dim=[batch_size, n_channels,
                                                     data_dim, data_dim, data_dim])

        elif encode_type == 'c':

            rec_layer = net.layer.add()
            rec_layer.update(name='rec',
                             type='Input',
                             top=['rec'])
            rec_layer.input_param.shape.update(dim=[batch_size, n_rec_channels,
                                                    data_dim, data_dim, data_dim])

            lig_layer = net.layer.add()
            lig_layer.update(name='lig',
                             type='Input',
                             top=['lig'])
            lig_layer.input_param.shape.update(dim=[batch_size, n_lig_channels,
                                                    data_dim, data_dim, data_dim])

    if encode_type == 'a':
        curr_top = 'data'
        curr_n_filters = n_channels
        label_top = 'data'
        label_n_filters = n_channels

    elif encode_type == 'c':
        curr_top = 'rec'
        curr_n_filters = n_rec_channels
        label_top = 'lig'
        label_n_filters = n_lig_channels

    curr_dim = data_dim
    next_n_filters = n_filters

    # encoder
    pool_factors = []
    for i in range(n_levels):

        if i > 0: # pool before convolution

            pool_name = 'level{}_pool'.format(i)
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
                pool_param.pool = params.Pooling.MAX

            elif pool_type == 'a': # average pooling

                pool_layer.type = 'Pooling'
                pool_param = pool_layer.pooling_param
                pool_param.pool = params.Pooling.AVE

            for pool_factor in [2, 3, 5, curr_dim]:
                if curr_dim % pool_factor == 0:
                    break
            pool_factors.append(pool_factor)
            pool_param.update(kernel_size=[pool_factor], stride=[pool_factor], pad=[0])

            curr_top = pool_name
            curr_dim /= pool_factor
            next_n_filters *= width_factor
        
        for j in range(conv_per_level): # convolutions

            conv_name = 'level{}_conv{}'.format(i, j)
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

            relu_name = 'level{}_relu{}'.format(i, j)
            relu_layer = net.layer.add()
            relu_layer.update(name=relu_name,
                              type='ReLU',
                              bottom=[conv_name],
                              top=[conv_name])
            relu_layer.relu_param.negative_slope = 0.0

            curr_top = conv_name
            curr_n_filters = next_n_filters

    print(pool_factors)

    # decoder
    for i in reversed(range(n_levels)):

        if i < n_levels-1: # upsample before convolution

            depool_name = 'level{}_depool'.format(i)
            depool_layer = net.layer.add()
            depool_layer.update(name=depool_name,
                                bottom=[curr_top],
                                top=[depool_name])

            if depool_type == 'c': # deconvolution with stride 2

                depool_layer.type = 'Deconvolution'
                depool_param = depool_layer.convolution_param
                depool_param.update(num_output=curr_n_filters,
                                    group=curr_n_filters,
                                    weight_filler=dict(type='xavier'))

            elif depool_type == 'n': # nearest-neighbor interpolation

                depool_layer.type = 'Deconvolution'
                depool_layer.update(param=[dict(lr_mult=0.0, decay_mult=0.0)])
                depool_param = depool_layer.convolution_param
                depool_param.update(num_output=curr_n_filters,
                                    group=curr_n_filters,
                                    weight_filler=dict(type='constant', value=1.0),
                                    bias_term=False)

            curr_top = depool_name
            
            pool_factor = pool_factors.pop(-1)
            depool_param.update(kernel_size=[pool_factor], stride=[pool_factor], pad=[0])
            curr_dim *= pool_factor

            next_n_filters /= width_factor

        for j in range(conv_per_level): # convolutions

            last_conv = i == 0 and j+1 == conv_per_level

            if last_conv:
                next_n_filters = label_n_filters

            deconv_name = 'level{}_deconv{}'.format(i, j)
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
                                weight_filler=dict(type='xavier'))

            derelu_name = 'level{}_derelu{}'.format(i, j)
            derelu_layer = net.layer.add()
            derelu_layer.update(name=derelu_name,
                                type='ReLU',
                                bottom=[deconv_name],
                                top=[deconv_name])
            derelu_layer.relu_param.negative_slope = 0.0

            curr_top = deconv_name
            curr_n_filters = next_n_filters

    pred_top = curr_top

    # loss
    if 'e' in loss_types:

        loss_name = 'l2_loss'
        loss_layer = net.layer.add()
        loss_layer.update(name=loss_name,
                          type='EuclideanLoss',
                          bottom=[pred_top, label_top],
                          top=[loss_name],
                          loss_weight=[1.0])

    return net


def keyword_product(**kwargs):
    for values in itertools.product(*kwargs.itervalues()):
        yield dict(itertools.izip(kwargs.iterkeys(), values))


def make_model_grid(name_format, **grid_kwargs):
    for kwargs in keyword_product(**grid_kwargs):
        yield name_format.format(**kwargs), make_model(**kwargs)


if __name__ == '__main__':

    if False:
        name_format = '{encode_type}e12_{data_dim}_{resolution}_{n_levels}_{conv_per_level}' \
                    + '_{n_filters}_{width_factor}_{loss_types}'

        model_grid = make_model_grid(name_format,
                                     encode_type=['c', 'a'],
                                     data_dim=[24],
                                     resolution=[0.5, 1.0],
                                     n_levels=[2, 3],
                                     conv_per_level=[2, 3],
                                     n_filters=[16, 32, 64],
                                     width_factor=[1, 2],
                                     loss_types=['e'])

        for model_name, net_param in model_grid:
            net_param.to_prototxt(model_name + '.model')

    net_param = make_model('c', 24, 1.0, 5, 1, 16, 2, 'e')
    net = caffe_util.Net.from_param(net_param, phase=TRAIN)
    net.print_params()


