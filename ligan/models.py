import sys
import numpy as np
from scipy import stats
import torch
from torch import nn
from .interpolation import Interpolation


# mapping of unpool_types to Upsample modes
unpool_type_map = dict(
    n='nearest',
    t='trilinear',
)


def as_list(obj):
    return obj if isinstance(obj, list) else [obj]


def reduce_list(obj):
    return obj[0] if isinstance(obj, list) and len(obj) == 1 else obj


def is_positive_int(x):
    return isinstance(x, int) and x > 0


def get_n_params(model):
    total = 0
    for p in list(model.parameters()):
        n = 1
        for dim in p.shape:
            n *= dim
        total += n
    return total


def caffe_init_weights(module):
    '''
    Xavier initialization with fan-in variance
    norm mode, as implemented in caffe.
    '''
    if isinstance(module, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
        fan_in = nn.init._calculate_correct_fan(module.weight, 'fan_in')
        scale = np.sqrt(3 / fan_in)
        nn.init.uniform_(module.weight, -scale, scale)
        nn.init.constant_(module.bias, 0)


def compute_grad_norm(model):
    '''
    Compute the L2 norm of the gradient
    on model parameters.
    '''
    grad_norm2 = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        grad_norm2 += (p.grad.data**2).sum().item()
    return grad_norm2**(1/2)


def clip_grad_norm(model, max_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def sample_latent(
    batch_size,
    n_latent,
    means=None,
    log_stds=None,
    var_factor=1.0,
    post_factor=1.0,
    truncate=None,
    z_score=None,
    device='cuda',
):
    '''
    Draw batch_size latent vectors of size n_latent
    from a standard normal distribution (the prior)
    and reparameterize them using the posterior pa-
    rameters (means and log_stds), if provided.

    The standard deviation of the latent distribution
    is scaled by var_factor.

    If posterior parameters are provided, they are
    linearly interpolated with the prior parameters
    according to post_factor, where post_factor=1.0
    is purely posterior and 0.0 is purely prior.

    If truncate is provided, samples are instead drawn
    from a normal distribution truncated at that value.

    If z_score is provided, the magnitude of each
    vector is normalized and then scaled by z_score.
    '''
    assert batch_size is not None, batch_size

    # draw samples from standard normal distribution
    if not truncate:
        #print('Drawing latent samples from normal distribution')
        latents = torch.randn((batch_size, n_latent), device=device)
    else:
        #print('Drawing latent samples from truncated normal distribution')
        latents = torch.as_tensor(stats.truncnorm.rvs(
            a=-truncate,
            b=truncate,
            size=(batch_size, n_latent)
        ))

    if z_score not in {None, False}:
        # normalize and scale by z_score
        #  CAUTION: don't know how applicable this is in high-dims
        #print('Normalizing and scaling latent samples')
        latents = latents / latents.norm(dim=1, keepdim=True) * z_score

    #print(f'var_factor = {var_factor}, post_factor = {post_factor}')

    if log_stds is not None: # posterior stds
        stds = torch.exp(log_stds)

        # interpolate b/tw posterior and prior
        #   post_factor*stds + (1-post_factor)*1
        stds = post_factor*stds + (1-post_factor)

        # scale by standard deviation
        latents *= stds

    latents *= var_factor

    if means is not None:

        # interpolate b/tw posterior and prior
        #   post_factor*means + (1-post_factor)*0
        means = post_factor*means

        # shift by mean
        latents += means

    return latents


class Conv3DReLU(nn.Sequential):
    '''
    A 3D convolutional layer followed by leaky ReLU.

    Batch normalization can be applied either before
    (batch_norm=1) or after (batch_norm=2) the ReLU.

    Spectral normalization is applied by indicating
    the number of power iterations (spectral_norm).
    '''
    conv_type = nn.Conv3d

    def __init__(
        self,
        n_channels_in,
        n_channels_out,
        kernel_size=3,
        relu_leak=0.1,
        batch_norm=False,
        spectral_norm=False,
    ):
        modules = [
            self.conv_type(
                in_channels=n_channels_in,
                out_channels=n_channels_out,
                kernel_size=kernel_size,
                padding=kernel_size//2,
            ),
            nn.LeakyReLU(
                negative_slope=relu_leak,
                inplace=True,
            )
        ]

        if batch_norm > 0: # value indicates order wrt conv and relu
            modules.insert(batch_norm, nn.BatchNorm3d(n_channels_out))

        if spectral_norm > 0: # value indicates num power iterations
            modules[0] = nn.utils.spectral_norm(
                modules[0], n_power_iterations=spectral_norm
            )

        super().__init__(*modules)


class TConv3DReLU(Conv3DReLU):
    '''
    A 3D transposed convolution layer and leaky ReLU.

    Batch normalization can be applied either before
    (batch_norm=1) or after (batch_norm=2) the ReLU.

    Spectral normalization is applied by indicating
    the number of power iterations (spectral_norm).
    '''
    conv_type = nn.ConvTranspose3d


class Conv3DBlock(nn.Module):
    '''
    A sequence of n_convs ConvReLUs with the same settings.
    '''
    conv_type = Conv3DReLU

    def __init__(
        self,
        n_convs,
        n_channels_in,
        n_channels_out,
        block_type='c',
        growth_rate=8,
        bottleneck_factor=0,
        debug=False,
        **kwargs
    ):
        super().__init__()

        assert block_type in {'c', 'r', 'd'}, block_type
        self.residual = (block_type == 'r')
        self.dense = (block_type == 'd')

        if self.residual:
            self.init_skip_conv(
                n_channels_in=n_channels_in,
                n_channels_out=n_channels_out,
                **kwargs
            )

        if self.dense:
            self.init_final_conv(
                n_channels_in=n_channels_in,
                n_convs=n_convs,
                growth_rate=growth_rate,
                n_channels_out=n_channels_out,
                **kwargs
            )
            n_channels_out = growth_rate

        self.init_conv_sequence(
            n_convs=n_convs,
            n_channels_in=n_channels_in,
            n_channels_out=n_channels_out,
            bottleneck_factor=bottleneck_factor, 
            **kwargs
        )

    def init_skip_conv(
        self, n_channels_in, n_channels_out, kernel_size, **kwargs
    ):
        if n_channels_out != n_channels_in:

            # 1x1x1 conv to map input to output channels
            self.skip_conv = self.conv_type(
                n_channels_in=n_channels_in,
                n_channels_out=n_channels_out,
                kernel_size=1,
                **kwargs
            )
        else:
            self.skip_conv = nn.Identity()

    def init_final_conv(
        self,
        n_channels_in,
        n_convs,
        growth_rate,
        n_channels_out,
        kernel_size,
        **kwargs
    ):
        # 1x1x1 final "compression" convolution
        self.final_conv = self.conv_type(
            n_channels_in=n_channels_in + n_convs*growth_rate,
            n_channels_out=n_channels_out,
            kernel_size=1,
            **kwargs
        )

    def bottleneck_conv(
        self,
        n_channels_in,
        n_channels_bn,
        n_channels_out,
        kernel_size,
        **kwargs
    ):
        assert n_channels_bn > 0, \
            (n_channels_in, n_channels_bn, n_channels_out)

        return nn.Sequential(
            self.conv_type(
                n_channels_in=n_channels_in,
                n_channels_out=n_channels_bn,
                kernel_size=1,
                **kwargs,
            ),
            self.conv_type(
                n_channels_in=n_channels_bn,
                n_channels_out=n_channels_bn,
                kernel_size=kernel_size,
                **kwargs
            ),
            self.conv_type(
                n_channels_in=n_channels_bn,
                n_channels_out=n_channels_out,
                kernel_size=1,
                **kwargs,
            )
        )

    def init_conv_sequence(
        self,
        n_convs,
        n_channels_in,
        n_channels_out,
        bottleneck_factor,
        **kwargs
    ):
        self.conv_modules = []
        for i in range(n_convs):

            if bottleneck_factor: # bottleneck convolution
                conv = self.bottleneck_conv(
                    n_channels_in=n_channels_in,
                    n_channels_bn=n_channels_in//bottleneck_factor,
                    n_channels_out=n_channels_out,
                    **kwargs
                )
            else: # single convolution
                conv = self.conv_type(
                    n_channels_in=n_channels_in,
                    n_channels_out=n_channels_out,
                    **kwargs
                )
            self.conv_modules.append(conv)
            self.add_module(str(i), conv)

            if self.dense:
                n_channels_in += n_channels_out
            else:
                n_channels_in = n_channels_out

    def __len__(self):
        return len(self.conv_modules)

    def forward(self, inputs):

        if not self.conv_modules:
            return inputs

        if self.dense:
            all_inputs = [inputs]

        # convolution sequence
        for i, f in enumerate(self.conv_modules):
            
            if self.residual:
                identity = self.skip_conv(inputs) if i == 0 else inputs
                outputs = f(inputs) + identity
            else:
                outputs = f(inputs)

            if self.dense:
                all_inputs.append(outputs)
                inputs = torch.cat(all_inputs, dim=1)
            else:
                inputs = outputs

        if self.dense:
            outputs = self.final_conv(inputs)

        return outputs


class TConv3DBlock(Conv3DBlock):
    '''
    A sequence of n_convs TConvReLUs with the same settings.
    '''
    conv_type = TConv3DReLU


class Pool3D(nn.Sequential):
    '''
    A layer that decreases 3D spatial dimensions,
    either by max pooling (pool_type=m), average
    pooling (pool_type=a), or strided convolution
    (pool_type=c).
    '''
    def __init__(self, n_channels, pool_type, pool_factor):

        if pool_type == 'm':
            pool = nn.MaxPool3d(
                kernel_size=pool_factor,
                stride=pool_factor,
            )

        elif pool_type == 'a':
            pool = nn.AvgPool3d(
                kernel_size=pool_factor,
                stride=pool_factor,
            )

        elif pool_type == 'c':
            pool = nn.Conv3d(
                in_channels=n_channels,
                out_channels=n_channels,
                groups=n_channels,
                kernel_size=pool_factor,
                stride=pool_factor,
            )

        else:
            raise ValueError('unknown pool_type ' + repr(pool_type))

        super().__init__(pool)


class Unpool3D(nn.Sequential):
    '''
    A layer that increases the 3D spatial dimensions,
    either by nearest neighbor (unpool_type=n), tri-
    linear interpolation (unpool_type=t), or strided
    transposed convolution (unpool_type=c).
    '''
    def __init__(self, n_channels, unpool_type, unpool_factor):

        if unpool_type in unpool_type_map:
            
            unpool = nn.Upsample(
                scale_factor=unpool_factor,
                mode=unpool_type_map[unpool_type],
            )

        elif unpool_type == 'c':
            
            unpool = nn.ConvTranspose3d(
                in_channels=n_channels,
                out_channels=n_channels,
                groups=n_channels,
                kernel_size=unpool_factor,
                stride=unpool_factor,
            )

        else:
            raise ValueError('unknown unpool_type ' + repr(unpool_type))

        super().__init__(unpool)


class Reshape(nn.Module):
    '''
    A layer that reshapes the input.
    '''
    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(shape)

    def __repr__(self):
        return 'Reshape(shape={})'.format(self.shape)

    def forward(self, x):
        return x.reshape(self.shape)


class Grid2Vec(nn.Sequential):
    '''
    A fully connected layer applied to a
    flattened version of the input, for
    transforming from grids to vectors.
    '''
    def __init__(
        self, in_shape, n_output, activ_fn=None, spectral_norm=0
    ):
        n_input = np.prod(in_shape)
        modules = [
            Reshape(shape=(-1, n_input)),
            nn.Linear(n_input, n_output)
        ]

        if activ_fn:
            modules.append(activ_fn)

        if spectral_norm > 0:
            modules[1] = nn.utils.spectral_norm(
                modules[1], n_power_iterations=spectral_norm
            )

        super().__init__(*modules)


class Vec2Grid(nn.Sequential):
    '''
    A fully connected layer followed by
    reshaping the output, for transforming
    from vectors to grids.
    '''
    def __init__(
        self, n_input, out_shape, relu_leak, batch_norm, spectral_norm
    ):
        n_output = np.prod(out_shape)
        modules = [
            nn.Linear(n_input, n_output),
            Reshape(shape=(-1, *out_shape)),
            nn.LeakyReLU(negative_slope=relu_leak, inplace=True),
        ]

        if batch_norm > 0:
            modules.insert(batch_norm+1, nn.BatchNorm3d(out_shape[0]))

        if spectral_norm > 0:
            modules[0] = nn.utils.spectral_norm(
                modules[0], n_power_iterations=spectral_norm
            )

        super().__init__(*modules)


class GridEncoder(nn.Module):
    '''
    A sequence of 3D convolution blocks and
    pooling layers, followed by one or more
    fully connected output tasks.
    '''
    # TODO reimplement the following:
    # - self-attention
    # - batch discrimination
    
    def __init__(
        self,
        n_channels,
        grid_size=48,
        n_filters=32,
        width_factor=2,
        n_levels=4,
        conv_per_level=3,
        kernel_size=3,
        relu_leak=0.1,
        batch_norm=0,
        spectral_norm=0,
        pool_type='a',
        pool_factor=2,
        n_output=1,
        output_activ_fn=None,
        init_conv_pool=False,
        block_type='c',
        growth_rate=8,
        bottleneck_factor=0,
        debug=False,
    ):
        super().__init__()
        self.debug = debug

        # sequence of convs and/or pools
        self.grid_modules = []

        # track changing grid dimensions
        self.n_channels = n_channels
        self.grid_size = grid_size

        if init_conv_pool:

            self.add_conv3d(
                name='init_conv',
                n_filters=n_filters,
                kernel_size=kernel_size+2,
                relu_leak=relu_leak,
                batch_norm=batch_norm,
                spectral_norm=spectral_norm,
            )
            self.add_pool3d(
                name='init_pool',
                pool_type=pool_type,
                pool_factor=pool_factor
            )
            n_filters *= width_factor

        for i in range(n_levels):

            if i > 0: # downsample between conv blocks
                self.add_pool3d(
                    name='level'+str(i)+'_pool',
                    pool_type=pool_type,
                    pool_factor=pool_factor
                )
                n_filters *= width_factor
 
            self.add_conv3d_block(
                name='level'+str(i),
                n_convs=conv_per_level,
                n_filters=n_filters,
                kernel_size=kernel_size,
                relu_leak=relu_leak,
                batch_norm=batch_norm,
                spectral_norm=spectral_norm,
                block_type=block_type,
                growth_rate=growth_rate,
                bottleneck_factor=bottleneck_factor,
                debug=debug,
            )

        # fully-connected outputs
        n_output = as_list(n_output)
        assert n_output and all(n_o > 0 for n_o in n_output)

        output_activ_fn = as_list(output_activ_fn)
        if len(output_activ_fn) == 1:
            output_activ_fn *= len(n_output)
        assert len(output_activ_fn) == len(n_output)

        self.n_tasks = len(n_output)
        self.task_modules = []

        for i, (n_output_i, activ_fn_i) in enumerate(
            zip(n_output, output_activ_fn)
        ):
            self.add_grid2vec(
                name='fc'+str(i),
                n_output=n_output_i,
                activ_fn=activ_fn_i,
                spectral_norm=spectral_norm
            )

    def print(self, *args, **kwargs):
        if self.debug:
            print('DEBUG', *args, **kwargs, file=sys.stderr)

    def add_conv3d(self, name, n_filters, **kwargs):
        conv = Conv3DReLU(
            n_channels_in=self.n_channels,
            n_channels_out=n_filters,
            **kwargs
        )
        self.add_module(name, conv)
        self.grid_modules.append(conv)
        self.n_channels = n_filters
        self.print(name, self.n_channels, self.grid_size)

    def add_pool3d(self, name, pool_factor, **kwargs):
        assert self.grid_size % pool_factor == 0, \
            'cannot pool remaining spatial dims ({} % {})'.format(
                self.grid_size, pool_factor
            )
        pool = Pool3D(
            n_channels=self.n_channels,
            pool_factor=pool_factor,
            **kwargs
        )
        self.add_module(name, pool)
        self.grid_modules.append(pool)
        self.grid_size //= pool_factor
        self.print(name, self.n_channels, self.grid_size)

    def add_conv3d_block(self, name, n_filters, **kwargs):
        conv_block = Conv3DBlock(
            n_channels_in=self.n_channels,
            n_channels_out=n_filters,
            **kwargs
        )
        self.add_module(name, conv_block)
        self.grid_modules.append(conv_block)
        self.n_channels = n_filters
        self.print(name, self.n_channels, self.grid_size)

    def add_grid2vec(self, name, **kwargs):
        fc = Grid2Vec(
            in_shape=(self.n_channels,) + (self.grid_size,)*3,
            **kwargs
        )
        self.add_module(name, fc)
        self.task_modules.append(fc)
        self.print(name, self.n_channels, self.grid_size)

    def forward(self, inputs):

        # conv-pool sequence
        conv_features = []
        for f in self.grid_modules:

            outputs = f(inputs)
            self.print(inputs.shape, '->', f, '->', outputs.shape)

            if not isinstance(f, Pool3D):
                conv_features.append(outputs)
            inputs = outputs

        # fully-connected outputs
        outputs = [f(inputs) for f in self.task_modules]
        outputs_shape = [o.shape for o in outputs]
        self.print(inputs.shape, '->', self.task_modules, '->', outputs_shape)

        return reduce_list(outputs), conv_features


# this is basically just an alias
class Discriminator(GridEncoder):
    pass


class GridDecoder(nn.Module):
    '''
    A fully connected layer followed by a
    sequence of 3D transposed convolution
    blocks and unpooling layers.
    '''
    # TODO re-implement the following:
    # - self-attention
    # - gaussian output

    def __init__(
        self,
        n_input,
        grid_size,
        n_channels,
        width_factor,
        n_levels,
        tconv_per_level,
        kernel_size,
        relu_leak,
        batch_norm,
        spectral_norm,
        unpool_type,
        unpool_factor,
        n_channels_out,
        final_unpool=False,
        skip_connect=False,
        block_type='c',
        growth_rate=8,
        bottleneck_factor=0,
        debug=False,
    ):
        super().__init__()
        self.skip_connect = bool(skip_connect)
        self.debug = debug

        # first fc layer maps to initial grid shape
        self.fc_modules = []
        self.n_input = n_input
        self.add_vec2grid(
            name='fc',
            n_input=n_input,
            n_channels=n_channels,
            grid_size=grid_size,
            relu_leak=relu_leak,
            batch_norm=batch_norm,
            spectral_norm=spectral_norm,
        )
        n_filters = n_channels

        self.grid_modules = []
        for i in reversed(range(n_levels)):

            if i + 1 < n_levels: # unpool between deconv blocks
                unpool_name = 'level' + str(i) + '_unpool'
                self.add_unpool3d(
                    name=unpool_name,
                    unpool_type=unpool_type,
                    unpool_factor=unpool_factor
                )
                n_filters //= width_factor

            if skip_connect:
                n_skip_channels = self.n_channels
                if i < n_levels - 1:
                    n_skip_channels //= width_factor
            else:
                n_skip_channels = 0

            tconv_block_name = 'level' + str(i)
            self.add_tconv3d_block(
                name=tconv_block_name,
                n_convs=tconv_per_level,
                n_filters=n_filters,
                kernel_size=kernel_size,
                relu_leak=relu_leak,
                batch_norm=batch_norm,
                spectral_norm=spectral_norm,
                block_type=block_type,
                growth_rate=growth_rate,
                bottleneck_factor=bottleneck_factor,
                n_skip_channels=n_skip_channels,
                debug=debug,
            )

        if final_unpool:

            self.add_unpool3d(
                name='final_unpool',
                unpool_type=unpool_type,
                unpool_factor=unpool_factor,
            )
            n_skip_channels //= width_factor

            self.add_tconv3d_block(
                name='final_conv',
                n_convs=tconv_per_level,
                n_filters=n_channels_out,
                kernel_size=kernel_size,
                relu_leak=relu_leak,
                batch_norm=batch_norm,
                spectral_norm=spectral_norm,
                block_type=block_type,
                growth_rate=growth_rate,
                bottleneck_factor=bottleneck_factor,
                n_skip_channels=n_skip_channels,
                debug=debug,
            )

        else: # final tconv maps to correct n_output channels

            self.add_tconv3d(
                name='final_conv',
                n_filters=n_channels_out,
                kernel_size=kernel_size,
                relu_leak=relu_leak,
                batch_norm=batch_norm,
                spectral_norm=spectral_norm,
            )

    def print(self, *args, **kwargs):
        if self.debug:
            print('DEBUG', *args, **kwargs, file=sys.stderr)

    def add_vec2grid(self, name, n_channels, grid_size, **kwargs):
        vec2grid = Vec2Grid(
            out_shape=(n_channels,) + (grid_size,)*3,
            **kwargs
        )
        self.add_module(name, vec2grid)
        self.fc_modules.append(vec2grid)
        self.n_channels = n_channels
        self.grid_size = grid_size
        self.print(name, self.n_channels, self.grid_size)

    def add_unpool3d(self, name, unpool_factor, **kwargs):
        unpool = Unpool3D(
            n_channels=self.n_channels,
            unpool_factor=unpool_factor,
            **kwargs
        )
        self.add_module(name, unpool)
        self.grid_modules.append(unpool)
        self.grid_size *= unpool_factor
        self.print(name, self.n_channels, self.grid_size)

    def add_tconv3d(self, name, n_filters, **kwargs):
        tconv = TConv3DReLU(
            n_channels_in=self.n_channels,
            n_channels_out=n_filters,
            **kwargs
        )
        self.add_module(name, tconv)
        self.grid_modules.append(tconv)
        self.n_channels = n_filters
        self.print(name, self.n_channels, self.grid_size)

    def add_tconv3d_block(
        self, name, n_filters, n_skip_channels, **kwargs
    ):
        tconv_block = TConv3DBlock(
            n_channels_in=self.n_channels + n_skip_channels,
            n_channels_out=n_filters,
            **kwargs
        )
        self.add_module(name, tconv_block)
        self.grid_modules.append(tconv_block)
        self.n_channels = n_filters
        self.print(name, self.n_channels, self.grid_size)

    def forward(self, inputs, skip_features=None):

        for f in self.fc_modules:
            outputs = f(inputs)
            self.print(inputs.shape, '->', f, '->', outputs.shape)
            inputs = outputs

        for f in self.grid_modules:

            if self.skip_connect and isinstance(f, TConv3DBlock):
                skip_inputs = skip_features.pop()
                inputs = torch.cat([inputs, skip_inputs], dim=1)
                inputs_shape = [inputs.shape, skip_inputs.shape]
            else:
                inputs_shape = inputs.shape

            outputs = f(inputs)
            self.print(inputs_shape, '->', f, '->', outputs.shape)
            inputs = outputs

        return outputs


class GridGenerator(nn.Sequential):
    '''
    A generative model of 3D grids that can take the form
    of an encoder-decoder architecture (e.g. AE, VAE) or
    a decoder-only architecture (e.g. GAN). The model can
    also have a conditional encoder (e.g. CE, CVAE, CGAN).
    '''
    is_variational = False
    has_input_encoder = False
    has_conditional_encoder = False
    has_stage2 = False

    def __init__(
        self,
        n_channels_in=None,
        n_channels_cond=None,
        n_channels_out=19,
        grid_size=48,
        n_filters=32,
        width_factor=2,
        n_levels=4,
        conv_per_level=3,
        kernel_size=3,
        relu_leak=0.1,
        batch_norm=0,
        spectral_norm=0,
        pool_type='a',
        unpool_type='n',
        pool_factor=2,
        n_latent=1024,
        init_conv_pool=False,
        skip_connect=False,
        block_type='c',
        growth_rate=8,
        bottleneck_factor=0,
        n_samples=0,
        device='cuda',
        debug=False,
    ):
        assert type(self) != GridGenerator, 'GridGenerator is abstract'
        self.debug = debug

        super().__init__()
        self.check_encoder_channels(n_channels_in, n_channels_cond)
        assert is_positive_int(n_channels_out)
        assert is_positive_int(n_latent)

        self.n_channels_in = n_channels_in
        self.n_channels_cond = n_channels_cond
        self.n_channels_out = n_channels_out
        self.n_latent = n_latent

        if self.has_input_encoder:

            if self.is_variational: # means and log_stds
                encoder_output = [n_latent, n_latent]
            else:
                encoder_output = n_latent

            self.input_encoder = GridEncoder(
                n_channels=n_channels_in,
                grid_size=grid_size,
                n_filters=n_filters,
                width_factor=width_factor,
                n_levels=n_levels,
                conv_per_level=conv_per_level,
                kernel_size=kernel_size,
                relu_leak=relu_leak,
                batch_norm=batch_norm,
                spectral_norm=spectral_norm,
                pool_type=pool_type,
                pool_factor=pool_factor,
                n_output=encoder_output,
                init_conv_pool=init_conv_pool,
                block_type=block_type,
                growth_rate=growth_rate,
                bottleneck_factor=bottleneck_factor,
                debug=debug,
            )

        if self.has_conditional_encoder:

            self.conditional_encoder = GridEncoder(
                n_channels=n_channels_cond,
                grid_size=grid_size,
                n_filters=n_filters,
                width_factor=width_factor,
                n_levels=n_levels,
                conv_per_level=conv_per_level,
                kernel_size=kernel_size,
                relu_leak=relu_leak,
                batch_norm=batch_norm,
                spectral_norm=spectral_norm,
                pool_type=pool_type,
                pool_factor=pool_factor,
                n_output=n_latent,
                init_conv_pool=init_conv_pool,
                block_type=block_type,
                growth_rate=growth_rate,
                bottleneck_factor=bottleneck_factor,
                debug=debug,
            )

        n_pools = n_levels - 1 + init_conv_pool

        self.decoder = GridDecoder(
            n_input=self.n_decoder_input,
            grid_size=grid_size // pool_factor**n_pools,
            n_channels=n_filters * width_factor**n_pools,
            width_factor=width_factor,
            n_levels=n_levels,
            tconv_per_level=conv_per_level,
            kernel_size=kernel_size,
            relu_leak=relu_leak,
            batch_norm=batch_norm,
            spectral_norm=spectral_norm,
            unpool_type=unpool_type,
            unpool_factor=pool_factor,
            n_channels_out=n_channels_out,
            final_unpool=init_conv_pool,
            skip_connect=skip_connect,
            block_type=block_type,
            growth_rate=growth_rate,
            bottleneck_factor=bottleneck_factor,
            debug=debug,
        )

        # latent interpolation state
        self.latent_interp = Interpolation(n_samples=n_samples)

        super().to(device)
        self.device = device

    def check_encoder_channels(self, n_channels_in, n_channels_cond):
        if self.has_input_encoder:
            assert is_positive_int(n_channels_in), n_channels_in
        else:
            assert n_channels_in is None, n_channels_in

        if self.has_conditional_encoder:
            assert is_positive_int(n_channels_cond), n_channels_cond
        else:
            assert n_channels_cond is None, n_channels_cond

    @property
    def n_decoder_input(self):
        n = 0
        if self.has_input_encoder or self.is_variational:
            n += self.n_latent
        if self.has_conditional_encoder:
            n += self.n_latent
        return n

    def sample_latent(
        self, batch_size, means=None, log_stds=None, interpolate=False, spherical=False, **kwargs
    ):
        latent_vecs = sample_latent(
            batch_size=batch_size,
            n_latent=self.n_latent,
            means=means,
            log_stds=log_stds,
            device=self.device,
            **kwargs
        )

        if interpolate:
            if not self.latent_interp.is_initialized:
                self.latent_interp.initialize(sample_latent(
                    batch_size=1,
                    n_latent=self.n_latent,
                    device=self.device,
                    **kwargs
                )[0])
            latent_vecs = self.latent_interp(latent_vecs, spherical=spherical)

        return latent_vecs


class AE(GridGenerator):
    is_variational = False
    has_input_encoder = True
    has_conditional_encoder = False

    def forward(self, inputs=None, conditions=None, batch_size=None):

        if inputs is None: # "prior", not expected to work
            in_latents = self.sample_latent(batch_size)
        else: # posterior
            in_latents, _ = self.input_encoder(inputs)

        outputs = self.decoder(inputs=in_latents)
        return outputs, in_latents, None, None


class VAE(GridGenerator):
    is_variational = True
    has_input_encoder = True
    has_conditional_encoder = False

    def forward(self, inputs=None, conditions=None, batch_size=None):

        if inputs is None: # prior
            means, log_stds = None, None
        else: # posterior
            (means, log_stds), _ = self.input_encoder(inputs)

        var_latents = self.sample_latent(batch_size, means, log_stds)
        outputs = self.decoder(inputs=var_latents)
        return outputs, var_latents, means, log_stds


class CE(GridGenerator):
    is_variational = False
    has_input_encoder = False
    has_conditional_encoder = True

    def forward(self, inputs=None, conditions=None, batch_size=None):
        cond_latents, cond_features = self.conditional_encoder(conditions)
        outputs = self.decoder(
            inputs=cond_latents, skip_features=cond_features
        )
        return outputs, cond_latents, None, None


class CVAE(GridGenerator):
    is_variational = True
    has_input_encoder = True
    has_conditional_encoder = True

    def forward(
        self, inputs=None, conditions=None, batch_size=None, **kwargs
    ):
        if inputs is None: # prior
            means, log_stds = None, None
        else: # posterior
            (means, log_stds), _ = self.input_encoder(inputs)

        in_latents = self.sample_latent(batch_size, means, log_stds, **kwargs)
        cond_latents, cond_features = self.conditional_encoder(conditions)
        cat_latents = torch.cat([in_latents, cond_latents], dim=1)

        outputs = self.decoder(
            inputs=cat_latents, skip_features=cond_features
        )
        return outputs, in_latents, means, log_stds


class GAN(GridGenerator):
    is_variational = True # just means we provide noise to decoder
    has_input_encoder = False
    has_conditional_encoder = False

    def forward(self, inputs=None, conditions=None, batch_size=None):
        var_latents = self.sample_latent(batch_size)
        outputs = self.decoder(inputs=var_latents)
        return outputs, var_latents, None, None


class CGAN(GridGenerator):
    is_variational = True # just means we provide noise to decoder
    has_input_encoder = False
    has_conditional_encoder = True

    def forward(self, inputs=None, conditions=None, batch_size=None):
        var_latents = self.sample_latent(batch_size)
        cond_latents, cond_features = self.conditional_encoder(conditions)
        cat_latents = torch.cat([var_latents, cond_latents], dim=1)
        outputs = self.decoder(
            inputs=cat_latents, skip_features=cond_features
        )
        return outputs, var_latents, None, None


# aliases that correspond to solver type
VAEGAN = VAE
CVAEGAN = CVAE


class VAE2(VAE):
    has_stage2 = True
    '''
    This is a module that allows insertion of
    a prior model, aka stage 2 VAE, into an
    existing VAE model, aka a two-stage VAE.
    '''
    def forward2(
        self,
        prior_model,
        inputs=None,
        conditions=None,
        batch_size=None,
        **kwargs,
    ):
        if inputs is None: # prior
            var_latents = means = log_stds = None

        else: # stage-1 posterior
            (means, log_stds), _ = self.input_encoder(inputs)
            var_latents = self.sample_latent(
                batch_size, means, log_stds, **kwargs
            )

        # insert prior model (output is stage-2 posterior or prior)
        gen_latents, _, means2, log_stds2 = prior_model(
            inputs=var_latents, batch_size=batch_size
        )

        outputs = self.decoder(inputs=gen_latents)
        return (
            outputs, var_latents, means, log_stds,
            gen_latents, means2, log_stds2
        )


class CVAE2(CVAE):
    has_stage2 = True
    '''
    Two-stage CVAE.
    '''
    def forward2(
        self,
        prior_model,
        inputs=None,
        conditions=None,
        batch_size=None,
        **kwargs,
    ):
        if inputs is None: # prior
            in_latents = means = log_stds = None

        else: # stage-1 posterior
            (means, log_stds), _ = self.input_encoder(inputs)
            in_latents = self.sample_latent(
                batch_size, means, log_stds, **kwargs
            )

        # insert prior model (output is stage-2 posterior or prior)
        gen_latents, _, means2, log_stds2 = prior_model(
            inputs=in_latents, batch_size=batch_size
        )

        cond_latents, cond_features = self.conditional_encoder(conditions)
        cat_latents = torch.cat([gen_latents, cond_latents], dim=1)

        outputs = self.decoder(
            inputs=cat_latents, skip_features=cond_features
        )
        return (
            outputs, in_latents, means, log_stds,
            gen_latents, means2, log_stds2
        )


class Stage2VAE(nn.Module):

    def __init__(
        self,
        n_input,
        n_h_layers,
        n_h_units,
        n_latent,
        relu_leak=0.1,
        device='cuda'
    ):
        super().__init__()

        # track dimensions for decoder
        n_inputs = []

        modules = []
        for i in range(n_h_layers): # build encoder
            modules.append(nn.Linear(n_input, n_h_units))
            modules.append(nn.LeakyReLU(negative_slope=relu_leak))
            n_inputs.append(n_input)
            n_input = n_h_units

        self.encoder = nn.Sequential(*modules)

        # variational latent space
        self.fc_mean = nn.Linear(n_input, n_latent)
        self.fc_log_std = nn.Linear(n_input, n_latent)

        modules = [ # decoder input fc
            nn.Linear(n_latent, n_input),
            nn.LeakyReLU(negative_slope=relu_leak)
        ]

        for n_output in reversed(n_inputs): # build decoder
            modules.append(nn.Linear(n_input, n_output))
            modules.append(nn.LeakyReLU(negative_slope=relu_leak))
            n_input = n_output

        self.decoder = nn.Sequential(*modules)

        self.n_latent = n_latent
        self.device = device

    def forward(self, inputs=None, batch_size=None):

        if inputs is None: # prior
            means, log_stds = None, None

        else: # posterior
            enc_outputs = self.encoder(inputs)
            means = self.fc_mean(enc_outputs)
            log_stds = self.fc_log_std(enc_outputs)

        var_latents = self.sample_latent(batch_size, means, log_stds)
        outputs = self.decoder(var_latents)
        return outputs, var_latents, means, log_stds

    def sample_latent(
        self, batch_size, means=None, log_stds=None, **kwargs
    ):
        return sample_latent(
            batch_size=batch_size,
            n_latent=self.n_latent,
            means=means,
            log_stds=log_stds,
            device=self.device,
            **kwargs
        )
