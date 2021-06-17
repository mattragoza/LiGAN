import numpy as np
import torch
from torch import nn


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


def initialize_weights(m, caffe=False):
    '''
    Xavier initialization with fan-in variance
    norm mode, as implemented in caffe.
    '''
    if isinstance(m, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
        fan_in = nn.init._calculate_correct_fan(m.weight, 'fan_in')
        if caffe:
            scale = np.sqrt(3 / fan_in)
            nn.init.uniform_(m.weight, -scale, scale)
            nn.init.constant_(m.bias, 0)
        else:
            pass # use default PyTorch initialization


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
        block_type=None,
        **kwargs
    ):
        super().__init__()

        self.conv_modules = []
        for i in range(n_convs):
            conv = self.conv_type(
                n_channels_in=n_channels_in,
                n_channels_out=n_channels_out,
                **kwargs
            )
            self.conv_modules.append(conv)
            self.add_module(str(i), conv)
            n_channels_in = n_channels_out

        self.block_type = block_type

    def __len__(self):
        return len(self.conv_modules)

    def forward(self, inputs):

        if not self.conv_modules:
            return inputs

        identity = inputs # for resnet
        all_inputs = [inputs] # for densenet

        for f in self.conv_modules:
            outputs = f(inputs)

            if self.block_type == 'd': # densenet
                all_inputs.append(outputs)
                inputs = torch.cat(all_inputs, dim=1)
            else:
                inputs = outputs

        if not self.conv_modules:
            return identity

        if self.block_type == 'r': # resnet
            return outputs + identity
        else:
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
    # - densely-connected
    # - batch discrimination
    # - fully-convolutional
    
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
        block_type=None,
    ):
        super().__init__()

        # sequence of convs and/or pools
        self.grid_modules = []

        # track changing grid dimensions
        self.n_channels = n_channels
        self.grid_size = grid_size

        if init_conv_pool:

            self.add_conv3d(
                name='init_conv',
                n_filters=n_filters,
                kernel_size=kernel_size,
                relu_leak=relu_leak,
                batch_norm=batch_norm,
                spectral_norm=spectral_norm,
            )
            self.add_pool3d(
                name='init_pool',
                pool_type=pool_type,
                pool_factor=pool_factor
            )

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

    def add_conv3d(self, name, n_filters, **kwargs):
        conv = Conv3DReLU(
            n_channels_in=self.n_channels,
            n_channels_out=n_filters,
            **kwargs
        )
        self.add_module(name, conv)
        self.grid_modules.append(conv)
        self.n_channels = n_filters

    def add_pool3d(self, name, pool_factor, **kwargs):
        assert self.grid_size % pool_factor == 0, \
            'cannot pool remaining spatial dims'
        pool = Pool3D(
            n_channels=self.n_channels,
            pool_factor=pool_factor,
            **kwargs
        )
        self.add_module(name, pool)
        self.grid_modules.append(pool)
        self.grid_size //= pool_factor

    def add_conv3d_block(self, name, n_filters, **kwargs):
        conv_block = Conv3DBlock(
            n_channels_in=self.n_channels,
            n_channels_out=n_filters,
            **kwargs
        )
        self.add_module(name, conv_block)
        self.grid_modules.append(conv_block)
        self.n_channels = n_filters

    def add_grid2vec(self, name, **kwargs):
        fc = Grid2Vec(
            in_shape=(self.n_channels,) + (self.grid_size,)*3,
            **kwargs
        )
        self.add_module(name, fc)
        self.task_modules.append(fc)

    def forward(self, inputs):

        # conv pool sequence
        conv_features = []
        for f in self.grid_modules:
            outputs = f(inputs)
            if not isinstance(f, Pool3D):
                conv_features.append(outputs)
            inputs = outputs

        # fully-connected outputs
        outputs = [f(inputs) for f in self.task_modules]

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
    # - densely-connected
    # - fully-convolutional
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
        block_type=None,
    ):
        super().__init__()
        self.skip_connect = skip_connect

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
            )

        if final_unpool:
            self.add_unpool3d(
                name='final_unpool',
                unpool_type=unpool_type,
                unpool_factor=unpool_factor,
            )

        # final tconv maps to correct n_output channels
        self.add_tconv3d(
            name='final_conv',
            n_filters=n_channels_out,
            kernel_size=kernel_size,
            relu_leak=relu_leak,
            batch_norm=batch_norm,
            spectral_norm=spectral_norm,
        )

    def add_vec2grid(self, name, n_channels, grid_size, **kwargs):
        vec2grid = Vec2Grid(
            out_shape=(n_channels,) + (grid_size,)*3,
            **kwargs
        )
        self.add_module(name, vec2grid)
        self.fc_modules.append(vec2grid)
        self.n_channels = n_channels
        self.grid_size = grid_size

    def add_unpool3d(self, name, unpool_factor, **kwargs):
        unpool = Unpool3D(
            n_channels=self.n_channels,
            unpool_factor=unpool_factor,
            **kwargs
        )
        self.add_module(name, unpool)
        self.grid_modules.append(unpool)
        self.grid_size *= unpool_factor

    def add_tconv3d(self, name, n_filters, **kwargs):
        tconv = TConv3DReLU(
            n_channels_in=self.n_channels * (1 + self.skip_connect),
            n_channels_out=n_filters,
            **kwargs
        )
        self.add_module(name, tconv)
        self.grid_modules.append(tconv)
        self.n_channels = n_filters

    def add_tconv3d_block(self, name, n_filters, **kwargs):
        tconv_block = TConv3DBlock(
            n_channels_in=self.n_channels * (1 + self.skip_connect),
            n_channels_out=n_filters,
            **kwargs
        )
        self.add_module(name, tconv_block)
        self.grid_modules.append(tconv_block)
        self.n_channels = n_filters

    def forward(self, inputs, conv_features=None):

        for f in self.fc_modules:
            outputs = f(inputs)
            inputs = outputs

        for i, f in enumerate(self.grid_modules):
            if self.skip_connect and not isinstance(f, Unpool3D):
                #print(i, input.shape, conv_features[-i-1].shape)
                inputs = torch.cat([inputs, conv_features[-i-1]], dim=1)
            outputs = f(inputs)
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
        block_type=None,
        device='cuda',
    ):
        assert type(self) != GridGenerator, 'GridGenerator is abstract'

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
            )

        self.skip_connect = skip_connect

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
            block_type=block_type,
        )

        super().to(device)
        self.device = device

    def check_encoder_channels(self, n_channels_in, n_channels_cond):
        if self.has_input_encoder:
            assert is_positive_int(n_channels_in)
        else:
            assert n_channels_in is None

        if self.has_conditional_encoder:
            assert is_positive_int(n_channels_cond)
        else:
            assert n_channels_cond is None

    @property
    def n_decoder_input(self):
        n = 0
        if self.has_input_encoder or self.is_variational:
            n += self.n_latent
        if self.has_conditional_encoder:
            n += self.n_latent
        return n

    def sample_latents(self, batch_size, means=None, log_stds=None):
        latents = torch.randn((batch_size, self.n_latent), device=self.device)
        if log_stds is not None:
            latents *= torch.exp(log_stds)
        if means is not None:
            latents += means
        return latents


class AE(GridGenerator):
    is_variational = False
    has_input_encoder = True
    has_conditional_encoder = False

    def forward(self, inputs=None, conditions=None, batch_size=None):

        if inputs is None: # "prior", not expected to work
            in_latents = self.sample_latents(batch_size)
        else: # posterior
            in_latents, _ = self.input_encoder(inputs)

        return self.decoder(in_latents), in_latents, None, None


class VAE(GridGenerator):
    is_variational = True
    has_input_encoder = True
    has_conditional_encoder = False

    def forward(self, inputs=None, conditions=None, batch_size=None):

        if inputs is None: # prior
            means, log_stds = None, None
        else: # posterior
            (means, log_stds), _ = self.input_encoder(inputs)

        var_latents = self.sample_latents(batch_size, means, log_stds)
        return self.decoder(var_latents), var_latents, means, log_stds


class CE(GridGenerator):
    is_variational = False
    has_input_encoder = False
    has_conditional_encoder = True

    def forward(self, inputs=None, conditions=None, batch_size=None):
        cond_latents, cond_features = self.conditional_encoder(conditions)
        return self.decoder(
            cond_latents, cond_features if self.skip_connect else None
        ), cond_latents, None, None


class CVAE(GridGenerator):
    is_variational = True
    has_input_encoder = True
    has_conditional_encoder = True

    def forward(self, inputs=None, conditions=None, batch_size=None):

        if inputs is None: # prior
            means, log_stds = None, None
        else: # posterior
            (means, log_stds), _ = self.input_encoder(inputs)

        in_latents = self.sample_latents(batch_size, means, log_stds)
        cond_latents, cond_features = self.conditional_encoder(conditions)
        cat_latents = torch.cat([in_latents, cond_latents], dim=1)

        return self.decoder(
            cat_latents, cond_features if self.skip_connect else None
        ), cat_latents, means, log_stds


class GAN(GridGenerator):
    is_variational = True # just means we provide noise to decoder
    has_input_encoder = False
    has_conditional_encoder = False

    def forward(self, inputs=None, conditions=None, batch_size=None):
        var_latents = self.sample_latents(batch_size)
        return self.decoder(var_latents), var_latents, None, None


class CGAN(GAN):
    is_variational = True # just means we provide noise to decoder
    has_input_encoder = False
    has_conditional_encoder = True

    def forward(self, inputs=None, conditions=None, batch_size=None):
        var_latents = self.sample_latents(batch_size)
        cond_latents, cond_features = self.conditional_encoder(conditions)
        cat_latents = torch.cat([var_latents, cond_latents], dim=1)
        return self.decoder(
            cat_latents, cond_features if self.skip_connect else None
        ), cat_latents, None, None


VAEGAN = VAE
CVAEGAN = CVAE
