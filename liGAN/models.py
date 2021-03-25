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


def initialize_weights(m):
    '''
    Xavier initialization with fan-in variance
    norm mode, as implemented in caffe.
    '''
    if isinstance(m, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
        fan_in = nn.init._calculate_correct_fan(m.weight, 'fan_in')
        scale = np.sqrt(3 / fan_in)
        nn.init.uniform_(m.weight, -scale, scale)
        nn.init.constant_(m.bias, 0)


def compute_grad_norm(model):
    grad_norm2 = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        grad_norm2 += (p.grad.data**2).sum().item()
    return grad_norm2**(1/2)


def normalize_grad(model):
    grad_norm = compute_grad_norm(model)
    if grad_norm is None:
        return
    for p in model.parameters():
        if p.grad is None:
            continue
        p.grad /= grad_norm


class ConvReLU(nn.Sequential):

    def __init__(
        self, n_input, n_output, kernel_size, relu_leak, batch_norm
    ):
        modules = [
            nn.Conv3d(
                in_channels=n_input,
                out_channels=n_output,
                kernel_size=kernel_size,
                padding=kernel_size//2,
            ),
            nn.LeakyReLU(
                negative_slope=relu_leak,
                inplace=True,
            )
        ]
        if batch_norm > 0:
            modules.insert(batch_norm, nn.BatchNorm3d(n_output))

        super().__init__(*modules)


class ConvBlock(nn.Sequential):

    def __init__(
        self,
        n_convs,
        n_input,
        n_output,
        kernel_size,
        relu_leak,
        batch_norm,
        dense_net=False,
    ):
        if dense_net:
            raise NotImplementedError('TODO densely-connected')

        modules = []
        for i in range(n_convs):
            conv_relu = ConvReLU(
                n_input, n_output, kernel_size, relu_leak, batch_norm
            )
            n_input = n_output
            modules.append(conv_relu)

        super().__init__(*modules)


class DeconvReLU(nn.Sequential):

    def __init__(
        self, n_input, n_output, kernel_size, relu_leak, batch_norm,
    ):
        modules = [
            nn.ConvTranspose3d(
                in_channels=n_input,
                out_channels=n_output,
                kernel_size=kernel_size,
                padding=kernel_size//2,
            ),
            nn.LeakyReLU(
                negative_slope=relu_leak,
                inplace=True,
            )
        ]
        if batch_norm > 0:
            modules.insert(batch_norm, nn.BatchNorm3d(n_output))

        super().__init__(*modules)


class DeconvBlock(nn.Sequential):

    def __init__(
        self,
        n_deconvs,
        n_input,
        n_output,
        kernel_size,
        relu_leak,
        batch_norm,
        dense_net=False,
    ):
        if dense_net:
            raise NotImplementedError('TODO densely-connected')

        modules = []
        for i in range(n_deconvs):
            deconv_relu = DeconvReLU(
                n_input, n_output, kernel_size, relu_leak, batch_norm
            )
            n_input = n_output
            modules.append(deconv_relu)

        super().__init__(*modules)


class Pooling(nn.Sequential):

    def __init__(self, n_input, pool_type, pool_factor):

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
                in_channels=n_input,
                out_channels=n_input,
                groups=n_input,
                kernel_size=pool_factor,
                stride=pool_factor,
            )

        else:
            raise ValueError('unknown pool_type ' + repr(pool_type))

        super().__init__(pool)


class Unpooling(nn.Sequential):

    def __init__(self, n_input, unpool_type, unpool_factor):

        if unpool_type in unpool_type_map:
            
            unpool = nn.Upsample(
                scale_factor=unpool_factor,
                mode=unpool_type_map[unpool_type],
            )

        elif unpool_type == 'c':
            
            unpool = nn.Deconv3d(
                in_channels=n_input,
                out_channels=n_input,
                groups=n_input,
                kernel_size=unpool_factor,
                stride=unpool_factor,
            )

        else:
            raise ValueError('unknown unpool_type ' + repr(unpool_type))

        super().__init__(unpool)


class Reshape(nn.Module):

    def __init__(self, *args):
        super().__init__()
        self.shape = tuple(args)

    def forward(self, x):
        return x.reshape(self.shape)


class ReshapeFc(nn.Sequential):

    def __init__(self, in_shape, n_output, activ_fn=None):
        n_input = np.prod(in_shape)
        modules = [
            Reshape(-1, n_input),
            nn.Linear(n_input, n_output)
        ]
        if activ_fn:
            modules.append(activ_fn)
        super().__init__(*modules)


class FcReshape(nn.Sequential):

    def __init__(self, n_input, out_shape, relu_leak, batch_norm):
        n_output = np.prod(out_shape)
        modules = [
            nn.Linear(n_input, n_output),
            nn.LeakyReLU(negative_slope=relu_leak, inplace=True),
            Reshape(-1, *out_shape)
        ]
        if batch_norm > 0:
            modules.append(nn.BatchNorm3d(out_shape[0]))
        super().__init__(*modules)


class Encoder(nn.Module):

    # TODO reimplement the following:
    # - self-attention
    # - densely-connected
    # - batch discrimination
    # - fully-convolutional
    # - skip connections
    
    def __init__(
        self,
        n_channels,
        grid_size,
        n_filters,
        width_factor,
        n_levels,
        conv_per_level,
        kernel_size,
        relu_leak,
        batch_norm,
        pool_type,
        pool_factor,
        n_output,
        output_activ_fn=None,
        init_conv_pool=False,
    ):
        super().__init__()

        # sequence of convs and/or pools
        self.grid_modules = []

        # track changing grid dimensions
        self.n_channels = n_channels
        self.grid_size = grid_size

        if init_conv_pool:
            self.add_conv(
                'init_conv', n_filters, kernel_size, relu_leak, batch_norm
            )
            self.add_pool('init_pool', pool_type, pool_factor)

        for i in range(n_levels):

            if i > 0: # downsample between conv blocks
                pool_name = 'level' + str(i) + '_pool'
                self.add_pool(
                    pool_name, pool_type, pool_factor
                )
                n_filters *= width_factor

            conv_block_name = 'level' + str(i)
            self.add_conv_block(
                conv_block_name,
                conv_per_level,
                n_filters,
                kernel_size,
                relu_leak,
                batch_norm
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

        for i, (n_o, activ_fn) in enumerate(zip(n_output, output_activ_fn)):
            fc_name = 'fc' + str(i)
            self.add_reshape_fc(fc_name, n_o, activ_fn)

    def add_conv(self, name, n_filters, kernel_size, relu_leak, batch_norm):
        conv = ConvReLU(
            self.n_channels, n_filters, kernel_size, relu_leak, batch_norm
        )
        self.add_module(name, conv)
        self.grid_modules.append(conv)
        self.n_channels = n_filters

    def add_pool(self, name, pool_type, pool_factor):
        pool = Pooling(self.n_channels, pool_type, pool_factor)
        self.add_module(name, pool)
        self.grid_modules.append(pool)
        self.grid_size //= pool_factor

    def add_conv_block(
        self, name, n_convs, n_filters, kernel_size, relu_leak, batch_norm
    ):
        conv_block = ConvBlock(
            n_convs,
            self.n_channels,
            n_filters,
            kernel_size,
            relu_leak,
            batch_norm
        )
        self.add_module(name, conv_block)
        self.grid_modules.append(conv_block)
        self.n_channels = n_filters

    def add_reshape_fc(self, name, n_output, activ_fn):
        in_shape = (self.n_channels,) + (self.grid_size,)*3
        fc = ReshapeFc(in_shape, n_output, activ_fn)
        self.add_module(name, fc)
        self.task_modules.append(fc)

    def forward(self, input):

        # conv pool sequence
        conv_features = []
        for f in self.grid_modules:
            output = f(input)
            input = output

            if not isinstance(f, Pooling):
                conv_features.append(output)

        # fully-connected outputs
        outputs = [f(input) for f in self.task_modules]

        return reduce_list(outputs), conv_features


class Decoder(nn.Module):

    # TODO re-implement the following:
    # - self-attention
    # - densely-connected
    # - fully-convolutional
    # - gaussian output
    # - skip connections

    def __init__(
        self,
        n_input,
        grid_size,
        n_channels,
        width_factor,
        n_levels,
        deconv_per_level,
        kernel_size,
        relu_leak,
        batch_norm,
        unpool_type,
        unpool_factor,
        n_output,
        final_unpool=False,
        skip_connect=False,
    ):
        super().__init__()
        self.skip_connect = skip_connect

        # first fc layer maps to initial grid shape
        self.fc_modules = []
        self.n_input = n_input
        self.add_fc_reshape(
            'fc', n_input, n_channels, grid_size, relu_leak, batch_norm
        )
        n_filters = n_channels

        self.grid_modules = []
        for i in reversed(range(n_levels)):

            if i + 1 < n_levels: # unpool between deconv blocks
                unpool_name = 'level' + str(i) + '_unpool'
                self.add_unpool(
                    unpool_name, unpool_type, unpool_factor
                )
                n_filters //= width_factor

            deconv_block_name = 'level' + str(i)
            self.add_deconv_block(
                deconv_block_name,
                deconv_per_level,
                n_filters,
                kernel_size,
                relu_leak,
                batch_norm
            )

        if final_unpool:
            self.add_unpool('final_unpool', unpool_type, unpool_factor)

        # final deconv maps to correct n_output channels
        self.add_deconv(
            'final_conv', n_output, kernel_size, relu_leak, batch_norm
        )

    def add_fc_reshape(
        self, name, n_input, n_channels, grid_size, relu_leak, batch_norm
    ):
        out_shape = (n_channels,) + (grid_size,)*3
        fc_reshape = FcReshape(n_input, out_shape, relu_leak, batch_norm)
        self.add_module(name, fc_reshape)
        self.fc_modules.append(fc_reshape)
        self.n_channels = n_channels
        self.grid_size = grid_size

    def add_unpool(self, name, unpool_type, unpool_factor):
        unpool = Unpooling(self.n_channels, unpool_type, unpool_factor)
        self.add_module(name, unpool)
        self.grid_modules.append(unpool)
        self.grid_size *= unpool_factor

    def add_deconv(
        self, name, n_filters, kernel_size, relu_leak, batch_norm
    ):
        n_channels = self.n_channels
        if self.skip_connect:
            n_channels *= 2
        deconv = DeconvReLU(
            n_channels, n_filters, kernel_size, relu_leak, batch_norm
        )
        self.add_module(name, deconv)
        self.grid_modules.append(deconv)
        self.n_channels = n_filters

    def add_deconv_block(
        self, name, n_deconvs, n_filters, kernel_size, relu_leak, batch_norm
    ):
        n_channels = self.n_channels
        if self.skip_connect:
            n_channels *= 2
        deconv_block = DeconvBlock(
            n_deconvs,
            n_channels,
            n_filters,
            kernel_size,
            relu_leak,
            batch_norm
        )
        self.add_module(name, deconv_block)
        self.grid_modules.append(deconv_block)
        self.n_channels = n_filters

    def forward(self, input, conv_featuresS=None):

        for f in self.fc_modules:
            output = f(input)
            input = output

        for i, f in enumerate(self.grid_modules):
            if self.skip_connect and not isinstance(f, Unpooling):
                print(i, input.shape, conv_features[i].shape)
                input = torch.cat([input, conv_features[-i-1]], dim=1)
            output = f(input)
            input = output

        return output


class Generator(nn.Sequential):
    has_input_encoder = False
    has_conditional_encoder = False
    variational = False

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
        pool_type='a',
        unpool_type='n',
        pool_factor=2,
        n_latent=1024,
        init_conv_pool=False,
        skip_connect=False,
        device='cuda',
    ):
        assert type(self) != Generator, 'Generator is abstract'

        super().__init__()
        self.check_encoder_channels(n_channels_in, n_channels_cond)
        assert is_positive_int(n_channels_out)
        assert is_positive_int(n_latent)

        self.n_channels_in = n_channels_in
        self.n_channels_cond = n_channels_cond
        self.n_channels_out = n_channels_out
        self.n_latent = n_latent

        if self.has_input_encoder:

            if self.variational:
                n_output = [n_latent, n_latent] # means and log_stds
            else:
                n_output = n_latent

            self.input_encoder = Encoder(
                n_channels=n_channels_in,
                grid_size=grid_size,
                n_filters=n_filters,
                width_factor=width_factor,
                n_levels=n_levels,
                conv_per_level=conv_per_level,
                kernel_size=kernel_size,
                relu_leak=relu_leak,
                batch_norm=batch_norm,
                pool_type=pool_type,
                pool_factor=pool_factor,
                n_output=n_output,
                init_conv_pool=init_conv_pool,
            )

        if self.has_conditional_encoder:

            self.conditional_encoder = Encoder(
                n_channels=n_channels_cond,
                grid_size=grid_size,
                n_filters=n_filters,
                width_factor=width_factor,
                n_levels=n_levels,
                conv_per_level=conv_per_level,
                kernel_size=kernel_size,
                relu_leak=relu_leak,
                batch_norm=batch_norm,
                pool_type=pool_type,
                pool_factor=pool_factor,
                n_output=n_latent,
                init_conv_pool=init_conv_pool,
            )

        self.skip_connect = skip_connect

        self.decoder = Decoder(
            n_input=self.n_decoder_input,
            grid_size=grid_size // pool_factor**(n_levels-1),
            n_channels=n_filters * width_factor**(n_levels-1),
            width_factor=width_factor,
            n_levels=n_levels,
            deconv_per_level=conv_per_level,
            kernel_size=kernel_size,
            relu_leak=relu_leak,
            batch_norm=batch_norm,
            unpool_type=unpool_type,
            unpool_factor=pool_factor,
            n_output=n_channels_out,
            final_unpool=init_conv_pool,
        )

        super().to(device)
        self.device = device

    def check_encoder_channels(self, n_channels_in, n_channels_cond):
        if self.has_input_encoder:
            assert is_positive_int(n_channels_in)
        if self.has_conditional_encoder:
            assert is_positive_int(n_channels_cond)

    @property
    def n_decoder_input(self):
        n = 0
        if self.has_input_encoder or self.variational:
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


class AE(Generator):
    has_input_encoder = True

    def forward(self, inputs):
        latents, _ = self.input_encoder(inputs)
        return self.decoder(latents), latents


class VAE(AE):
    variational = True

    def forward(self, inputs, batch_size):

        if inputs is not None: # posterior
            (means, log_stds), _ = self.input_encoder(inputs)
        else: # prior
            means, log_stds = None, None

        latents = self.sample_latents(batch_size, means, log_stds)

        return self.decoder(latents), latents, means, log_stds


class CE(Generator):
    has_conditional_encoder = True

    def forward(self, conditions):
        latents, cond_features = self.conditional_encoder(conditions)
        return self.decoder(
            latents, cond_features if self.skip_connect else None
        ), latents


class CVAE(VAE):
    has_conditional_encoder = True

    def forward(self, inputs, conditions, batch_size):

        if inputs is not None: # posterior
            (means, log_stds), _ = self.input_encoder(inputs)
        else: # prior
            means, log_stds = None, None

        input_latents = self.sample_latents(batch_size, means, log_stds)
        cond_latents, cond_features = self.conditional_encoder(conditions)
        latents = torch.cat([input_latents, cond_latents], dim=1)

        return self.decoder(
            latents, cond_features if self.skip_connect else None
        ), latents, means, log_stds


class GAN(Generator):
    variational = True # just means we provide noise to decoder here

    def forward(self, batch_size):
        latents = self.sample_latents(batch_size)
        return self.decoder(latents), latents


class CGAN(GAN):
    has_conditional_encoder = True

    def forward(self, conditions, batch_size):
        sampled_latents = self.sample_latents(batch_size)
        cond_latents, cond_features = self.conditional_encoder(conditions)
        latents = torch.cat([sampled_latents, cond_latents], dim=1)
        return self.decoder(
            latents, cond_features if self.skip_connect else None
        ), latents
