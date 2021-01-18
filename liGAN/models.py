import numpy as np
from torch import nn


# mapping of unpool_types to Upsample modes
unpool_type_map = dict(
    n='nearest',
    t='trilinear',
)


class ConvReLU(nn.Sequential):

    def __init__(self, n_input, n_output, kernel_size, relu_leak):

        conv = nn.Conv3d(
            in_channels=n_input,
            out_channels=n_output,
            kernel_size=kernel_size,
            padding=kernel_size//2,
        )
        relu = nn.LeakyReLU(
            negative_slope=relu_leak,
            inplace=True,
        )
        super().__init__(conv, relu)


class ConvBlock(nn.Sequential):

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

        modules = []
        for i in range(n_convs):
            conv_relu = ConvReLU(n_input, n_output, kernel_size, relu_leak)
            n_input = n_output
            modules.append(conv_relu)

        super().__init__(*modules)


class DeconvReLU(nn.Sequential):

    def __init__(self, n_input, n_output, kernel_size, relu_leak):

        deconv = nn.ConvTranspose3d(
            in_channels=n_input,
            out_channels=n_output,
            kernel_size=kernel_size,
            padding=kernel_size//2,
        )
        relu = nn.LeakyReLU(
            negative_slope=relu_leak,
            inplace=True,
        )
        super().__init__(deconv, relu)


class DeconvBlock(nn.Sequential):

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

        modules = []
        for i in range(n_deconvs):
            deconv_relu = DeconvReLU(
                n_input, n_output, kernel_size, relu_leak
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


class ReshapeFc(nn.Module):

    def __init__(self, in_shape, n_output, relu_leak):
        super().__init__()
        self.n_input = np.prod(in_shape)
        self.fc = nn.Linear(self.n_input, n_output)
        self.relu = nn.LeakyReLU(negative_slope=relu_leak, inplace=True)

    def forward(self, x):
        return self.relu(self.fc(x.reshape(-1, self.n_input)))


class FcReshape(nn.Module):

    def __init__(self, n_input, out_shape, relu_leak):
        super().__init__()
        self.fc = nn.Linear(n_input, np.prod(out_shape))
        self.relu = nn.LeakyReLU(negative_slope=relu_leak, inplace=True)
        self.out_shape = (-1,) + tuple(out_shape)

    def forward(self, x):
        return self.relu(self.fc(x)).reshape(self.out_shape)


class Encoder(nn.Sequential):

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
        self.modules = []

        # track changing dimensions
        self.n_channels = n_channels
        self.grid_dim = grid_dim

        if init_conv_pool:
            self.add_conv(n_filters, kernel_size, relu_leak)
            self.add_pool(pool_type, pool_factor)

        for i in range(n_levels):

            if i > 0: # downsample between conv blocks
                self.add_pool(pool_type, pool_factor)
                n_filters *= width_factor

            self.add_conv_block(
                conv_per_level, n_filters, kernel_size, relu_leak
            )

        self.add_reshape_fc(n_output)

        super().__init__(*self.modules)

    def add_conv(self, n_filters, kernel_size, relu_leak):
        conv = ConvReLU(self.n_channels, n_filters, kernel_size, relu_leak)
        self.modules.append(conv)
        self.n_channels = n_filters

    def add_pool(self, pool_type, pool_factor):
        pool = Pooling(self.n_channels, pool_type, pool_factor)
        self.modules.append(pool)
        self.grid_dim //= pool_factor

    def add_conv_block(self, n_convs, n_filters, kernel_size, relu_leak):
        conv_block = ConvBlock(
            n_convs, self.n_channels, n_filters, kernel_size, relu_leak
        )
        self.modules.append(conv_block)
        self.n_channels = n_filters

    def add_reshape_fc(self, n_output):
        in_shape = (self.n_channels,) + (self.grid_dim,)*3
        fc = ReshapeFc(in_shape, n_output, relu_leak=0)
        self.modules.append(fc)


class Decoder(nn.Sequential):

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
        self.modules = []

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

        super().__init__(*self.modules)

    def add_fc_reshape(self, n_input, n_channels, grid_dim, relu_leak):
        out_shape = (n_channels,) + (grid_dim,)*3
        fc_reshape = FcReshape(n_input, out_shape, relu_leak)
        self.modules.append(fc_reshape)
        self.n_channels = n_channels
        self.grid_dim = grid_dim

    def add_unpool(self, unpool_type, unpool_factor):
        unpool = Unpooling(self.n_channels, unpool_type, unpool_factor)
        self.modules.append(unpool)
        self.grid_dim *= unpool_factor

    def add_deconv(self, n_filters, kernel_size, relu_leak):
        deconv = DeconvReLU(
            self.n_channels, n_filters, kernel_size, relu_leak
        )
        self.modules.append(deconv)
        self.n_channels = n_filters

    def add_deconv_block(self, n_deconvs, n_filters, kernel_size, relu_leak):
        deconv_block = DeconvBlock(
            n_deconvs, self.n_channels, n_filters, kernel_size, relu_leak
        )
        self.modules.append(deconv_block)
        self.n_channels = n_filters


class EncoderDecoder(nn.Module):

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
        pool_type='a',
        unpool_type='n',
        pool_factor=2,
        n_latent=1024,
        init_conv_pool=False,
    ):
        super().__init__()
        
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

    def forward(self, x):
        return self.decoder(self.encoder(x))