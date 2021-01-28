import sys, os, pytest
import numpy as numpy
from numpy import isclose
from numpy.linalg import norm
import torch

sys.path.insert(0, '.')
import liGAN.models as models


class TestConvReLU(object):

    @pytest.fixture
    def conv(self):
        return models.ConvReLU(19, 16, 3, 0.1)

    def test_init(self, conv):
        assert len(conv) == 2

    def test_forward_cpu(self, conv):
        x = torch.zeros(10, 19, 8, 8, 8).cpu()
        y = conv.to('cpu')(x)
        assert y.shape == (10, 16, 8, 8, 8)

    def test_forward_cuda(self, conv):
        x = torch.zeros(10, 19, 8, 8, 8).cuda()
        y = conv.to('cuda')(x)
        assert y.shape == (10, 16, 8, 8, 8)


class TestConvBlock(object):

    @pytest.fixture
    def convs(self):
        return models.ConvBlock(4, 19, 16, 3, 0.1)

    def test_init(self, convs):
        assert len(convs) == 4

    def test_forward_cpu(self, convs):
        x = torch.zeros(10, 19, 8, 8, 8).cpu()
        y = convs.to('cpu')(x)
        assert y.shape == (10, 16, 8, 8, 8)

    def test_forward_cuda(self, convs):
        x = torch.zeros(10, 19, 8, 8, 8).cuda()
        y = convs.to('cuda')(x)
        assert y.shape == (10, 16, 8, 8, 8)


class TestEncoder(object):

    def get_enc(self, n_output):
        return models.Encoder(
            n_channels=19,
            grid_dim=8,
            n_filters=5,
            width_factor=2,
            n_levels=3,
            conv_per_level=1,
            kernel_size=3,
            relu_leak=0.1,
            pool_type='a',
            pool_factor=2,
            n_output=n_output,
        ).cuda()

    @pytest.fixture
    def enc0(self):
        return self.get_enc(n_output=[])

    @pytest.fixture
    def enc1(self):
        return self.get_enc(n_output=128)

    @pytest.fixture
    def enc2(self):
        return self.get_enc(n_output=[128, 128])

    def test_enc1_init(self, enc1):
        assert len(enc1.grid_modules) == 5
        assert len(enc1.task_modules) == 1
        assert enc1.n_channels == 20
        assert enc1.grid_dim == 2

    def test_enc1_forward(self, enc1):
        x = torch.zeros(10, 19, 8, 8, 8).cuda()
        y = enc1(x)
        assert y.shape == (10, 128)

    def test_enc1_backward(self, enc1):
        x = torch.zeros(10, 19, 8, 8, 8).cuda()
        y = enc1(x)
        y.backward(torch.zeros(10, 128).cuda())

    def test_enc2_init(self, enc2):
        assert len(enc2.grid_modules) == 5
        assert len(enc2.task_modules) == 2
        assert enc2.n_channels == 20
        assert enc2.grid_dim == 2

    def test_enc2_forward(self, enc2):
        x = torch.zeros(10, 19, 8, 8, 8).cuda()
        y0, y1 = enc2(x)
        assert y0.shape == y1.shape == (10, 128)

    def test_enc2_backward(self, enc2):
        x = torch.zeros(10, 19, 8, 8, 8).cuda()
        y0, y1 = enc2(x)
        (y0 + y1).backward(torch.zeros(10, 128).cuda())


class TestDecoder(object):

    @pytest.fixture
    def dec(self):
        return models.Decoder(
            n_input=128,
            grid_dim=2,
            n_channels=64,
            width_factor=2,
            n_levels=3,
            deconv_per_level=1,
            kernel_size=3,
            relu_leak=0.1,
            unpool_type='n',
            unpool_factor=2,
            n_output=19,
        ).cuda()

    def test_init(self, dec):
        assert len(dec.modules) == 7
        assert dec.n_channels == 19
        assert dec.grid_dim == 8

    def test_forward(self, dec):
        x = torch.zeros(10, 128).cuda()
        y = dec(x)
        assert y.shape == (10, 19, 8, 8, 8)

    def test_backward(self, dec):
        x = torch.zeros(10, 128).cuda()
        y = dec(x)
        y.backward(torch.zeros(10, 19, 8, 8, 8).cuda())


class TestGenerator(object):

    def get_gen(self, n_channels_in, var_input):
        return models.Generator(
            n_channels_in=n_channels_in,
            n_channels_out=19,
            grid_dim=8,
            n_filters=5,
            width_factor=2,
            n_levels=3,
            conv_per_level=1,
            kernel_size=3,
            relu_leak=0.1,
            pool_type='a',
            unpool_type='n',
            n_latent=128,
            var_input=var_input,
        ).cuda()

    @pytest.fixture
    def gen0(self):
        return self.get_gen(
            n_channels_in=[],
            var_input=None,
        )

    @pytest.fixture
    def gen1(self):
        return self.get_gen(
            n_channels_in=19,
            var_input=None,
        )

    @pytest.fixture
    def gen2(self):
        return self.get_gen(
            n_channels_in=[19, 16],
            var_input=None,
        )

    @pytest.fixture
    def vae(self):
        return self.get_gen(
            n_channels_in=[19],
            var_input=0,
        )

    @pytest.fixture
    def cvae(self):
        return self.get_gen(
            n_channels_in=[19, 16],
            var_input=0,
        )

    def test_gen1_init(self, gen1):
        assert gen1.n_inputs == 1
        assert not gen1.variational

    def test_gen2_init(self, gen2):
        assert gen2.n_inputs == 2
        assert not gen2.variational

    def test_vae_init(self, vae):
        assert vae.n_inputs == 1
        assert vae.variational

    def test_cvae_init(self, cvae):
        assert cvae.n_inputs == 2
        assert cvae.variational

    def test_gen1_forward(self, gen1):
        x = torch.zeros(10, 19, 8, 8, 8).cuda()
        y = gen1(x)
        assert y.shape == (10, 19, 8, 8, 8)

    def test_gen2_forward(self, gen2):
        x0 = torch.zeros(10, 19, 8, 8, 8).cuda()
        x1 = torch.zeros(10, 16, 8, 8, 8).cuda()
        y = gen2(x0, x1)
        assert y.shape == (10, 19, 8, 8, 8)

    def test_vae_forward(self, vae):
        x = torch.zeros(10, 19, 8, 8, 8).cuda()
        y = vae(x)
        assert y.shape == (10, 19, 8, 8, 8)

    def test_cvae_forward(self, cvae):
        x0 = torch.zeros(10, 19, 8, 8, 8).cuda()
        x1 = torch.zeros(10, 16, 8, 8, 8).cuda()
        y = cvae(x0, x1)
        assert y.shape == (10, 19, 8, 8, 8)

