import sys, os, pytest
import numpy as numpy
from numpy import isclose
from numpy.linalg import norm
import torch

os.environ['GLOG_minloglevel'] = '1'
import caffe

sys.path.insert(0, '.')
import liGAN.models as models


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
        )

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
        x = torch.zeros(10, 19, 8, 8, 8)
        y = enc1(x)
        assert y.shape == (10, 128)

    def test_enc1_backward(self, enc1):
        x = torch.zeros(10, 19, 8, 8, 8)
        y = enc1(x)
        y.backward(torch.zeros(10, 128))

    def test_enc2_init(self, enc2):
        assert len(enc2.grid_modules) == 5
        assert len(enc2.task_modules) == 2
        assert enc2.n_channels == 20
        assert enc2.grid_dim == 2

    def test_enc2_forward(self, enc2):
        x = torch.zeros(10, 19, 8, 8, 8)
        y0, y1 = enc2(x)
        assert y0.shape == y1.shape == (10, 128)

    def test_enc2_backward(self, enc2):
        x = torch.zeros(10, 19, 8, 8, 8)
        y0, y1 = enc2(x)
        (y0 + y1).backward(torch.zeros(10, 128))


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
        )

    def test_init(self, dec):
        assert len(dec.modules) == 7
        assert dec.n_channels == 19
        assert dec.grid_dim == 8

    def test_forward(self, dec):
        x = torch.zeros(10, 128)
        y = dec(x)
        assert y.shape == (10, 19, 8, 8, 8)

    def test_backward(self, dec):
        x = torch.zeros(10, 128)
        y = dec(x)
        y.backward(torch.zeros(10, 19, 8, 8, 8))


class TestGenerator(object):

    def get_gen(self, n_channels_in):
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
        )

    @pytest.fixture
    def gen0(self):
        return self.get_gen([])

    @pytest.fixture
    def gen1(self):
        return self.get_gen(19)

    @pytest.fixture
    def gen2(self):
        return self.get_gen([19, 16])

    def test_0_input_init(self, gen0):
        assert gen0.n_inputs == 0

    def test_1_input_init(self, gen1):
        assert gen1.n_inputs == 1

    def test_2_input_init(self, gen2):
        assert gen2.n_inputs == 2

    def test_0_input_forward(self, gen0):
        x = torch.zeros(10, 128)
        y = gen0(x)
        assert y.shape == (10, 19, 8, 8, 8)

    def test_1_input_forward(self, gen1):
        x = torch.zeros(10, 19, 8, 8, 8)
        y = gen1(x)
        assert y.shape == (10, 19, 8, 8, 8)

    def test_2_input_forward(self, gen2):
        x0 = torch.zeros(10, 19, 8, 8, 8)
        x1 = torch.zeros(10, 16, 8, 8, 8)
        y = gen2(x0, x1)
        assert y.shape == (10, 19, 8, 8, 8)
