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
            grid_size=8,
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
        assert enc1.grid_size == 2

    def test_enc1_forward(self, enc1):
        x = torch.zeros(10, 19, 8, 8, 8).cuda()
        y, _ = enc1(x)
        assert y.shape == (10, 128)

    def test_enc1_backward(self, enc1):
        x = torch.zeros(10, 19, 8, 8, 8).cuda()
        y, _ = enc1(x)
        y.backward(torch.zeros(10, 128).cuda())

    def test_enc2_init(self, enc2):
        assert len(enc2.grid_modules) == 5
        assert len(enc2.task_modules) == 2
        assert enc2.n_channels == 20
        assert enc2.grid_size == 2

    def test_enc2_forward(self, enc2):
        x = torch.zeros(10, 19, 8, 8, 8).cuda()
        (y0, y1), _ = enc2(x)
        assert y0.shape == y1.shape == (10, 128)

    def test_enc2_backward(self, enc2):
        x = torch.zeros(10, 19, 8, 8, 8).cuda()
        (y0, y1), _ = enc2(x)
        (y0 + y1).backward(torch.zeros(10, 128).cuda())


class TestDecoder(object):

    @pytest.fixture
    def dec(self):
        return models.Decoder(
            n_input=128,
            grid_size=2,
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
        assert len(dec.fc_modules) == 1
        assert len(dec.grid_modules) == 6
        assert dec.n_channels == 19
        assert dec.grid_size == 8

    def test_forward(self, dec):
        x = torch.zeros(10, 128).cuda()
        y = dec(x)
        assert y.shape == (10, 19, 8, 8, 8)

    def test_backward(self, dec):
        x = torch.zeros(10, 128).cuda()
        y = dec(x)
        y.backward(torch.zeros(10, 19, 8, 8, 8).cuda())


class TestGenerator(object):

    @pytest.fixture(params=['AE','CE','VAE','CVAE','GAN','CGAN'])
    def gen(self, request):
        model_type = getattr(models, request.param)
        return model_type(
            n_channels_in=19 if model_type.has_input_encoder else None,
            n_channels_cond=16 if model_type.has_conditional_encoder else None,
            n_channels_out=19,
            grid_size=8,
            n_filters=5,
            width_factor=2,
            n_levels=3,
            conv_per_level=1,
            kernel_size=3,
            relu_leak=0.1,
            pool_type='a',
            unpool_type='n',
            n_latent=128,
            variational=model_type.variational,
            device='cuda'
        )

    def test_gen_init(self, gen):
        assert True

    def test_gen_forward(self, gen):

        batch_size = 10
        inputs = torch.zeros(batch_size, 19, 8, 8, 8).cuda()
        conditions = torch.zeros(batch_size, 16, 8, 8, 8).cuda()

        if type(gen) == models.AE:
            generated, latents = gen(inputs)

        elif type(gen) == models.CE:
            generated, latents = gen(conditions)

        elif type(gen) == models.VAE:
            generated, latents, means, log_stds = gen(inputs, batch_size)

        elif type(gen) == models.CVAE:
            generated, latents, means, log_stds = gen(inputs, conditions, batch_size)

        elif type(gen) == models.GAN:
            generated, latents = gen(batch_size)

        elif type(gen) == models.CGAN:
            generated, latents = gen(conditions, batch_size)

        assert generated.shape == (batch_size, 19, 8, 8, 8)
        assert latents.shape == (batch_size, gen.n_decoder_input)

    def test_gen_forward_prior(self, gen):

        batch_size = 10
        inputs = None
        conditions = torch.zeros(batch_size, 16, 8, 8, 8).cuda()

        if type(gen) == models.AE:
            with pytest.raises(TypeError):
                generated, latents = gen(inputs)
            return

        elif type(gen) == models.CE:
            generated, latents = gen(conditions)

        elif type(gen) == models.VAE:
            generated, latents, means, log_stds = gen(inputs, batch_size)

        elif type(gen) == models.CVAE:
            generated, latents, means, log_stds = gen(inputs, conditions, batch_size)

        elif type(gen) == models.GAN:
            generated, latents = gen(batch_size)

        elif type(gen) == models.CGAN:
            generated, latents = gen(conditions, batch_size)

        assert generated.shape == (batch_size, 19, 8, 8, 8)
        assert latents.shape == (batch_size, gen.n_decoder_input)

    def test_gen_grad_norm0(self, gen):

        batch_size = 10
        inputs = torch.zeros(batch_size, 19, 8, 8, 8).cuda()
        conditions = torch.zeros(batch_size, 16, 8, 8, 8).cuda()

        if type(gen) == models.AE:
            generated, latents = gen(inputs)

        elif type(gen) == models.CE:
            generated, latents = gen(conditions)

        elif type(gen) == models.VAE:
            generated, latents, means, log_stds = gen(inputs, batch_size)

        elif type(gen) == models.CVAE:
            generated, latents, means, log_stds = gen(inputs, conditions, batch_size)

        elif type(gen) == models.GAN:
            generated, latents = gen(batch_size)

        elif type(gen) == models.CGAN:
            generated, latents = gen(conditions, batch_size)

        generated.backward(torch.zeros_like(generated))

        assert isclose(0, models.compute_grad_norm(gen))

    def test_gen_grad_norm1(self, gen):

        batch_size = 10
        inputs = torch.zeros(batch_size, 19, 8, 8, 8).cuda()
        conditions = torch.zeros(batch_size, 16, 8, 8, 8).cuda()

        if type(gen) == models.AE:
            generated, latents = gen(inputs)

        elif type(gen) == models.CE:
            generated, latents = gen(conditions)

        elif type(gen) == models.VAE:
            generated, latents, means, log_stds = gen(inputs, batch_size)

        elif type(gen) == models.CVAE:
            generated, latents, means, log_stds = gen(inputs, conditions, batch_size)

        elif type(gen) == models.GAN:
            generated, latents = gen(batch_size)

        elif type(gen) == models.CGAN:
            generated, latents = gen(conditions, batch_size)

        generated.backward(torch.ones_like(generated))

        assert not isclose(0, models.compute_grad_norm(gen))

    def test_gen_grad_normalize(self, gen):

        batch_size = 10
        inputs = torch.zeros(batch_size, 19, 8, 8, 8).cuda()
        conditions = torch.zeros(batch_size, 16, 8, 8, 8).cuda()

        if type(gen) == models.AE:
            generated, latents = gen(inputs)

        elif type(gen) == models.CE:
            generated, latents = gen(conditions)

        elif type(gen) == models.VAE:
            generated, latents, means, log_stds = gen(inputs, batch_size)

        elif type(gen) == models.CVAE:
            generated, latents, means, log_stds = gen(inputs, conditions, batch_size)

        elif type(gen) == models.GAN:
            generated, latents = gen(batch_size)

        elif type(gen) == models.CGAN:
            generated, latents = gen(conditions, batch_size)

        generated.backward(torch.ones_like(generated))
        models.normalize_grad(gen)

        assert isclose(1, models.compute_grad_norm(gen))