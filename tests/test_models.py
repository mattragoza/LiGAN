import sys, os, pytest, time
import numpy as numpy
from numpy import isclose
from numpy.linalg import norm
import torch

sys.path.insert(0, '.')
import liGAN.models as models
from liGAN.models import AE, VAE, CE, CVAE, GAN, CGAN
from liGAN.models import compute_grad_norm as param_grad_norm


class TestConv3DReLU(object):

    @pytest.fixture
    def conv(self):
        return models.Conv3DReLU(
            n_channels_in=19,
            n_channels_out=16,
            kernel_size=3,
            relu_leak=0.1,
            batch_norm=2,
            spectral_norm=1
        )

    def test_init(self, conv):
        assert len(conv) == 3, 'different num modules'

    def test_forward_cpu(self, conv):
        x = torch.zeros(10, 19, 8, 8, 8).cpu()
        y = conv.to('cpu')(x)
        assert y.shape == (10, 16, 8, 8, 8), 'different output shape'

    def test_forward_cuda(self, conv):
        x = torch.zeros(10, 19, 8, 8, 8).cuda()
        y = conv.to('cuda')(x)
        assert y.shape == (10, 16, 8, 8, 8), 'different output shape'


class TestConv3DBlock(object):

    @pytest.fixture
    def convs(self):
        return models.Conv3DBlock(
            n_convs=4,
            n_channels_in=19,
            n_channels_out=16,
            kernel_size=3,
            relu_leak=0.1,
            batch_norm=2,
            spectral_norm=1
        )

    def test_init(self, convs):
        assert len(convs) == 4, 'different num modules'

    def test_forward_cpu(self, convs):
        x = torch.zeros(10, 19, 8, 8, 8).cpu()
        y = convs.to('cpu')(x)
        assert y.shape == (10, 16, 8, 8, 8), 'different output shape'

    def test_forward_cuda(self, convs):
        x = torch.zeros(10, 19, 8, 8, 8).cuda()
        y = convs.to('cuda')(x)
        assert y.shape == (10, 16, 8, 8, 8), 'different output shape'


class TestGridEncoder(object):

    def get_enc(self, n_output):
        return models.GridEncoder(
            n_channels=19,
            grid_size=8,
            n_filters=5,
            width_factor=2,
            n_levels=3,
            conv_per_level=1,
            kernel_size=3,
            relu_leak=0.1,
            batch_norm=2,
            spectral_norm=1,
            pool_type='a',
            pool_factor=2,
            n_output=n_output,
        ).cuda()

    @pytest.fixture
    def enc0(self):
        return self.get_enc([])

    @pytest.fixture
    def enc1(self):
        return self.get_enc(128)

    @pytest.fixture
    def enc2(self):
        return self.get_enc([128, 128])

    def test_enc1_init(self, enc1):
        assert len(enc1.grid_modules) == 5, 'different num grid modules'
        assert len(enc1.task_modules) == 1, 'different num task modules'
        assert enc1.n_channels == 20, 'different num grid channels'
        assert enc1.grid_size == 2, 'different grid size'

    def test_enc1_forward(self, enc1):
        x = torch.zeros(10, 19, 8, 8, 8).cuda()
        y, _ = enc1(x)
        assert y.shape == (10, 128), 'different output shape'
        assert y.norm() > 0, 'output norm is zero'

    def test_enc1_backward0(self, enc1):
        x = torch.zeros(10, 19, 8, 8, 8).cuda()
        x.requires_grad = True
        y, _ = enc1(x)
        y.backward(torch.zeros_like(y))
        assert x.grad is not None, 'input has no gradient'
        assert x.grad.norm() == 0, 'input gradient not zero'

    def test_enc1_backward1(self, enc1):
        x = torch.zeros(10, 19, 8, 8, 8).cuda()
        x.requires_grad = True
        y, _ = enc1(x)
        y.backward(torch.ones_like(y))
        assert x.grad is not None, 'input has no gradient'
        assert x.grad.norm() > 0, 'input gradient is zero'

    def test_enc2_init(self, enc2):
        assert len(enc2.grid_modules) == 5, 'different num grid modules'
        assert len(enc2.task_modules) == 2, 'different num task modules'
        assert enc2.n_channels == 20, 'different num grid channels'
        assert enc2.grid_size == 2, 'different grid size'

    def test_enc2_forward(self, enc2):
        x = torch.zeros(10, 19, 8, 8, 8).cuda()
        (y0, y1), _ = enc2(x)
        assert y0.shape == y1.shape == (10, 128), 'different output shape'
        assert y0.norm() > 0, 'output norm is zero'
        assert y1.norm() > 0, 'output norm is zero'

    def test_enc2_backward0(self, enc2):
        x = torch.zeros(10, 19, 8, 8, 8).cuda()
        x.requires_grad = True
        (y0, y1), _ = enc2(x)
        (y0 + y1).backward(torch.zeros_like(y0 + y1))
        assert x.grad is not None, 'input has no gradient'
        assert x.grad.norm() == 0, 'input gradient not zero'

    def test_enc2_backward1(self, enc2):
        x = torch.zeros(10, 19, 8, 8, 8).cuda()
        x.requires_grad = True
        (y0, y1), _ = enc2(x)
        (y0 + y1).backward(torch.ones_like(y0 + y1))
        assert x.grad is not None, 'input has no gradient'
        assert x.grad.norm() > 0, 'input gradient is zero'


class TestGridDecoder(object):

    @pytest.fixture
    def dec(self):
        return models.GridDecoder(
            n_input=128,
            grid_size=2,
            n_channels=64,
            width_factor=2,
            n_levels=3,
            tconv_per_level=1,
            kernel_size=3,
            relu_leak=0.1,
            batch_norm=2,
            spectral_norm=1,
            unpool_type='n',
            unpool_factor=2,
            n_channels_out=19,
        ).cuda()

    def test_init(self, dec):
        assert len(dec.fc_modules) == 1, 'different num fc modules'
        assert len(dec.grid_modules) == 6, 'different num grid modules'
        assert dec.n_channels == 19, 'different num grid channels'
        assert dec.grid_size == 8, 'different grid size'

    def test_forward(self, dec):
        x = torch.zeros(10, 128).cuda()
        y = dec(x)
        assert y.shape == (10, 19, 8, 8, 8), 'different output shape'
        assert y.norm() > 0, 'output norm is zero'

    def test_backward0(self, dec):
        x = torch.zeros(10, 128).cuda()
        x.requires_grad = True
        y = dec(x)
        y.backward(torch.zeros_like(y))
        assert x.grad is not None, 'input has not gradient'
        assert x.grad.norm() == 0, 'input gradient not zero'

    def test_backward1(self, dec):
        x = torch.zeros(10, 128).cuda()
        x.requires_grad = True
        y = dec(x)
        y.backward(torch.ones_like(y))
        assert x.grad is not None, 'input has not gradient'
        assert x.grad.norm() > 0, 'input gradient is zero'


class TestGridGenerator(object):

    @pytest.fixture(params=[AE, CE, VAE, CVAE, GAN, CGAN])
    def gen(self, request):
        model_type = request.param
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
            batch_norm=2,
            spectral_norm=1,
            pool_type='a',
            unpool_type='n',
            n_latent=128,
            skip_connect=False,
            device='cuda'
        )

    def test_gen_init(self, gen):
        n = type(gen).__name__
        assert gen.is_variational == (n.endswith('VAE') or n.endswith('GAN'))
        assert gen.has_input_encoder == n.endswith('AE')
        assert gen.has_conditional_encoder == n.startswith('C')

    def test_gen_forward_poster(self, gen):

        batch_size = 10
        inputs = torch.zeros(batch_size, 19, 8, 8, 8).cuda()
        conditions = torch.zeros(batch_size, 16, 8, 8, 8).cuda()

        outputs, latents, means, log_stds = gen(inputs, conditions, batch_size)

        assert outputs.shape == (batch_size, 19, 8, 8, 8), 'different output shape'
        assert latents.shape == (batch_size, gen.n_decoder_input), 'different latent shape'
        assert outputs.norm() > 0, 'output norm is zero'

    def test_gen_forward_prior(self, gen):

        batch_size = 10
        inputs = None
        conditions = torch.zeros(batch_size, 16, 8, 8, 8).cuda()

        outputs, latents, means, log_stds = gen(inputs, conditions, batch_size)

        assert outputs.shape == (batch_size, 19, 8, 8, 8), 'different output shape'
        assert latents.shape == (batch_size, gen.n_decoder_input), 'different latent shape'
        assert outputs.norm() > 0, 'output norm is zero'

    def test_gen_backward_poster0(self, gen):

        batch_size = 10
        inputs = torch.zeros(batch_size, 19, 8, 8, 8).cuda()
        conditions = torch.zeros(batch_size, 16, 8, 8, 8).cuda()
        inputs.requires_grad = True
        conditions.requires_grad = True

        outputs, latents, means, log_stds = gen(inputs, conditions, batch_size)
        outputs.backward(torch.zeros_like(outputs))

        assert param_grad_norm(gen) == 0, 'param gradient not zero'
        assert param_grad_norm(gen.decoder) == 0, 'decoder gradient is zero'

        if gen.has_input_encoder:
            assert param_grad_norm(gen.input_encoder) == 0, \
                'input encoder gradient is zero'
            assert inputs.grad is not None, 'input has no gradient'
            assert inputs.grad.norm() == 0, 'input gradient not zero'
        else:
            assert inputs.grad is None, 'input has a gradient'

        if gen.has_conditional_encoder:
            assert param_grad_norm(gen.conditional_encoder) == 0, \
                'conditional encoder gradient is zero'
            assert conditions.grad is not None, 'condition has no gradient'
            assert conditions.grad.norm() == 0, 'condition gradient not zero'
        else:
            assert conditions.grad is None, 'condition has a gradient'

    def test_gen_backward_poster1(self, gen):

        batch_size = 10
        inputs = torch.zeros(batch_size, 19, 8, 8, 8).cuda()
        conditions = torch.zeros(batch_size, 16, 8, 8, 8).cuda()
        inputs.requires_grad = True
        conditions.requires_grad = True

        outputs, latents, means, log_stds = gen(inputs, conditions, batch_size)
        outputs.backward(torch.ones_like(outputs))

        assert param_grad_norm(gen) > 0, 'param gradient is zero'
        assert param_grad_norm(gen.decoder) > 0, 'decoder gradient is zero'

        if gen.has_input_encoder:
            assert param_grad_norm(gen.input_encoder) > 0, \
                'input encoder gradient is zero'
            assert inputs.grad is not None, 'input has no gradient'
            assert inputs.grad.norm() > 0, 'input gradient is zero'
        else:
            assert inputs.grad is None, 'input has a gradient'

        if gen.has_conditional_encoder:
            assert param_grad_norm(gen.conditional_encoder) > 0, \
                'conditional encoder gradient is zero'
            assert conditions.grad is not None, 'condition has no gradient'
            assert conditions.grad.norm() > 0, 'condition gradient is zero'
        else:
            assert conditions.grad is None, 'condition has a gradient'

    def test_gen_backward_prior0(self, gen):

        batch_size = 10
        inputs = None
        conditions = torch.zeros(batch_size, 16, 8, 8, 8).cuda()
        conditions.requires_grad = True

        outputs, latents, means, log_stds = gen(inputs, conditions, batch_size)
        outputs.backward(torch.zeros_like(outputs))

        assert param_grad_norm(gen) == 0, 'param gradient not zero'
        assert param_grad_norm(gen.decoder) == 0, 'decoder gradient not zero'

        if gen.has_input_encoder:
            assert param_grad_norm(gen.input_encoder) == 0, \
                'input encoder gradient not zero'

        if gen.has_conditional_encoder:
            assert param_grad_norm(gen.conditional_encoder) == 0, \
                'conditional encoder gradient not zero'
            assert conditions.grad is not None, 'condition has no gradient'
            assert conditions.grad.norm() == 0, 'condition gradient not zero'
        else:
            assert conditions.grad is None, 'condition has a gradient'

    def test_gen_backward_prior1(self, gen):

        batch_size = 10
        inputs = None
        conditions = torch.zeros(batch_size, 16, 8, 8, 8).cuda()
        conditions.requires_grad = True

        outputs, latents, means, log_stds = gen(inputs, conditions, batch_size)
        outputs.backward(torch.ones_like(outputs))

        assert param_grad_norm(gen) > 0, 'param gradient is zero'
        assert param_grad_norm(gen.decoder) > 0, 'decoder gradient is zero'

        if gen.has_input_encoder:
            assert param_grad_norm(gen.input_encoder) == 0, \
                'input encoder gradient not zero'

        if gen.has_conditional_encoder:
            assert param_grad_norm(gen.conditional_encoder) > 0, \
                'conditional encoder gradient is zero'
            assert conditions.grad is not None, 'condition has no gradient'
            assert conditions.grad.norm() > 0, 'condition gradient is zero'
        else:
            assert conditions.grad is None, 'condition has a gradient'

    def test_gen_benchmark(self, gen):

        t0 = time.time()
        for i in range(10):

            batch_size = 10
            inputs = torch.zeros(batch_size, 19, 8, 8, 8).cuda()
            conditions = torch.zeros(batch_size, 16, 8, 8, 8).cuda()

            generated, latents, means, log_stds = gen(inputs, conditions, batch_size)

        t_delta = time.time() - t0
        assert t_delta < 1, 'too slow'
