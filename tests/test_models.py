import sys, os, pytest, time
from contextlib import redirect_stderr
import numpy as numpy
from numpy import isclose
from numpy.linalg import norm
import torch

sys.path.insert(0, '.')
from liGAN import models, interpolation
from liGAN.models import AE, VAE, CE, CVAE, GAN, CGAN, VAE2, CVAE2
from liGAN.models import compute_grad_norm as param_grad_norm
from liGAN.models import get_n_params


batch_size = 10
n_lig_channels = 16
n_rec_channels = 16
grid_size = 48



def test_interpolate():

    n_examples = 4
    n_samples = 10
    batch_size = 5
    n_latent = 2

    # total # interpolation steps so far
    interp_step = 0
    end_pts = torch.zeros((1, n_latent)) + 1e-6

    for example_idx in range(n_examples):
        for sample_idx in range(n_samples):
            full_idx = example_idx*n_samples + sample_idx
            batch_idx = full_idx % batch_size

            if batch_idx == 0: # forward

                latents = torch.randn((batch_size, 1))
                batch_idxs = torch.arange(batch_size)
                latents = (
                    (full_idx + batch_idxs) % (n_samples*2) == 0
                ).float().unsqueeze(1)
                latents = torch.cat([latents, 1-latents], dim=1)

                is_endpt = (interp_step + batch_idxs) % n_samples == 0
                end_pts = torch.cat([end_pts, latents[is_endpt]])

                start_idx = (interp_step + batch_idxs) // n_samples
                stop_idx = start_idx + 1
                start_pts = end_pts[start_idx]
                stop_pts = end_pts[stop_idx]
                k_interp = (
                    (interp_step + batch_idxs) % n_samples + 1
                ).unsqueeze(1) / n_samples

                new_latents = interpolation.slerp(start_pts, stop_pts, k_interp)
                print(new_latents)

                interp_step += batch_size


class TestConv3DReLU(object):

    @pytest.fixture(params=[models.Conv3DReLU, models.TConv3DReLU])
    def conv(self, request):
        return request.param(
            n_channels_in=n_rec_channels,
            n_channels_out=n_lig_channels,
            kernel_size=3,
            relu_leak=0.1,
            batch_norm=2,
            spectral_norm=1
        )

    def test_init(self, conv):
        assert len(conv) == 3, 'different num modules'

    def test_forward_cpu(self, conv):
        x = torch.zeros(batch_size, n_rec_channels, grid_size, grid_size, grid_size).cpu()
        y = conv.to('cpu')(x)
        assert y.shape == (batch_size, n_lig_channels, grid_size, grid_size, grid_size), 'different output shape'

    def test_forward_cuda(self, conv):
        x = torch.zeros(batch_size, n_rec_channels, grid_size, grid_size, grid_size).cuda()
        y = conv.to('cuda')(x)
        assert y.shape == (batch_size, n_lig_channels, grid_size, grid_size, grid_size), 'different output shape'


class TestConv3DBlock(object):

    @pytest.fixture(params=[models.Conv3DBlock, models.TConv3DBlock])
    def conv_type(self, request):
        return request.param

    @pytest.fixture(params=['c', 'r', 'd'])
    def block_type(self, request):
        return request.param

    @pytest.fixture(params=[1, 2, 4])
    def bn_factor(self, request):
        return request.param

    @pytest.fixture
    def conv_block(self, conv_type, block_type, bn_factor):
        return conv_type(
            n_convs=3,
            n_channels_in=n_rec_channels,
            n_channels_out=n_lig_channels,
            kernel_size=3,
            relu_leak=0.1,
            batch_norm=0,
            spectral_norm=1,
            block_type=block_type,
            bottleneck_factor=bn_factor,
        )

    def test_init(self, conv_block):
        assert len(conv_block) == 3, 'different num modules'

    def test_forward_cuda(self, conv_block):
        x = torch.zeros(batch_size, n_rec_channels, grid_size, grid_size, grid_size).cuda()
        y = conv_block.to('cuda')(x)
        print(conv_block)
        print(get_n_params(conv_block))
        assert y.shape == (batch_size, n_lig_channels, grid_size, grid_size, grid_size), 'different output shape'


class TestGridEncoder(object):

    def get_enc(self, n_output):
        return models.GridEncoder(
            n_channels=n_lig_channels,
            grid_size=grid_size,
            n_filters=32,
            width_factor=2,
            n_levels=3,
            conv_per_level=4,
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

    @pytest.fixture
    def x(self):
        return torch.zeros(
            batch_size, n_lig_channels, grid_size, grid_size, grid_size
        ).cuda()

    def test_enc1_init(self, enc1):
        assert len(enc1.grid_modules) == 5, 'different num grid modules'
        assert len(enc1.task_modules) == 1, 'different num task modules'
        assert enc1.n_channels == 128, 'different num grid channels'
        assert enc1.grid_size == grid_size//4, 'different grid size'

    def test_enc1_forward(self, enc1, x):
        y, _ = enc1(x)
        assert y.shape == (10, 128), 'different output shape'
        assert y.norm() > 0, 'output norm is zero'

    def test_enc1_backward0(self, enc1, x):
        x.requires_grad = True
        y, _ = enc1(x)
        y.backward(torch.zeros_like(y))
        assert x.grad is not None, 'input has no gradient'
        assert x.grad.norm() == 0, 'input gradient not zero'

    def test_enc1_backward1(self, enc1, x):
        x.requires_grad = True
        y, _ = enc1(x)
        y.backward(torch.ones_like(y))
        assert x.grad is not None, 'input has no gradient'
        assert x.grad.norm() > 0, 'input gradient is zero'

    def test_enc2_init(self, enc2):
        assert len(enc2.grid_modules) == 5, 'different num grid modules'
        assert len(enc2.task_modules) == 2, 'different num task modules'
        assert enc2.n_channels == 128, 'different num grid channels'
        assert enc2.grid_size == grid_size//4, 'different grid size'

    def test_enc2_forward(self, enc2, x):
        (y0, y1), _ = enc2(x)
        assert y0.shape == y1.shape == (10, 128), 'different output shape'
        assert y0.norm() > 0, 'output norm is zero'
        assert y1.norm() > 0, 'output norm is zero'

    def test_enc2_backward0(self, enc2, x):
        x.requires_grad = True
        (y0, y1), _ = enc2(x)
        (y0 + y1).backward(torch.zeros_like(y0 + y1))
        assert x.grad is not None, 'input has no gradient'
        assert x.grad.norm() == 0, 'input gradient not zero'

    def test_enc2_backward1(self, enc2, x):
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
            grid_size=grid_size // 2**2,
            n_channels=32 * 2**2,
            width_factor=2,
            n_levels=3,
            tconv_per_level=4,
            kernel_size=3,
            relu_leak=0.1,
            batch_norm=2,
            spectral_norm=1,
            unpool_type='n',
            unpool_factor=2,
            n_channels_out=n_rec_channels,
        ).cuda()

    @pytest.fixture
    def x(self):
        return torch.zeros(batch_size, 128).cuda()

    def test_init(self, dec):
        assert len(dec.fc_modules) == 1, 'different num fc modules'
        assert len(dec.grid_modules) == 6, 'different num grid modules'
        assert dec.n_channels == n_rec_channels, 'different num grid channels'
        assert dec.grid_size == grid_size, 'different grid size'

    def test_forward(self, dec, x):
        y = dec(x)
        assert y.shape == (
            batch_size, n_rec_channels, grid_size, grid_size, grid_size
        ), 'different output shape'
        assert y.norm() > 0, 'output norm is zero'

    def test_backward0(self, dec, x):
        x.requires_grad = True
        y = dec(x)
        y.backward(torch.zeros_like(y))
        assert x.grad is not None, 'input has not gradient'
        assert x.grad.norm() == 0, 'input gradient not zero'

    def test_backward1(self, dec, x):
        x.requires_grad = True
        y = dec(x)
        y.backward(torch.ones_like(y))
        assert x.grad is not None, 'input has not gradient'
        assert x.grad.norm() > 0, 'input gradient is zero'


class TestGridGenerator(object):

    @pytest.fixture(params=[AE, CE, VAE, CVAE, GAN, CGAN, VAE2, CVAE2])
    def model_type(self, request):
        return request.param

    @pytest.fixture(params=['c'])#, 'r', 'd'])
    def block_type(self, request):
        return request.param

    @pytest.fixture(params=[0])#, 2, 4])
    def bn_factor(self, request):
        return request.param

    @pytest.fixture(params=[0, 1])
    def init_conv_pool(self, request):
        return request.param

    @pytest.fixture
    def gen(self, model_type, init_conv_pool, block_type, bn_factor):
        model = model_type(
            n_channels_in=n_lig_channels if model_type.has_input_encoder else None,
            n_channels_cond=n_rec_channels if model_type.has_conditional_encoder else None,
            n_channels_out=n_lig_channels,
            grid_size=grid_size,
            n_filters=32,
            width_factor=2,
            n_levels=4 - bool(init_conv_pool),
            conv_per_level=3,
            kernel_size=3,
            relu_leak=0.1,
            batch_norm=0,
            spectral_norm=1,
            pool_type='a',
            unpool_type='n',
            n_latent=128,
            skip_connect=model_type.has_conditional_encoder,
            init_conv_pool=init_conv_pool,
            block_type=block_type,
            bottleneck_factor=bn_factor,
            device='cuda',
            debug=True,
        )
        model.name = '{}_{}_{}_{}'.format(
            model_type.__name__, init_conv_pool, block_type, bn_factor
        )
        return model

    @pytest.fixture
    def inputs(self):
        return torch.zeros(
            batch_size, n_lig_channels, grid_size, grid_size, grid_size
        ).cuda()

    @pytest.fixture
    def conditions(self):
        return torch.zeros(
            batch_size, n_rec_channels, grid_size, grid_size, grid_size
        ).cuda()

    def test_gen_init(self, gen):
        n = type(gen).__name__
        assert gen.is_variational == ('VAE' in n) or n.endswith('GAN')
        assert gen.has_input_encoder == ('AE' in n)
        assert gen.has_conditional_encoder == n.startswith('C')

    def test_gen_forward_poster(self, gen, inputs, conditions):
        outputs, latents, means, log_stds = gen(inputs, conditions, batch_size)
        assert outputs.shape == (batch_size, n_lig_channels, grid_size, grid_size, grid_size), 'different output shape'
        assert latents.shape == (batch_size, gen.n_latent), 'different latent shape'
        assert outputs.norm() > 0, 'output norm is zero'

    def test_gen_forward_prior(self, gen, conditions):
        outputs, latents, means, log_stds = gen(None, conditions, batch_size)
        assert outputs.shape == (batch_size, n_lig_channels, grid_size, grid_size, grid_size), 'different output shape'
        assert latents.shape == (batch_size, gen.n_latent), 'different latent shape'
        assert outputs.norm() > 0, 'output norm is zero'

    def test_gen_backward_poster0(self, gen, inputs, conditions):
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

    def test_gen_backward_poster1(self, gen, inputs, conditions):
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

    def test_gen_backward_prior0(self, gen, conditions):
        conditions.requires_grad = True

        outputs, latents, means, log_stds = gen(None, conditions, batch_size)
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

    def test_gen_backward_prior1(self, gen, conditions):
        conditions.requires_grad = True

        outputs, latents, means, log_stds = gen(None, conditions, batch_size)
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

    def test_gen_benchmark(self, gen, inputs, conditions):
        n_trials = 10

        t0 = time.time()
        for i in range(n_trials):
            if i == 0:
                debug_file = 'tests/output/TEST_{}.model_debug'.format(gen.name)
                with open(debug_file, 'w') as f:
                    with redirect_stderr(f):
                        generated, latents, means, log_stds = gen(
                            inputs, conditions, batch_size
                        )
            else:
                generated, latents, means, log_stds = gen(
                    inputs, conditions, batch_size
                )

        t_delta = time.time() - t0
        t_delta /= n_trials
        n_params = models.get_n_params(gen)
        assert t_delta < 1, \
            '{:.1f}M params\t{:.2f}s / batch'.format(
                n_params / 1e6, t_delta
            )


class TestStage2VAE(object):

    @pytest.fixture(params=[
        (0, 0), (1, 96), (2, 96)
    ])
    def model(self, request):
        n_h_layers, n_h_units = request.param
        return models.Stage2VAE(
            n_input=128,
            n_h_layers=n_h_layers,
            n_h_units=n_h_units,
            n_latent=64,
        ).to('cuda')

    @pytest.fixture
    def inputs(self):
        return torch.zeros(10, 128, device='cuda')

    def test_init(self, model):
        assert model.n_latent == 64

    def test_forward_poster(self, model, inputs):
        outputs, _, _, _ = model(inputs=inputs, batch_size=10)
        assert outputs.shape == inputs.shape

    def test_forward_prior(self, model, inputs):
        outputs, _, _, _ = model(batch_size=10)
        assert outputs.shape == inputs.shape
