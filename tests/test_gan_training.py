import sys, os, pytest
from numpy import isclose, isnan
from torch import optim
import pandas as pd
pd.set_option('display.max_rows', 100)

sys.path.insert(0, '.')
import liGAN


@pytest.fixture(params=['GAN', 'CGAN', 'VAEGAN', 'CVAEGAN'])
def solver(request):
    return getattr(
        liGAN.training, request.param + 'Solver'
    )(
        train_file='data/molportFULL_rand_test0_50.types',
        test_file='data/molportFULL_rand_test0_50.types',
        data_kws=dict(
            data_root='data/molport',
            batch_size=50,
            rec_map_file='data/my_rec_map',
            lig_map_file='data/my_lig_map',
            resolution=1.0,
            grid_size=8,
            shuffle=False,
            random_rotation=False,
            random_translation=0,
            rec_molcache=None,
            lig_molcache=None,
        ),
        gen_model_kws=dict(
            n_filters=5,
            n_levels=3,
            conv_per_level=1,
            batch_norm=2,
            n_latent=128,
            init_conv_pool=False,
            skip_connect=True,
        ),
        disc_model_kws=dict(

        ),
        loss_fn_kws=dict(
            types=dict(
                recon_loss='2',
                gan_loss='w'
            )
        ),
        gen_optim_kws=dict(
            type='RMSprop',
            lr=1e-8,
            momentum=0,
            clip_gradient=0,
            n_train_iters=2,
        ),
        disc_optim_kws=dict(
            type='RMSprop',
            lr=1e-7,
            momentum=0,
            clip_gradient=1,
            n_train_iters=2,
        ),
        atom_fitting_kws=dict(),
        out_prefix='tests/TEST',
        device='cuda'
    )


class TestGANSolver(object):

    def test_solver_init(self, solver):
        assert solver.curr_iter == 0
        for name, params in solver.named_parameters():
            if name.endswith('weight'):
                assert params.detach().norm().cpu() > 0, 'weights are zero'
            elif name.endswith('bias'):
                assert params.detach().norm().cpu() == 0, 'bias is non-zero'

    def test_solver_disc_forward_real(self, solver):
        loss, metrics = solver.disc_forward(
            solver.train_data, real=True
        )
        assert not isclose(0, loss.item()), 'loss is zero'
        assert not isnan(loss.item()), 'loss is nan'

    def test_solver_disc_forward_gen(self, solver):
        loss, metrics = solver.disc_forward(
            solver.train_data, real=False
        )
        assert not isclose(0, loss.item()), 'loss is zero'
        assert not isnan(loss.item()), 'loss is nan'

    def test_solver_gen_forward(self, solver):
        loss, metrics = solver.gen_forward(solver.train_data)
        assert not isclose(0, loss.item()), 'loss is zero'
        assert not isnan(loss.item()), 'loss is nan'

    def test_solver_disc_step_real(self, solver):
        metrics_i = solver.disc_step(real=True)
        _, metrics_f = solver.disc_forward(solver.train_data, real=True)
        assert metrics_f['loss'] < metrics_i['loss'], 'loss did not decrease'
        assert metrics_i['disc_grad_norm'] <= 1, 'gradient not normalized'

    def test_solver_disc_step_gen(self, solver):
        metrics_i = solver.disc_step(real=False)
        _, metrics_f = solver.disc_forward(solver.train_data, real=False)
        assert metrics_f['loss'] < metrics_i['loss'], 'loss did not decrease'
        assert metrics_i['disc_grad_norm'] <= 1, 'gradient not normalized'

    def test_solver_gen_step(self, solver):
        metrics_i = solver.gen_step()
        _, metrics_f = solver.gen_forward(solver.train_data)
        assert metrics_f['loss'] < metrics_i['loss'], 'loss did not decrease'

    def test_solver_test(self, solver):
        solver.test(1, fit_atoms=False)
        assert solver.curr_iter == 0
        assert len(solver.metrics) == 2

    def test_solver_test_fit(self, solver):
        solver.test(1, fit_atoms=True)
        assert solver.curr_iter == 0
        assert len(solver.metrics) == 2
        assert 'lig_gen_fit_n_atoms' in solver.metrics

    def test_solver_train(self, solver):
        solver.train(
            max_iter=10,
            test_interval=10,
            n_test_batches=1,
            fit_interval=10,
            save_interval=10,
        )
        assert solver.curr_iter == 10
        print(solver.metrics)         # test train test test_on_train
        assert len(solver.metrics) == (2*1 + 2*10 + 2*1 + 2*2)
        assert 'gan_loss' in solver.metrics
        if isinstance(solver, liGAN.training.VAEGANSolver):
            assert 'recon_loss' in solver.metrics
            assert 'kldiv_loss' in solver.metrics
        loss_i = solver.metrics.loc[( 0,  0, 'test', 'gen'), 'loss'].mean()
        loss_f = solver.metrics.loc[(10, 10, 'test', 'gen'), 'loss'].mean()
        assert (loss_f - loss_i) < 0, 'generator loss did not decrease'
        loss_i = solver.metrics.loc[( 0,  0, 'test', 'disc'), 'loss'].mean()
        loss_f = solver.metrics.loc[(10, 10, 'test', 'disc'), 'loss'].mean()
        assert (loss_f - loss_i) < 0, 'discriminator loss did not decrease'
