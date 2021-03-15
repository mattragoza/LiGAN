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
        data_root='data/molport',
        train_file='data/molportFULL_rand_test0_1000.types',
        test_file='data/molportFULL_rand_test0_1000.types',
        batch_size=1000,
        rec_map_file='data/my_rec_map',
        lig_map_file='data/my_lig_map',
        resolution=1.0,
        grid_size=8,
        shuffle=False,
        random_rotation=False,
        random_translation=0,
        rec_molcache=None,
        lig_molcache=None,
        n_filters=5,
        width_factor=2,
        n_levels=3,
        conv_per_level=1,
        kernel_size=3,
        relu_leak=0.1,
        pool_type='a',
        unpool_type='n',
        pool_factor=2,
        n_latent=128,
        init_conv_pool=False,
        skip_connect=True,
        loss_weights=None,
        loss_types={'gan_loss': 'w'},
        grad_norm_types={'disc': '2'},
        optim_type=optim.RMSprop,
        optim_kws=dict(lr=1e-8, momentum=0),
        atom_fitter_type=liGAN.atom_fitting.AtomFitter,
        atom_fitter_kws=dict(),
        out_prefix='TEST',
        device='cuda'
    )


class TestGANSolver(object):

    def test_solver_init(self, solver):
        assert solver.curr_iter == 0
        for params in solver.parameters():
            assert params.detach().norm().cpu() > 0, 'params are zero'

    def test_solver_disc_forward_real(self, solver):
        predictions, loss, metrics = solver.disc_forward(
            solver.train_data, real=True
        )
        assert predictions.detach().norm().cpu() > 0, 'predictions are zero'
        assert not isclose(0, loss.item()), 'loss is zero'
        assert not isnan(loss.item()), 'loss is nan'

    def test_solver_disc_forward_gen(self, solver):
        predictions, loss, metrics = solver.disc_forward(
            solver.train_data, real=False
        )
        assert predictions.detach().norm().cpu() > 0, 'predictions are zero'
        assert not isclose(0, loss.item()), 'loss is zero'
        assert not isnan(loss.item()), 'loss is nan'

    def test_solver_gen_forward(self, solver):
        predictions, loss, metrics = solver.gen_forward(solver.train_data)
        assert predictions.detach().norm().cpu() > 0, 'predictions are zero'
        assert not isclose(0, loss.item()), 'loss is zero'
        assert not isnan(loss.item()), 'loss is nan'

    def test_solver_disc_step_real(self, solver):
        metrics0 = solver.disc_step(real=True)
        _, _, metrics1 = solver.disc_forward(solver.train_data, real=True)
        assert metrics1['loss'] < metrics0['loss'], 'loss did not decrease'
        assert isclose(1, metrics0['disc_grad_norm']), 'gradient not normalized'

    def test_solver_disc_step_gen(self, solver):
        metrics0 = solver.disc_step(real=False)
        _, _, metrics1 = solver.disc_forward(solver.train_data, real=False)
        assert metrics1['loss'] < metrics0['loss'], 'loss did not decrease'
        assert isclose(1, metrics0['disc_grad_norm']), 'gradient not normalized'

    def test_solver_gen_step(self, solver):
        metrics0 = solver.gen_step()
        _, _, metrics1 = solver.gen_forward(solver.train_data)
        assert metrics1['loss'] < metrics0['loss'], 'loss did not decrease'

    def test_solver_test(self, solver):
        solver.test(1)
        assert solver.curr_iter == 0
        assert len(solver.metrics) == 2

    def test_solver_train(self, solver):
        solver.train(
            max_iter=10,
            n_gen_train_iters=2,
            n_disc_train_iters=2,
            test_interval=10,
            n_test_batches=1,
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
