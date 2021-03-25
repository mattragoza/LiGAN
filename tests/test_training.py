import sys, os, pytest
from numpy import isclose, isnan
from torch import optim

sys.path.insert(0, '.')
import liGAN


@pytest.fixture(params=['AE', 'CE', 'VAE', 'CVAE'])
def solver(request):
    return getattr(
        liGAN.training, request.param + 'Solver'
    )(
        data_root='data/molport',
        train_file='data/molportFULL_rand_test0_50.types',
        test_file='data/molportFULL_rand_test0_50.types',
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
        pool_factor=2,
        n_latent=128,
        init_conv_pool=False,
        skip_connect=True,
        loss_weights=None,
        loss_types={},
        grad_norm_types={},
        optim_type=optim.Adam,
        optim_kws=dict(
            lr=1e-5,
            betas=(0.9, 0.999),
        ),
        atom_fitter_type=liGAN.atom_fitting.AtomFitter,
        atom_fitter_kws=dict(),
        out_prefix='TEST',
        device='cuda'
    )


class TestSolver(object):

    def test_solver_init(self, solver):
        assert solver.curr_iter == 0
        for name, params in solver.named_parameters():
            if name.endswith('weight'):
                assert params.detach().norm().cpu() > 0, 'weights are zero'
            elif name.endswith('bias'):
                assert params.detach().norm().cpu() == 0, 'bias is non-zero'

    def test_solver_forward(self, solver):
        loss, metrics = solver.forward(solver.train_data)
        assert not isclose(0, loss.item()), 'loss is zero'
        assert not isnan(loss.item()), 'loss is nan'

    def test_solver_step(self, solver):
        metrics_i = solver.step()
        _, metrics_f = solver.forward(solver.train_data)
        assert metrics_f['loss'] < metrics_i['loss'], 'loss did not decrease'

    def test_solver_test(self, solver):
        solver.test(1, fit_atoms=False)
        assert solver.curr_iter == 0
        assert len(solver.metrics) == 1

    def test_solver_test_fit(self, solver):
        solver.test(1, fit_atoms=True)
        assert solver.curr_iter == 0
        assert len(solver.metrics) == 1
        assert 'lig_gen_fit_type_diff' in solver.metrics

    def test_solver_train(self, solver):
        solver.train(
            max_iter=10,
            test_interval=10,
            n_test_batches=1,
            fit_interval=10,
            save_interval=10,
        )
        assert solver.curr_iter == 10
        assert len(solver.metrics) == (1 + 10 + 1 + 1)
        assert 'recon_loss' in solver.metrics
        if isinstance(solver, liGAN.training.VAESolver):
            assert 'kldiv_loss' in solver.metrics
        loss_i = solver.metrics.loc[( 0, 'test'), 'loss'].mean()
        loss_f = solver.metrics.loc[(10, 'test'), 'loss'].mean()
        assert loss_f < loss_i, 'loss did not decrease'
