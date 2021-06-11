import sys, os, pytest, time
from numpy import isclose, isnan
import pandas as pd
from torch import optim
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

sys.path.insert(0, '.')
import liGAN


@pytest.fixture
def train_params():
    return dict(
        max_iter=10,
        test_interval=10,
        n_test_batches=1,
        fit_interval=0,
        norm_interval=0,
        save_interval=10,
    )


@pytest.fixture(params=['GAN', 'CGAN'])
def solver(request):
    model_type = request.param
    return getattr(
        liGAN.training, model_type + 'Solver'
    )(
        train_file='data/it2_tt_0_lowrmsd_valid_mols_head1.types',
        test_file='data/it2_tt_0_lowrmsd_valid_mols_head1.types',
        data_kws=dict(
            data_root='data/crossdock2020',
            batch_size=1,
            rec_typer='on-1',
            lig_typer='on-1',
            resolution=0.5,
            grid_size=23.5,
            shuffle=False,
            random_rotation=False,
            random_translation=0,
            cache_structs=False,
        ),
        gen_model_kws=dict(
            n_filters=32,
            n_levels=4,
            conv_per_level=3,
            spectral_norm=2,
            n_latent=128,
            init_conv_pool=False,
            skip_connect=True,
        ),
        disc_model_kws=dict(
            n_filters=32,
            n_levels=4,
            conv_per_level=3,
            spectral_norm=2,
            n_output=1,
        ),
        loss_fn_kws=dict(
            types=dict(
                recon_loss='2',
                gan_loss='w'
            ),
            weights=dict(
                kldiv_loss=1.0,
                recon_loss=1.0,
                gan_loss=1.0
            )
        ),
        gen_optim_kws=dict(
            type='RMSprop',
            lr=1e-8,
            momentum=0,
            clip_gradient=0,
            n_train_iters=1,
        ),
        disc_optim_kws=dict(
            type='RMSprop',
            lr=1e-7,
            momentum=0,
            clip_gradient=1,
            n_train_iters=2,
        ),
        atom_fitting_kws=dict(),
        bond_adding_kws=dict(),
        out_prefix='tests/output/TEST_' + model_type,
        device='cuda'
    )


class TestGANSolver(object):

    def test_solver_init(self, solver):
        assert solver.curr_iter == 0
        for name, params in solver.named_parameters():
            if name.endswith('weight'):
                assert params.detach().norm().cpu() > 0, 'weights are zero'
            elif name.endswith('bias'):
                pass
                #assert params.detach().norm().cpu() == 0, 'bias is non-zero'

    def test_solver_disc_forward_real(self, solver):
        loss, metrics = solver.disc_forward(
            solver.train_data, grid_type='real'
        )
        assert not isclose(0, loss.item()), 'loss is zero'
        assert not isnan(loss.item()), 'loss is nan'

    def test_solver_disc_forward_gen(self, solver):
        loss, metrics = solver.disc_forward(
            solver.train_data, grid_type='prior'
        )
        assert not isclose(0, loss.item()), 'loss is zero'
        assert not isnan(loss.item()), 'loss is nan'

    def test_solver_gen_forward(self, solver):
        loss, metrics = solver.gen_forward(solver.train_data)
        assert not isclose(0, loss.item()), 'loss is zero'
        assert not isnan(loss.item()), 'loss is nan'

    def test_solver_disc_step_real(self, solver):
        metrics_i = solver.disc_step(grid_type='real')
        _, metrics_f = solver.disc_forward(solver.train_data, grid_type='real')
        assert metrics_f['loss'] < metrics_i['loss'], 'loss did not decrease'

    def test_solver_disc_step_gen(self, solver):
        metrics_i = solver.disc_step(grid_type='prior')
        _, metrics_f = solver.disc_forward(solver.train_data, grid_type='prior')
        assert metrics_f['loss'] < metrics_i['loss'], 'loss did not decrease'

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

    def test_solver_train(self, solver, train_params):

        max_iter = train_params['max_iter']
        test_interval = train_params['test_interval']
        n_test_batches = train_params['n_test_batches']

        t0 = time.time()
        solver.train(**train_params)
        t_delta = time.time() - t0

        print(solver.metrics)
        assert solver.curr_iter == max_iter
        assert len(solver.metrics) == (
            3*(max_iter + 1) + 2*(max_iter//test_interval + 1) * n_test_batches
        )
        assert 'gan_loss' in solver.metrics
        loss_i = solver.metrics.loc[( 0,  0, 'test', 'gen'), 'loss'].mean()
        loss_f = solver.metrics.loc[(10, 20, 'test', 'gen'), 'loss'].mean()
        assert (loss_f - loss_i) < 0, 'generator loss did not decrease'
        loss_i = solver.metrics.loc[( 0,  0, 'test', 'disc'), 'loss'].mean()
        loss_f = solver.metrics.loc[(10, 20, 'test', 'disc'), 'loss'].mean()
        assert (loss_f - loss_i) < 0, 'discriminator loss did not decrease'

        t_per_iter = t_delta / max_iter
        iters_per_day = (24*60*60) / t_per_iter
        k_iters_per_day = int(iters_per_day//1000)
        assert k_iters_per_day >= 100, 'too slow ({:d}k iters/day)'.format(
            k_iters_per_day
        )
