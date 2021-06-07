import sys, os, pytest, time
from numpy import isclose, isnan
import pandas as pd
from torch import optim

sys.path.insert(0, '.')
import liGAN


@pytest.fixture(params=['AE']) #, 'CE', 'VAE', 'CVAE'])
def solver(request):
    model_type = request.param
    return getattr(
        liGAN.training, model_type + 'Solver'
    )(
        train_file='data/it2_tt_0_lowrmsd_valid_mols_head1.types',
        test_file='data/it2_tt_0_lowrmsd_valid_mols_head1.types',
        data_kws=dict(
            data_root='data/crossdock2020',
            batch_size=10,
            rec_typer='on-1',
            lig_typer='on-1',
            resolution=0.5,
            grid_size=1,
            shuffle=False,
            random_rotation=False,
            random_translation=0,
            rec_molcache=None,
            lig_molcache=None,
        ),
        gen_model_kws=dict(
            n_filters=1,
            n_levels=1,
            conv_per_level=1,
            batch_norm=0,
            n_latent=1,
            init_conv_pool=False,
            skip_connect=True,
        ),
        disc_model_kws=None,
        loss_fn_kws=dict(
            types=dict(recon_loss='2'),
            weights=dict(kldiv_loss=0.1, recon_loss=1.0)
        ),
        gen_optim_kws=dict(
            type='Adam',
            lr=1e-5,
            betas=(0.9, 0.999),
        ),
        disc_optim_kws=None,
        atom_fitting_kws=dict(),
        bond_adding_kws=dict(),
        out_prefix='tests/output/TEST_' + model_type,
        device='cuda'
    )


class TestSolver(object):

    def test_solver_init(self, solver):
        assert solver.curr_iter == 0
        for name, params in solver.named_parameters():
            if name.endswith('weight'):
                assert params.detach().norm().cpu() > 0, 'weights are zero'
            elif name.endswith('bias'):
                pass
                #assert params.detach().norm().cpu() == 0, 'bias is non-zero'

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

    def asdf_test_solver_test_fit(self, solver):
        solver.test(1, fit_atoms=True)
        assert solver.curr_iter == 0
        assert len(solver.metrics) == 1
        assert 'lig_gen_fit_type_diff' in solver.metrics

    def test_solver_train(self, solver):
        max_iter = 100
        test_interval = 10
        n_test_batches = 1
        fit_interval = 0
        save_interval = 100

        t0 = time.time()
        solver.train(
            max_iter,
            test_interval,
            n_test_batches,
            fit_interval,
            save_interval
        )
        t_delta = time.time() - t0

        pd.set_option('display.max_columns', 100)
        print(solver.metrics)
        assert solver.curr_iter == max_iter
        assert len(solver.metrics) == (
            max_iter + 1 + (max_iter//test_interval + 1) * n_test_batches
        )
        assert 'recon_loss' in solver.metrics
        if isinstance(solver, liGAN.training.VAESolver):
            assert 'kldiv_loss' in solver.metrics
        loss_i = solver.metrics.loc[(0, 'train'), 'loss'].mean()
        loss_f = solver.metrics.loc[(max_iter, 'train'), 'loss'].mean()
        assert loss_f < loss_i, 'loss did not decrease'
        assert t_delta < 10, 'too slow'
