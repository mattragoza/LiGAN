import sys, os, pytest
from numpy import isclose, isnan
from torch import optim

sys.path.insert(0, '.')
import liGAN


def get_data(model_type):
    data = liGAN.data.AtomGridData(
        data_root='data/molport',
        batch_size=1000,
        rec_map_file='data/my_rec_map',
        lig_map_file='data/my_lig_map',
        resolution=1.0,
        dimension=7,
        shuffle=False,
        random_rotation=False,
        random_translation=0,
        split_rec_lig=(model_type in {'CE', 'CVAE'}),
        ligand_only=(model_type in {'AE', 'VAE'}),
        device='cuda'
    )
    data.populate('data/molportFULL_rand_test0_1000.types')
    return data


def get_model(model_type):
    if model_type == 'base':
        model = liGAN.models.Encoder(
            n_channels=19+16,
            grid_dim=8,
            n_filters=5,
            width_factor=2,
            n_levels=3,
            conv_per_level=1,
            kernel_size=3,
            relu_leak=0.1,
            pool_type='a',
            pool_factor=2,
            n_output=1,
        )
    else:
        model = liGAN.models.Generator(
            n_channels_in=dict(
                AE=19, CE=16, VAE=19, CVAE=[16, 19]
            )[model_type],
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
            var_input=dict(
                VAE=0, CVAE=1
            ).get(model_type, None)
        )
    return model.cuda()


@pytest.fixture(params=['base', 'AE', 'CE', 'VAE', 'CVAE'])
def solver(request):
    model_type = request.param
    return dict(
        base=liGAN.training.Solver,
        AE=liGAN.training.AESolver,
        CE=liGAN.training.CESolver,
        VAE=liGAN.training.VAESolver,
        CVAE=liGAN.training.CVAESolver,
    )[model_type](
        train_data=get_data(model_type),
        test_data=get_data(model_type),
        model=get_model(model_type),
        loss_fn=lambda yp, yt: ((yt - yp)**2).sum() / 2 / yt.shape[0],
        optim_type=optim.Adam,
        lr=1e-5,
        betas=(0.9, 0.999),
        save_prefix='TEST'
    )


class TestSolver(object):

    def test_solver_init(self, solver):
        assert solver.curr_iter == 0
        for params in solver.model.parameters():
            assert params.detach().norm().cpu() > 0, 'params are zero'

    def test_solver_forward(self, solver):
        predictions, loss, metrics = solver.forward(solver.train_data)
        assert predictions.detach().norm().cpu() > 0, 'predictions are zero'
        assert not isclose(0, loss.item()), 'loss is zero'
        assert not isnan(loss.item()), 'loss is nan'

    def test_solver_step(self, solver):
        metrics0 = solver.step()
        _, _, metrics1 = solver.forward(solver.train_data)
        assert metrics1['loss'] < metrics0['loss'], 'loss did not decrease'

    def test_solver_test(self, solver):
        solver.test(1)
        assert solver.curr_iter == 0
        assert len(solver.metrics) == 1

    def test_solver_train(self, solver):
        solver.train(
            max_iter=10,
            test_interval=10,
            n_test_batches=1,
            save_interval=10,
        )
        assert solver.curr_iter == 10
        assert len(solver.metrics) == 13
        loss_i = solver.metrics.loc[( 0, 'test'), 'loss'].mean()
        loss_f = solver.metrics.loc[(10, 'test'), 'loss'].mean()
        assert loss_f < loss_i, 'loss did not decrease'
