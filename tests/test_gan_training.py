import sys, os, pytest

import numpy as np
from numpy import isclose, isnan

import torch
from torch import nn, optim

sys.path.insert(0, '.')
import liGAN


def get_data(split_rec_lig, ligand_only):
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
        split_rec_lig=split_rec_lig,
        ligand_only=ligand_only,
    )
    data.populate('data/molportFULL_rand_test0_1000.types')
    return data


def get_encoder():
    return liGAN.models.Encoder(
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
        n_output=1,
        output_activ_fn=nn.Sigmoid(),
    ).cuda()


def get_decoder():
    return liGAN.models.Decoder(
        n_input=1024,
        grid_dim=2,
        n_channels=20,
        width_factor=2,
        n_levels=3,
        deconv_per_level=1,
        kernel_size=3,
        relu_leak=0.1,
        unpool_type='n',
        unpool_factor=2,
        n_output=19,
    ).cuda()


def L2_loss(y_pred, y_true):
    return ((y_true - y_pred)**2).sum() / 2 / y_true.shape[0]


class TestGANSolver(object):

    @pytest.fixture
    def solver(self):
        return liGAN.training.GANSolver(
            train_data=get_data(split_rec_lig=False, ligand_only=True),
            test_data=get_data(split_rec_lig=False, ligand_only=True),
            gen_model=get_decoder(),
            disc_model=get_encoder(),
            loss_fn=nn.BCELoss(),
            optim_type=optim.Adam,
            lr=1e-4,
            betas=(0.9, 0.999),
            save_prefix='TEST_GAN'
        )

    def test_solver_init(self, solver):
        assert solver.curr_iter == 0
        for model in (solver.gen_model, solver.disc_model):
            for params in model.parameters():
                assert params.detach().norm().cpu() > 0, 'params are zero'

    def test_solver_disc_forward_real(self, solver):
        predictions, loss = solver.disc_forward(solver.train_data, real=True)
        assert predictions.detach().norm().cpu() > 0, 'predictions are zero'
        assert not isclose(0, loss.item()), 'loss is zero'
        assert not isnan(loss.item()), 'loss is nan'

    def test_solver_disc_forward_gen(self, solver):
        predictions, loss = solver.disc_forward(solver.train_data, real=False)
        assert predictions.detach().norm().cpu() > 0, 'predictions are zero'
        assert not isclose(0, loss.item()), 'loss is zero'
        assert not isnan(loss.item()), 'loss is nan'

    def test_solver_gen_forward(self, solver):
        predictions, loss = solver.gen_forward(solver.train_data)
        assert predictions.detach().norm().cpu() > 0, 'predictions are zero'
        assert not isclose(0, loss.item()), 'loss is zero'
        assert not isnan(loss.item()), 'loss is nan'

    def test_solver_disc_step_real(self, solver):
        _, loss0 = solver.disc_step(real=True)
        _, loss1 = solver.disc_forward(solver.train_data, real=True)
        assert loss1.detach() < loss0.detach(), 'loss did not decrease'

    def test_solver_disc_step_gen(self, solver):
        _, loss0 = solver.disc_step(real=False)
        _, loss1 = solver.disc_forward(solver.train_data, real=False)
        assert loss1.detach() < loss0.detach(), 'loss did not decrease'

    def test_solver_gen_step(self, solver):
        _, loss0 = solver.gen_step()
        _, loss1 = solver.gen_forward(solver.train_data)
        assert loss1.detach() < loss0.detach(), 'loss did not decrease'

if False:

    def test_solver_test(self, solver):
        solver.test(n_iters=1)
        assert solver.curr_iter == 0
        assert len(solver.metrics) == 1

    def test_solver_train(self, solver):
        solver.train(
            n_iters=10,
            test_interval=10,
            test_iters=10,
            save_interval=10,
            print_interval=1,
        )
        assert solver.curr_iter == 10
        assert len(solver.metrics) == 13
        loss_i = solver.metrics.loc[( 0, 'test'), 'loss']
        loss_f = solver.metrics.loc[(10, 'test'), 'loss']
        assert (loss_f - loss_i) < 0
