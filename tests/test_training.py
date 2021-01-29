import sys, os, pytest

import numpy as np
from numpy import isclose

import torch
from torch import optim

sys.path.insert(0, '.')
import liGAN


def get_data(ligand_only):
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
        ligand_only=ligand_only,
    )
    data.populate('data/molportFULL_rand_test0_1000.types')
    return data


def get_encoder():
    return liGAN.models.Encoder(
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
    ).cuda()


def get_generator(n_channels_in, var_input):
    return liGAN.models.Generator(
        n_channels_in=n_channels_in,
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
        var_input=var_input,
    ).cuda()


def L2_loss(y_true, y_pred):
    return ((y_true - y_pred)**2).sum() / 2 / y_true.shape[0]


class TestSolver(object):

    @pytest.fixture
    def solver(self):
        return liGAN.training.Solver(
            train_data=get_data(False),
            test_data=get_data(False),
            model=get_encoder(),
            loss_fn=L2_loss,
            optim_type=optim.SGD,
            lr=0.001,
            momentum=0.9
        )

    def test_solver_init(self, solver):
        assert solver.curr_iter == 0
        assert solver.curr_test == 0
        for params in solver.model.parameters():
            assert params.detach().norm().cpu() > 0

    def test_solver_forward(self, solver):
        predictions, loss = solver.forward(solver.train_data)
        assert not isclose(0, predictions.detach().norm().cpu())
        assert not isclose(0, loss.item())

    def test_solver_step(self, solver):
        _, loss0 = solver.step()
        _, loss1 = solver.forward(solver.train_data)
        assert solver.curr_iter == 1
        assert (loss1.detach() - loss0.detach()).cpu() < 0

    def test_solver_test(self, solver):
        solver.test(n_iters=1)
        assert solver.curr_iter == 0
        assert solver.curr_test == 1
        assert len(solver.metrics) == 1

    def test_solver_train(self, solver):
        solver.train(
            n_iters=10,
            test_interval=10,
            test_iters=10,
            save_interval=10,
        )
        assert solver.curr_iter == 10
        assert solver.curr_test == 2
        assert len(solver.metrics) == 12


class TestAESolver(object):

    @pytest.fixture
    def solver(self):
        return liGAN.training.AESolver(
            train_data=get_data(True),
            test_data=get_data(True),
            model=get_generator(19, None),
            loss_fn=L2_loss,
            optim_type=optim.Adam,
            lr=0,
            betas=(0.9, 0.999),
        )

    def test_solver_init(self, solver):
        assert solver.curr_iter == 0
        assert solver.curr_test == 0
        for params in solver.model.parameters():
            assert params.detach().norm().cpu() > 0

    def test_solver_forward(self, solver):
        predictions, loss = solver.forward(solver.train_data)
        assert not isclose(0, predictions.detach().norm().cpu())
        assert not isclose(0, loss.item())

    def test_solver_step(self, solver):
        _, loss0 = solver.step()
        _, loss1 = solver.forward(solver.train_data)
        print(loss0, loss1)
        assert solver.curr_iter == 1
        assert (loss1.detach() - loss0.detach()).cpu() < 0

    def test_solver_test(self, solver):
        solver.test(n_iters=1)
        assert solver.curr_iter == 0
        assert solver.curr_test == 1
        assert len(solver.metrics) == 1

    def test_solver_train(self, solver):
        solver.train(
            n_iters=10,
            test_interval=10,
            test_iters=10,
            save_interval=10,
        )
        assert solver.curr_iter == 10
        assert solver.curr_test == 2
        assert len(solver.metrics) == 12
        print(solver.metrics)
        loss_i = solver.metrics.loc[( 0, 'test'), 'loss']
        loss_f = solver.metrics.loc[(10, 'test'), 'loss']
        assert (loss_f - loss_i) < 0
