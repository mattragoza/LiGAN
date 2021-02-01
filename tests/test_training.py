import sys, os, pytest

import numpy as np
from numpy import isclose

import torch
from torch import optim

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


def L2_loss(y_pred, y_true):
    return ((y_true - y_pred)**2).sum() / 2 / y_true.shape[0]


class TestSolver(object):

    @pytest.fixture
    def solver(self):
        return liGAN.training.Solver(
            train_data=get_data(split_rec_lig=False, ligand_only=False),
            test_data=get_data(split_rec_lig=False, ligand_only=False),
            model=get_encoder(),
            loss_fn=L2_loss,
            optim_type=optim.SGD,
            lr=0.001,
            momentum=0.9
        )

    def test_solver_init(self, solver):
        assert solver.curr_iter == 0
        for params in solver.model.parameters():
            assert params.detach().norm().cpu() > 0

    def test_solver_forward(self, solver):
        predictions, loss = solver.forward(solver.train_data)
        assert not isclose(0, predictions.detach().norm().cpu())
        assert not isclose(0, loss.item())

    def test_solver_step(self, solver):
        _, loss0 = solver.step()
        _, loss1 = solver.forward(solver.train_data)
        assert solver.curr_iter == 0
        assert (loss1.detach() - loss0.detach()).cpu() < 0

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


class TestAESolver(object):

    @pytest.fixture
    def solver(self):
        return liGAN.training.AESolver(
            train_data=get_data(split_rec_lig=False, ligand_only=True),
            test_data=get_data(split_rec_lig=False, ligand_only=True),
            model=get_generator(n_channels_in=19, var_input=None),
            loss_fn=L2_loss,
            optim_type=optim.Adam,
            lr=1e-4,
            betas=(0.9, 0.999),
        )

    def test_solver_init(self, solver):
        assert solver.curr_iter == 0
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
        assert solver.curr_iter == 0
        assert (loss1.detach() - loss0.detach()).cpu() < 0

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


class TestCESolver(object):

    @pytest.fixture
    def solver(self):
        return liGAN.training.CESolver(
            train_data=get_data(split_rec_lig=True, ligand_only=False),
            test_data=get_data(split_rec_lig=True, ligand_only=False),
            model=get_generator(n_channels_in=16, var_input=None),
            loss_fn=L2_loss,
            optim_type=optim.Adam,
            lr=1e-4,
            betas=(0.9, 0.999),
        )

    def test_solver_init(self, solver):
        assert solver.curr_iter == 0
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
        assert solver.curr_iter == 0
        assert (loss1.detach() - loss0.detach()).cpu() < 0

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
