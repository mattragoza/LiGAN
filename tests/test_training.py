import sys, os, pytest

import numpy as np
from numpy import isclose

import torch
from torch import optim

sys.path.insert(0, '.')
import liGAN


class TestSolver(object):

    @pytest.fixture
    def solver(self):

        data = liGAN.data.AtomGridData(
            data_root='data/molport',
            batch_size=10,
            rec_map_file='data/my_rec_map',
            lig_map_file='data/my_lig_map',
            resolution=1.0,
            dimension=7,
            shuffle=False,
        )
        data.populate('data/molportFULL_rand_test0_1000.types')

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
        ).cuda()

        def loss_fn(y_pred, y_true):
            return ((y_true - y_pred)**2).sum() / 2

        return liGAN.training.Solver(
            data, model, loss_fn, optim.SGD, lr=0.001, momentum=0.9
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
        assert solver.curr_iter == 1
        assert (loss1.detach().cpu() - loss0.detach().cpu()) < 0
