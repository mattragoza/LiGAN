import sys, os, pytest
import numpy as numpy
from numpy import isclose
from numpy.linalg import norm
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
        )
        model = model.to(device='cuda')

        def loss_fn(y_pred, y_true):
            return ((y_true - y_pred)**2).sum()

        return liGAN.training.Solver(
            data, model, loss_fn, optim.SGD, lr=0.01, momentum=0.9
        )

    def test_solver_init(self, solver):
        assert solver.curr_iter == 0

    def test_solver_forward(self, solver):
        loss = solver.forward()
        assert not isclose(0, loss.item())

    def test_solver_step(self, solver):
        solver.step()
        assert solver.curr_iter == 1
