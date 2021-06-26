import sys, os, pytest
import numpy as np
from numpy import isclose, allclose
import torch

sys.path.insert(0, '.')
from liGAN.atom_types import Atom, AtomTyper
from liGAN.atom_grids import AtomGrid, unravel_index


class TestAtomGrid(object):

    @pytest.fixture
    def typer(self):
        return AtomTyper(
            prop_funcs=[Atom.atomic_num],
            prop_ranges=[[8]],
            radius_func=lambda x: 1
        )

    @pytest.fixture
    def grid(self, typer):
        return AtomGrid(
            values=np.random.randn(1, 5, 5, 5),
            center=np.zeros(3),
            resolution=1.0,
            typer=typer,
        )

    def test_init(self, grid):
        assert grid.n_channels == 1, 'incorrect num channels'
        assert grid.size == 5, 'incorrect grid size'

    def test_dimension(self, grid):
        assert grid.dimension == 4.0, 'incorrect grid dimension'

    def test_new_like(self, grid):
        new_grid = grid.new_like(
            values=np.random.randn(1, 5, 5, 5)
        )
        assert new_grid.values is not grid.values, 'same grid values'
        assert new_grid.center is grid.center, 'different center'
        assert new_grid.resolution is grid.resolution, 'different resolution'
        assert new_grid.typer is grid.typer, 'different atom typer'

    def test_to_dx(self, grid):
        dx_files = grid.to_dx('tests/output/TEST')
        print(dx_files)
        assert dx_files == ['tests/output/TEST_atomic_num=8.dx'], \
            'incorrect file names'

    def test_from_dx(self, typer):
        grid = AtomGrid.from_dx('tests/output/TEST', typer)
        assert grid.n_channels == 1, 'incorrect num channels'
        assert grid.size == 5, 'incorrect grid size'

    def test_to_and_from_dx(self, grid):
        dx_files = grid.to_dx('tests/output/TEST')
        new_grid = AtomGrid.from_dx('tests/output/TEST', grid.typer)
        print(grid.values)
        print()
        print(new_grid.values)
        assert allclose(new_grid.values, grid.values), 'different values'

    def test_get_coords(self, grid):
        idx = torch.arange(grid.size**3)
        idx_xyz = unravel_index(idx, grid.shape[1:])
        assert len(idx_xyz.shape) == 2
        assert idx_xyz.shape[1] == 3
        coords = grid.get_coords(idx_xyz)

    def test_prop_values(self, grid):
        out_values = torch.cat(
            [grid.elem_values] + list(grid.prop_values), dim=0
        )
        assert (out_values == grid.values).all(), 'different values'
