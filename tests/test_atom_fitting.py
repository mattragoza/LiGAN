import sys, os, pytest
from numpy import isclose
import torch

sys.path.insert(0, '.')
from molgrid import GridMaker, Coords2Grid
from liGAN import molecules as mols
from liGAN.atom_types import Atom, AtomTyper
from liGAN.atom_grids import AtomGrid
from liGAN.atom_fitting import AtomFitter


test_sdf_files = [
    'data/O_2_0_0.sdf',
    'data/N_2_0_0.sdf',
    'data/C_2_0_0.sdf',
    'data/benzene.sdf',
    'data/neopentane.sdf',
    'data/sulfone.sdf', #TODO reassign guanidine double bond
    'data/ATP.sdf',
]

class TestAtomFitter(object):

    @pytest.fixture(params=['oad', 'oadc', 'on', 'oh'])
    def typer(self, request):
        return AtomTyper.get_typer(
            prop_funcs=request.param,
            radius_func=Atom.cov_radius,
        )

    @pytest.fixture
    def gridder(self):
        return Coords2Grid(GridMaker(resolution=0.5, dimension=11.5))

    @pytest.fixture
    def fitter(self):
        return AtomFitter()

    @pytest.fixture(params=test_sdf_files)
    def mol(self, request):
        sdf_file = request.param
        mol = mols.read_ob_mols_from_file(sdf_file, 'sdf')[0]
        mol.AddHydrogens() # this is needed to determine donor/acceptor
        mol.name = os.path.splitext(os.path.basename(sdf_file))[0]
        return mol

    @pytest.fixture
    def struct(self, mol, typer):
        return typer.make_struct(mol, dtype=torch.float32, device='cuda')

    @pytest.fixture
    def grid(self, struct, gridder):
        gridder.center = tuple(float(v) for v in struct.center)
        return AtomGrid(
            values=gridder.forward(
                coords=struct.coords,
                types=struct.types,
                radii=struct.atomic_radii,
            ),
            center=struct.center,
            resolution=0.5,
            typer=struct.typer,
            src_struct=struct
        )

    def test_init(self, fitter):
        pass

    def test_gridder(self, grid):
        assert grid.values.norm().item() > 0, 'empty grid'
        for type_vec in grid.info['src_struct'].types:
            assert all(grid.values[type_vec > 0].sum(dim=(1,2,3)) > 0), \
                'empty grid channel'

    def test_init_kernel(self, typer, fitter):
        assert fitter.kernel is None
        fitter.init_kernel(typer, resolution=0.5)
        assert fitter.kernel.shape[0] == typer.n_elem_types
        assert fitter.kernel.shape[1] % 2 == 1
        assert all(fitter.kernel.sum(dim=(1,2,3)) > 0)

    def test_convolve(self, typer, fitter, grid):
        fitter.convolve(grid)
