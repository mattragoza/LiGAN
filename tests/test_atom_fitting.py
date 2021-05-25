import sys, os, pytest
from numpy import isclose, allclose
import torch

sys.path.insert(0, '.')
from molgrid import GridMaker, Coords2Grid
from liGAN import molecules as mols
from liGAN.atom_types import Atom, AtomTyper
from liGAN.atom_grids import AtomGrid
from liGAN.atom_fitting import AtomFitter
from liGAN.metrics import compute_struct_rmsd


test_sdf_files = [
    'data/O_2_0_0.sdf',
    'data/N_2_0_0.sdf',
    'data/C_2_0_0.sdf',
    'data/benzene.sdf', #TODO only fitting 1 atom
    'data/neopentane.sdf',
    'data/sulfone.sdf',
    'data/ATP.sdf',
]


def write_pymol(visited_structs, grid, in_mol):
    pymol_file = 'tests/TEST_' + in_mol.name + '_fit.pymol'
    with open(pymol_file, 'w') as f:
        f.write('load {}\n'.format(write_structs(visited_structs, in_mol)))
        f.write('load_group {}, {}\n'.format(*write_grid(grid, in_mol)))
        f.write('show_as nb_spheres\n')
        f.write('show sticks\n')
        f.write('util.cbam\n')
        f.write('set_atom_level 0.5, job_name=TEST')


def write_grid(grid, in_mol):
    dx_prefix = 'tests/TEST_' + in_mol.name + '_lig_fit_0'
    dx_files = grid.to_dx(dx_prefix)
    return dx_prefix + '*.dx', dx_prefix


def write_structs(visited_structs, in_mol):
    visited_mols = [m.to_ob_mol()[0] for m in visited_structs]
    write_mols = visited_mols + [in_mol]
    mol_file = 'tests/TEST_{}_fit.sdf'.format(in_mol.name)
    mols.write_ob_mols_to_sdf_file(mol_file, write_mols)
    return mol_file


class TestAtomFitter(object):

    @pytest.fixture(params=['oad'] )#, 'oadc', 'on', 'oh'])
    def typer(self, request):
        return AtomTyper.get_typer(
            prop_funcs=request.param,
            radius_func=Atom.vdw_radius,
        )

    @pytest.fixture
    def gridder(self):
        return Coords2Grid(GridMaker(
            resolution=0.5,
            dimension=10,
        ))

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
            resolution=gridder.gmaker.get_resolution(),
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
        fitter.init_kernel(0.5, typer)
        assert fitter.kernel.shape[0] == typer.n_elem_types
        assert fitter.kernel.shape[1] % 2 == 1
        assert all(fitter.kernel.sum(dim=(1,2,3)) > 0)

    def test_convolve(self, fitter, grid):
        conv_values = fitter.convolve(
            grid.elem_values, grid.resolution, grid.typer
        )
        dims = (1, 2, 3)
        assert all(
            conv_values.sum(dim=dims) >= grid.elem_values.sum(dim=dims)
        )

    def test_apply_peak_value(self, fitter, grid):
        peak_values = fitter.apply_peak_value(grid.elem_values)
        assert (peak_values <= fitter.peak_value).all()

    def test_sort_grid_points(self, fitter, grid):
        values, idx_xyz, idx_c = fitter.sort_grid_points(grid.elem_values)
        idx_x, idx_y, idx_z = idx_xyz[:,0], idx_xyz[:,1], idx_xyz[:,2]
        assert (grid.elem_values[idx_c, idx_x, idx_y, idx_z] == values).all()

    def test_apply_threshold(self, fitter, grid):
        values, idx_xyz, idx_c = fitter.sort_grid_points(grid.elem_values)
        values, idx_xyz, idx_c = fitter.apply_threshold(values, idx_xyz, idx_c)
        assert (values > fitter.threshold).all()

    def test_suppress_non_max(self, fitter, grid):
        values, idx_xyz, idx_c = fitter.sort_grid_points(grid.elem_values)
        values, idx_xyz, idx_c = fitter.apply_threshold(values, idx_xyz, idx_c)
        coords = grid.get_coords(idx_xyz)
        coords_mat, idx_xyz_mat, idx_c_mat = fitter.suppress_non_max(
            values, coords, idx_xyz, idx_c, grid, matrix=True
        )
        coords_for, idx_xyz_for, idx_c_for = fitter.suppress_non_max(
            values, coords, idx_xyz, idx_c, grid, matrix=False
        )
        assert len(coords_mat) == len(idx_c_mat)
        assert len(coords_for) == len(idx_c_for)
        assert len(coords_mat) == len(coords_for)
        assert len(coords_mat) <= len(coords)
        assert coords_mat.shape[1] == 3
        assert coords_for.shape[1] == 3
        assert (coords_mat == coords_for).all()
        assert (idx_c_mat == idx_c_for).all()

    def test_detect_atoms(self, fitter, grid):
        coords, types = fitter.detect_atoms(grid)
        assert coords.shape == (fitter.n_atoms_detect, 3)
        assert types.shape == (fitter.n_atoms_detect, grid.n_channels)
        assert coords.dtype == types.dtype == grid.dtype
        assert coords.device == types.device == grid.device

    def test_fit_struct(self, fitter, grid):
        struct = grid.info['src_struct']
        fit_struct, fit_grid, visited_structs = fitter.fit_struct(grid)
        write_pymol(visited_structs, grid, struct.info['src_mol'])
        rmsd = compute_struct_rmsd(struct, fit_struct)
        assert rmsd < 0.5, 'RMSD too high ({:.2f})'.format(rmsd)
