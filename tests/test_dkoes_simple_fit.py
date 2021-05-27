import sys, os
import numpy as np
from openbabel import openbabel as ob
ligan_root = os.environ['LIGAN_ROOT']
sys.path.append(ligan_root)
from liGAN import atom_grids, atom_types, atom_fitting


def test_remove_tensors_circular():
    a = []
    b = [a]
    a.append(b)
    atom_fitting.remove_tensors(a)


def test_dkoes_atom_fitter():

    fitter = atom_fitting.DkoesAtomFitter(
        dkoes_make_mol=True,
        use_openbabel=False,
    )
    typer = atom_types.AtomTyper.get_typer('oad', 'v')
    grid_shape = (typer.n_types, 48, 48, 48)
    grid = atom_grids.AtomGrid(
        values=np.zeros(grid_shape),
        center=np.zeros(3),
        resolution=0.5,
        typer=typer,
    )
    grid = fitter.fit(grid, [])
    assert grid.info['src_struct'].n_atoms == 0
