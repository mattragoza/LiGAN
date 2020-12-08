import sys, os
import numpy as np
from openbabel import openbabel as ob
ligan_root = os.environ['LIGAN_ROOT']
sys.path.append(ligan_root)
import atom_types
import generate


def test_remove_tensors_circular():
    a = []
    b = [a]
    a.append(b)
    generate.remove_tensors(a)


def test_dkoes_atom_fitter():

    fitter = generate.DkoesAtomFitter(
        dkoes_make_mol=True,
        use_openbabel=False,
    )
    channels = atom_types.get_channels_from_file(
        os.path.join(ligan_root, 'my_lig_map'),
    )
    grid_shape = (len(channels), 48, 48, 48)
    grid = generate.AtomGrid(
        values=np.zeros(grid_shape),
        channels=channels,
        center=np.zeros(3),
        resolution=0.5,
    )
    grid = fitter.fit(grid, [])
    assert grid.info['src_struct'].n_atoms == 0
