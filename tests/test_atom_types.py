import sys, os, pytest
from numpy import isclose, allclose
import torch
import molgrid

sys.path.insert(0, '.')
import liGAN.molecules as mols
from liGAN.atom_structs import AtomStruct
from liGAN.atom_types import (
    make_one_hot, AtomTyper, Atom, ob
)


test_sdf_files = [
    'tests/input/O_2_0_0.sdf',
    'tests/input/N_2_0_0.sdf',
    'tests/input/C_2_0_0.sdf',
    'tests/input/benzene.sdf',
    'tests/input/neopentane.sdf',
    'tests/input/sulfone.sdf',
    'tests/input/ATP.sdf',
]

def test_make_one_hot():

    with pytest.raises(AssertionError):
        assert make_one_hot(0, []) == []

    # binary type
    assert make_one_hot(0, [0]) == [1]
    assert make_one_hot(0, [1]) == [0]
    assert make_one_hot(1, [0]) == [0]
    assert make_one_hot(1, [1]) == [1]
    assert make_one_hot(2, [0]) == [0]
    assert make_one_hot(2, [1]) == [0]

    # argmax type, with dummy value
    assert make_one_hot(0, [0, 1]) == [1, 0]
    assert make_one_hot(1, [0, 1]) == [0, 1]
    assert make_one_hot(2, [0, 1]) == [0, 1]
    assert make_one_hot(0, [0, 2]) == [1, 0]
    assert make_one_hot(1, [0, 2]) == [0, 1]
    assert make_one_hot(2, [0, 2]) == [0, 1]


class TestAtomTyper(object):

    @pytest.fixture(params=['oad', 'oadc', 'on', 'oh'])
    def prop_funcs(self, request):
        return request.param

    @pytest.fixture
    def typer(self, prop_funcs):
        return AtomTyper.get_typer(prop_funcs, radius_func=1.0)

    @pytest.fixture
    def rec_typer(self, prop_funcs):
        return AtomTyper.get_typer(
            prop_funcs, radius_func=1.0, rec=True
        )

    @pytest.fixture(params=test_sdf_files)
    def mol(self, request):
        sdf_file = request.param
        mol = mols.read_ob_mols_from_file(sdf_file, 'sdf')[0]
        mol.AddHydrogens() # this is needed to determine donor/acceptor
        mol.name = os.path.splitext(os.path.basename(sdf_file))[0]
        return mol

    def test_typer_defaults(self):
        assert AtomTyper.rec_elem_range == [6, 7, 8, 11, 12, 15, 16, 17, 19, 20, 30]
        assert AtomTyper.lig_elem_range == [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 26]

    def test_typer_init(self, typer):
        assert len(typer.prop_funcs) > 1
        assert len(typer.prop_funcs) == len(typer.prop_ranges)
        assert typer.prop_funcs[0] == Atom.atomic_num
        assert typer.explicit_h == (1 in typer.elem_range)
        assert typer.elem_range[typer.explicit_h:] == typer.lig_elem_range
        for f in typer.prop_funcs:
            assert typer.prop_funcs[typer.prop_idx[f]] == f

    def test_rec_typer_init(self, rec_typer):
        assert rec_typer.elem_range[rec_typer.explicit_h:] == rec_typer.rec_elem_range

    def test_typer_names(self, typer):
        print(list(typer.get_type_names()))

    def test_typer_type_vec(self, typer, mol):
        for i, atom in enumerate(ob.OBMolAtomIter(mol)):
            type_vec = torch.as_tensor(typer.get_type_vector(atom))
            if not type_vec.bool().any():
                continue
            for f, r in zip(typer.prop_funcs, typer.prop_ranges):
                prop_vec = type_vec[typer.type_vec_idx[f]]
                value = f(atom)
                if len(r) > 1: # one-hot vector
                    if value not in r:
                        value = r[-1]
                    assert r[prop_vec.argmax().item()] == value
                else: # binary
                    print(f, r, prop_vec, value)
                    assert bool(prop_vec > 0.5) == value

    def test_typer_atom_type(self, typer, mol):
        for i, atom in enumerate(ob.OBMolAtomIter(mol)):
            type_vec = torch.as_tensor(typer.get_type_vector(atom))
            if not type_vec.bool().any():
                continue
            atom_type = typer.get_atom_type(type_vec)
            for f in [
                Atom.atomic_num,
                Atom.aromatic,
                Atom.h_acceptor,
                Atom.h_donor,
                Atom.h_count,
                Atom.formal_charge,
            ]:
                if f in typer:
                    assert getattr(atom_type, f.__name__) == f(atom)

    def test_typer_coord_set(self, typer, mol):
        struct1 = typer.make_struct(mol, dtype=torch.float32)
        coord_set = molgrid.CoordinateSet(mol, typer)
        struct2 = AtomStruct.from_coord_set(coord_set, typer, dtype=torch.float32)
        assert (struct1.coords == struct2.coords).all(), 'different coords'
        assert (struct1.types == struct2.types).all(), 'different types'
        assert struct1.typer == struct2.typer, 'different typers'
        assert struct1.atom_types == struct2.atom_types, 'different atom types'
        assert (struct1.atomic_radii == struct2.atomic_radii).all(), \
            'different atomic radii'
