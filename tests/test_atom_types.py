import sys, os, pytest
from numpy import isclose, allclose
import torch
import molgrid

sys.path.insert(0, '.')
from liGAN.molecules import read_ob_mols_from_file
from liGAN.atom_structs import AtomStruct
from liGAN.atom_types import (
    make_one_hot, AtomTyper, Atom, ob
)


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

    @pytest.fixture
    def typer(self):
        return AtomTyper(
            prop_funcs=[
                Atom.atomic_num,
                Atom.aromatic,
                Atom.h_acceptor,
                Atom.h_donor,
            ],
            prop_ranges=[
                [1, 6, 8], [1], [1], [1],
            ],
            radius_func=lambda x: 1,
            explicit_h=False,
        )

    @pytest.fixture
    def benzene(self):
        sdf_file = os.path.join(
            os.environ['LIGAN_ROOT'], 'data', 'benzene.sdf'
        )
        mol = read_ob_mols_from_file(sdf_file, 'sdf')[0]
        mol.AddHydrogens() # NOTE needed to determine donor acceptor
        return mol

    @pytest.fixture
    def ammonia(self):
        sdf_file = os.path.join(
            os.environ['LIGAN_ROOT'], 'data', 'N_2_0_0.sdf'
        )
        mol = read_ob_mols_from_file(sdf_file, 'sdf')[0]
        mol.AddHydrogens() # NOTE needed to determine donor acceptor
        return mol

    def test_typer_init(self, typer):
        assert len(typer.prop_funcs) == 4
        assert len(typer.prop_ranges) == 4
        assert typer.n_types == 6

    def test_get_typer(self):
        AtomTyper.get_typer(prop_funcs='oad', radius_func='v')

    def test_typer_names(self, typer):
        print(list(typer.get_type_names()))

    def test_typer_type_vec(self, typer, benzene):
        for i, atom in enumerate(ob.OBMolAtomIter(benzene)):
            if i < 6: # aromatic carbon
                assert typer.get_type_vector(atom) == [
                    0, 1, 0, 1, 0, 0,
                ]
            else: # non-polar hydrogen
                assert typer.get_type_vector(atom) == [
                    0, 0, 0, 0, 0, 0,
                ]

    def test_typer_atom_type(self, typer, benzene):
        for i, atom in enumerate(ob.OBMolAtomIter(benzene)):
            type_vec = torch.as_tensor(typer.get_type_vector(atom))
            if type_vec.bool().any():
                atom_type = typer.get_atom_type(type_vec)
                assert atom_type.atomic_num == atom.GetAtomicNum()
                assert atom_type.aromatic == atom.IsAromatic()
                assert atom_type.h_acceptor == atom.IsHbondAcceptor()
                assert atom_type.h_donor == atom.IsHbondDonor()

    def test_typer_struct(self, typer, benzene):
        struct = typer.make_struct(benzene)
        for i, type_vec in enumerate(struct.types):
            if i < 6: # aromatic carbon
                assert list(type_vec) == [
                    0, 1, 0, 1, 0, 0,
                ]
            else: # non-polar hydrogen
                assert list(type_vec) == [
                    0, 0, 0, 0, 0, 0,
                ]

    def test_typer_coord_set(self, typer, benzene):
        struct1 = typer.make_struct(benzene)
        coord_set = molgrid.CoordinateSet(benzene, typer)
        struct2 = AtomStruct.from_coord_set(coord_set, typer)
        assert allclose(struct1.coords, struct2.coords)
        assert allclose(struct1.types, struct2.types)
        assert struct1.typer == struct2.typer
        assert struct1.atom_types == struct2.atom_types
        assert allclose(struct1.atomic_radii, struct2.atomic_radii)

    def test_typer_unknown(self, typer, ammonia):
        struct = typer.make_struct(ammonia)
        for i, atom_type in enumerate(struct.atom_types):
            if i == 0: # unknown type N gets cast as O
                assert atom_type.atomic_num == 8
            else:
                assert atom_type.atomic_num == 1

    def test_typer_ex_provider(self, typer):
        data_root = os.path.join(
            os.environ['LIGAN_ROOT'], 'data'
        )
        data_file = os.path.join(data_root, 'benzene.types')
        ex_provider = molgrid.ExampleProvider(
            typer, typer,
            data_root=data_root,
        )
        ex_provider.populate(data_file)
        ex = ex_provider.next_batch(1)[0]
        type_vecs = ex.coord_sets[1].type_vector
        for i, type_vec in enumerate(type_vecs):
            if i < 6: # aromatic carbon
                assert list(type_vec) == [
                    0, 1, 0, 1, 0, 0,
                ]
            else: # non-polar hydrogen
                assert list(type_vec) == [
                    0, 0, 0, 0, 0, 0,
                ]

