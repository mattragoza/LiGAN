import sys, os, pytest
from numpy import isclose

sys.path.insert(0, '.')
from liGAN.molecules import read_ob_mols_from_file
from liGAN.atom_types import (
    make_one_hot, UNK, AtomTyper, ob
)


def test_make_one_hot():

    assert make_one_hot(0, []) == []
    assert make_one_hot(0, [UNK]) == [1]

    assert make_one_hot(0, [1]) == [0]
    assert make_one_hot(1, [1]) == [1]
    assert make_one_hot(0, [1, UNK]) == [0, 1]
    assert make_one_hot(1, [1, UNK]) == [1, 0]

    assert make_one_hot(0, [0, 1]) == [1, 0]
    assert make_one_hot(1, [0, 1]) == [0, 1]
    assert make_one_hot(2, [0, 1]) == [0, 0]
    assert make_one_hot(0, [0, 1, UNK]) == [1, 0, 0]
    assert make_one_hot(1, [0, 1, UNK]) == [0, 1, 0]
    assert make_one_hot(2, [0, 1, UNK]) == [0, 0, 1]


class TestUnknown(object):

    def test_eq(self):
        assert 0 == UNK
        assert 1 == UNK
        assert False == UNK
        assert True == UNK
        assert 'a' == UNK

    def test_in(self):
        assert 0 in [UNK]
        assert 1 in [UNK]
        assert False in [UNK]
        assert True in [UNK]
        assert 'a' in [UNK]

    def test_idx(self):
        assert [UNK].index(0) == 0
        assert [UNK].index(1) == 0
        assert [UNK].index(False) == 0
        assert [UNK].index(True) == 0
        assert [UNK].index('a') == 0


class TestAtomTyper(object):

    @pytest.fixture
    def typer(self):
        return AtomTyper(
            type_funcs=[
                ob.OBAtom.GetAtomicNum,
                ob.OBAtom.IsAromatic,
                ob.OBAtom.IsHbondAcceptor,
                ob.OBAtom.IsHbondDonor,
            ],
            type_ranges=[
                [5, 6, 7, 8, 9, 15, 16, 17, 35, 53],
                [1],
                [1],
                [1],
            ],
            radius_func=lambda x: 1
        )

    @pytest.fixture
    def benzene(self):
        sdf_file = os.path.join(
            os.environ['LIGAN_ROOT'], 'data', 'benzene.sdf'
        )
        return read_ob_mols_from_file(sdf_file, 'sdf')[0]

    def test_typer_init(self, typer):
        assert len(typer.type_funcs) == 4
        assert len(typer.type_ranges) == 4
        assert typer.n_types == 13

    def test_typer_benzene_type_vec(self, typer, benzene):
        for i, atom in enumerate(ob.OBMolAtomIter(benzene)):
            if i < 6: # aromatic carbon
                assert typer.get_type_vector(atom) == [
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 0, 0,
                ]
            else: # non-polar hydrogen
                assert typer.get_type_vector(atom) == [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0,
                ]

