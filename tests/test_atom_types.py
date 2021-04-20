import sys, os, pytest
from numpy import isclose

sys.path.insert(0, '.')
from liGAN.molecules import read_ob_mols_from_file
from liGAN.atom_types import make_one_hot, AtomTyper, ob


def test_make_one_hot():

    assert make_one_hot(0, [], other=False) == []
    assert make_one_hot(0, [], other=True) == [1]

    assert make_one_hot(0, [1], other=False) == [0]
    assert make_one_hot(1, [1], other=False) == [1]
    assert make_one_hot(0, [1], other=True) == [0, 1]
    assert make_one_hot(1, [1], other=True) == [1, 0]

    assert make_one_hot(0, [0, 1], other=False) == [1, 0]
    assert make_one_hot(1, [0, 1], other=False) == [0, 1]
    assert make_one_hot(2, [0, 1], other=False) == [0, 0]
    assert make_one_hot(0, [0, 1], other=True) == [1, 0, 0]
    assert make_one_hot(1, [0, 1], other=True) == [0, 1, 0]
    assert make_one_hot(2, [0, 1], other=True) == [0, 0, 1]


class TestAtomTyper(object):

    @pytest.fixture
    def typer(self):
        return AtomTyper()

    @pytest.fixture
    def benzene(self):
        sdf_file = os.path.join(
            os.environ['LIGAN_ROOT'], 'data', 'benzene.sdf'
        )
        return read_ob_mols_from_file(sdf_file, 'sdf')[0]

    def test_typer_init(self, typer):
        assert len(typer.type_funcs) == 0
        assert len(typer.type_ranges) == 0
        assert typer.n_types == 0

    def test_typer_benzene_radius(self, typer, benzene):
        for atom in ob.OBMolAtomIter(benzene):
            assert typer.get_radius(atom) == 1

    def test_typer_benzene_type_vec(self, typer, benzene):
        for atom in ob.OBMolAtomIter(benzene):
            assert typer.get_type_vector(atom) == []
