import sys, os, pytest
from numpy import isclose

sys.path.insert(0, '.')
from liGAN.molecules import ob, read_ob_mols_from_file
from liGAN.atom_types import Atom, AtomTyper
from liGAN.bond_adding import BondAdder


class TestBondAdding(object):

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
                [1, 6, 7, 8], [1], [1], [1],
            ],
            radius_func=lambda x: 1
        )

    @pytest.fixture
    def adder(self):
        return BondAdder()

    @pytest.fixture
    def water(self):
        mol = read_ob_mols_from_file('data/O_0_0_0.sdf', 'sdf')[0]
        mol.AddHydrogens()
        return mol

    @pytest.fixture
    def ammonia(self):
        mol = read_ob_mols_from_file('data/N_0_0_0.sdf', 'sdf')[0]
        mol.AddHydrogens()
        return mol

    @pytest.fixture
    def methane(self):
        mol = read_ob_mols_from_file('data/C_0_0_0.sdf', 'sdf')[0]
        mol.AddHydrogens()
        return mol

    @pytest.fixture
    def benzene(self):
        mol = read_ob_mols_from_file('data/benzene.sdf', 'sdf')[0]
        mol.AddHydrogens()
        return mol

    def test_init(self, adder):
        pass

    def test_water_implicit_h(self, water):

        water.DeleteHydrogens()
        atoms = list(ob.OBMolAtomIter(water))
        for atom in atoms:
            assert atom.GetImplicitHCount() == 2
            assert ob.GetMaxBonds(atom.GetAtomicNum()) == 2

        water.AddHydrogens()
        for atom in atoms:
            assert atom.GetImplicitHCount() == 0
            assert ob.GetMaxBonds(atom.GetAtomicNum()) == 2

    def test_water_make_mol(self, adder, typer, water):
        assert water.NumAtoms() == 3
        struct = typer.make_struct(water)
        mol = adder.make_ob_mol(struct)

    def test_ammonia_make_mol(self, adder, typer, ammonia):
        assert ammonia.NumAtoms() == 4
        struct = typer.make_struct(ammonia)
        mol = adder.make_ob_mol(struct)

    def test_methane_make_mol(self, adder, typer, methane):
        assert methane.NumAtoms() == 5
        struct = typer.make_struct(methane)
        mol = adder.make_ob_mol(struct)

    def test_benzene_make_mol(self, adder, typer, benzene):
        assert benzene.NumAtoms() == 12
        struct = typer.make_struct(benzene)
        mol = adder.make_ob_mol(struct)
