import sys, os, pytest
from numpy import isclose

sys.path.insert(0, '.')
from liGAN import molecules as mols
from liGAN.molecules import ob


class TestOBMol(object):

    @pytest.fixture
    def water(self):
        mol = mols.read_ob_mols_from_file('tests/input/O_2_0_0.sdf', 'sdf')[0]
        mol.AddHydrogens()
        return mol

    @pytest.fixture
    def ammonia(self):
        mol = mols.read_ob_mols_from_file('tests/input/N_2_0_0.sdf', 'sdf')[0]
        mol.AddHydrogens()
        return mol

    @pytest.fixture
    def methane(self):
        mol = mols.read_ob_mols_from_file('tests/input/C_2_0_0.sdf', 'sdf')[0]
        mol.AddHydrogens()
        return mol

    @pytest.fixture
    def benzene(self):
        mol = mols.read_ob_mols_from_file('tests/input/benzene.sdf', 'sdf')[0]
        mol.AddHydrogens()
        return mol

    def test_water_imp_hs(self, water):

        water.DeleteHydrogens()
        assert water.NumAtoms() == 1
        assert not water.HasHydrogensAdded()
        atoms = list(ob.OBMolAtomIter(water))

        for atom in atoms:
            assert atom.GetImplicitHCount() == 2
            assert atom.GetExplicitValence() == 0
            assert atom.GetTotalValence() == 2
            assert atom.GetExplicitDegree() == 0
            assert atom.GetHvyDegree() == 0
            assert atom.GetTotalDegree() == 2
            assert ob.GetMaxBonds(atom.GetAtomicNum()) == 2
            atom.SetImplicitHCount(1)

        water.AddHydrogens()
        assert water.NumAtoms() == 2
        assert water.HasHydrogensAdded()

        for atom in atoms:
            assert atom.GetImplicitHCount() == 0
            assert atom.GetExplicitValence() == 1
            assert atom.GetTotalValence() == 1
            assert atom.GetExplicitDegree() == 1
            assert atom.GetHvyDegree() == 0
            assert atom.GetTotalDegree() == 1
            assert ob.GetMaxBonds(atom.GetAtomicNum()) == 2
            atom.SetImplicitHCount(1)
        
        water.SetHydrogensAdded(False)
        water.AddHydrogens()
        assert water.NumAtoms() == 3

        for atom in atoms:
            assert atom.GetImplicitHCount() == 0
            assert atom.GetExplicitValence() == 2
            assert atom.GetTotalValence() == 2
            assert atom.GetExplicitDegree() == 2
            assert atom.GetHvyDegree() == 0
            assert atom.GetTotalDegree() == 2
            assert ob.GetMaxBonds(atom.GetAtomicNum()) == 2

        assert water.HasNonZeroCoords(), 'all zero coords'

    def test_benzene_imp_hs(self, benzene):

        benzene.DeleteHydrogens()
        atoms = list(ob.OBMolAtomIter(benzene))
        for atom in atoms:
            assert atom.GetImplicitHCount() == 1
            assert atom.GetExplicitValence() == 3
            assert atom.GetTotalValence() == 4
            assert ob.GetMaxBonds(atom.GetAtomicNum()) == 4

        benzene.AddHydrogens()
        for atom in atoms:
            assert atom.GetImplicitHCount() == 0
            assert atom.GetExplicitValence() == 4
            assert atom.GetTotalValence() == 4
            assert ob.GetMaxBonds(atom.GetAtomicNum()) == 4

        assert benzene.HasNonZeroCoords(), 'all zero coords'

    def test_water_to_smi(self, water):
        smi = mols.ob_mol_to_smi(water, 'cnh').rstrip()
        assert smi == '[H]O[H]'

    def test_ammonia_to_smi(self, ammonia):
        smi = mols.ob_mol_to_smi(ammonia, 'cnh').rstrip()
        assert smi == '[H]N([H])[H]'

    def test_methane_to_smi(self, methane):
        smi = mols.ob_mol_to_smi(methane, 'cnh').rstrip()
        assert smi == '[H]C([H])([H])[H]'

    def test_benzene_to_smi(self, benzene):
        smi = mols.ob_mol_to_smi(benzene, 'cnh').rstrip()
        assert smi == '[H]c1c([H])c([H])c(c(c1[H])[H])[H]'
