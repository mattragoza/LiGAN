import sys, os, pytest
from numpy import isclose

sys.path.insert(0, '.')
import liGAN.molecules as mols
from liGAN.molecules import ob
from liGAN.atom_types import Atom, AtomTyper
from liGAN.bond_adding import BondAdder

test_sdf_files = [
    'data/O_0_0_0.sdf',
    'data/N_0_0_0.sdf',
    'data/C_0_0_0.sdf',
    'data/benzene.sdf',
    'data/neopentane.sdf',
]

class TestBondAdding(object):

    @pytest.fixture(params=[
        [Atom.h_acceptor, Atom.h_donor],
        [Atom.h_acceptor, Atom.h_donor, Atom.formal_charge],
        [Atom.h_degree],
        [],
    ])
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
        return BondAdder(
            min_bond_len=0.01
        )

    @pytest.fixture(params=test_sdf_files)
    def ob_mol(self, request):
        sdf_file = request.param
        mol = mols.read_ob_mols_from_file(sdf_file, 'sdf')[0]
        mol.AddHydrogens()
        mol.name = os.path.splitext(os.path.basename(sdf_file))[0]
        return mol

    def test_init(self, adder):
        pass

    def test_make_ob_mol(self, adder, typer, ob_mol):
        struct = typer.make_struct(ob_mol)
        out_mol = adder.make_ob_mol(struct)[0]
        for in_atom, out_atom in zip(
            ob.OBMolAtomIter(ob_mol), ob.OBMolAtomIter(out_mol)
        ):
            assert out_atom.GetAtomicNum() == in_atom.GetAtomicNum()
            assert out_atom.GetVector() == in_atom.GetVector()

    def test_set_atom_props(self, adder, typer, ob_mol):
        struct = typer.make_struct(ob_mol)
        out_mol, atoms = adder.make_ob_mol(struct)
        adder.set_atom_properties(out_mol, atoms, struct)
        ob_mol.DeleteHydrogens()
        for in_atom, out_atom in zip(
            ob.OBMolAtomIter(ob_mol), ob.OBMolAtomIter(out_mol)
        ):
            assert out_atom.IsAromatic() == in_atom.IsAromatic()
            assert out_atom.GetImplicitHCount() >= in_atom.IsHbondDonor()
            assert out_atom.GetImplicitHCount() > ~in_atom.IsHbondAcceptor()

        assert out_mol.HasAromaticPerceived()
        assert out_mol.NumBonds() == 0

    def test_add_within_dist(self, adder, typer, ob_mol):
        struct = typer.make_struct(ob_mol)
        out_mol, atoms = adder.make_ob_mol(struct)
        adder.add_within_distance(out_mol, atoms, struct)
        in_mol = ob_mol
        mols.write_ob_mols_to_sdf_file(
            'tests/TEST_{}.sdf'.format(ob_mol.name),
            [in_mol, out_mol]
        )
        assert in_mol.NumAtoms() == out_mol.NumAtoms()
        for i, (in_a, out_a) in enumerate(
            zip(ob.OBMolAtomIter(in_mol), ob.OBMolAtomIter(out_mol))
        ):
            for j, (in_b, out_b) in enumerate(
                zip(ob.OBMolAtomIter(in_mol), ob.OBMolAtomIter(out_mol))
            ):
                if j <= i:
                    continue
                in_bonded = bool(in_mol.GetBond(in_a, in_b))
                out_bonded = bool(out_mol.GetBond(out_a, out_b))
                bstr = '{}-{}'.format(
                    ob.GetSymbol(in_a.GetAtomicNum()),
                    ob.GetSymbol(in_b.GetAtomicNum())
                )
                if in_bonded: assert out_bonded, 'missing ' + bstr + ' bond'

    if False:
        try:
            assert out_mol.NumAtoms() == in_mol.NumAtoms(), 'different num atoms'
            in_smi = mols.ob_mol_to_smi(in_mol)
            out_smi = mols.ob_mol_to_smi(out_mol)
            assert out_smi == in_smi, 'different smiles strings'
        except AssertionError:
            raise
