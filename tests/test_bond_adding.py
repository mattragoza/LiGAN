import sys, os, pytest
from numpy import isclose

sys.path.insert(0, '.')
import liGAN.molecules as mols
from liGAN.molecules import ob, Molecule
from liGAN.atom_types import Atom, AtomTyper
from liGAN.bond_adding import BondAdder


test_sdf_files = [
    'data/O_2_0_0.sdf',
    'data/N_2_0_0.sdf',
    'data/C_2_0_0.sdf',
    'data/benzene.sdf',
    'data/neopentane.sdf',
    'data/sulfone.sdf',
]


prop_ranges = {
    Atom.atomic_num: [6, 7, 8, 16],
    Atom.aromatic: [1],
    Atom.h_acceptor: [1],
    Atom.h_donor: [1],
    Atom.formal_charge: [-1, 0, 1],
    Atom.h_degree: [0, 1, 2, 3, 4],
}


def iter_atoms(ob_mol, omit_h=False):
    '''
    Iterate over atoms in ob_mol,
    optionally omitting hydrogens.
    '''
    for atom in ob.OBMolAtomIter(ob_mol):
        if omit_h and atom.GetAtomicNum() == 1:
            continue
        yield atom


def iter_atom_pairs(in_mol, out_mol, omit_h=False):
    '''
    Iterate over pairs of atoms in in_mol and
    out_mol, optionally omitting hydrogens.
    '''
    if omit_h:
        n_in = in_mol.NumHvyAtoms()
        n_out = out_mol.NumHvyAtoms()
        assert n_out == n_in, 'different num heavy atoms ({} vs {})'.format(
            n_out, n_in
        )
    else:
        n_in = in_mol.NumAtoms()
        n_out = out_mol.NumAtoms()
        assert n_out == n_in, 'different num atoms ({} vs {})'.format(
            n_out, n_in
        )

    return zip(
        iter_atoms(in_mol, omit_h),
        iter_atoms(out_mol, omit_h),
    )


class TestBondAdding(object):

    @pytest.fixture(params=[
        [Atom.h_acceptor, Atom.h_donor],
        [Atom.h_acceptor, Atom.h_donor, Atom.formal_charge],
        [Atom.h_degree],
    ])
    def typer(self, request):
        prop_funcs = [Atom.atomic_num, Atom.aromatic] + request.param
        return AtomTyper(
            prop_funcs=prop_funcs,
            prop_ranges=[prop_ranges[f] for f in prop_funcs],
            radius_func=lambda x: 1,
            omit_h=True,
        )

    @pytest.fixture
    def adder(self):
        return BondAdder()

    @pytest.fixture(params=test_sdf_files)
    def in_mol(self, request):
        sdf_file = request.param
        mol = mols.read_ob_mols_from_file(sdf_file, 'sdf')[0]
        mol.AddHydrogens() # NOTE needed to determine donor/acceptor
        mol.name = os.path.splitext(os.path.basename(sdf_file))[0]
        return mol

    def test_init(self, adder):
        pass

    def test_make_ob_mol(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)
        out_mol, _ = struct.to_ob_mol()
        for in_atom, out_atom in iter_atom_pairs(in_mol, out_mol, typer.omit_h):
            assert out_atom.GetAtomicNum() == in_atom.GetAtomicNum()
            assert out_atom.GetVector() == in_atom.GetVector()

    def test_set_aromaticity(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)
        out_mol, atoms = struct.to_ob_mol()
        adder.set_aromaticity(out_mol, atoms, struct)
        for in_atom, out_atom in iter_atom_pairs(in_mol, out_mol, typer.omit_h):
            assert out_atom.IsAromatic() == in_atom.IsAromatic()
        assert out_mol.HasAromaticPerceived()

    def test_set_min_h_counts(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)
        out_mol, atoms = struct.to_ob_mol()
        adder.set_min_h_counts(out_mol, atoms, struct)
        for in_atom, out_atom in iter_atom_pairs(in_mol, out_mol, typer.omit_h):
            assert out_atom.GetImplicitHCount() >= in_atom.IsHbondDonor()

    def test_add_within_dist(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)
        out_mol, atoms = struct.to_ob_mol()
        adder.add_within_distance(out_mol, atoms, struct)

        for in_a, out_a in iter_atom_pairs(in_mol, out_mol, typer.omit_h):
            for in_b, out_b in iter_atom_pairs(in_mol, out_mol, typer.omit_h):

                in_bonded = bool(in_mol.GetBond(in_a, in_b))
                out_bonded = bool(out_mol.GetBond(out_a, out_b))
                bstr = '{}-{}'.format(
                    ob.GetSymbol(in_a.GetAtomicNum()),
                    ob.GetSymbol(in_b.GetAtomicNum())
                )
                if in_bonded: assert out_bonded, 'missing ' + bstr + ' bond'

    def test_add_bonds(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)
        out_mol, atoms = struct.to_ob_mol()
        out_mol, visited_mols = adder.add_bonds(out_mol, atoms, struct)

        for t in struct.atom_types: print(t)
        for m in visited_mols: m.AddHydrogens()
        mols.write_ob_mols_to_sdf_file(
            'tests/TEST_{}.sdf'.format(in_mol.name),
            visited_mols + [in_mol],
        )
 
        # check bonds between atoms in typed structure
        for in_a, out_a in iter_atom_pairs(in_mol, out_mol, typer.omit_h):
            for in_b, out_b in iter_atom_pairs(in_mol, out_mol, typer.omit_h):

                in_bond = in_mol.GetBond(in_a, in_b)
                out_bond = out_mol.GetBond(out_a, out_b)
                in_bonded = bool(in_bond)
                out_bonded = bool(out_bond)
                bstr = '{}-{}'.format(
                    ob.GetSymbol(in_a.GetAtomicNum()),
                    ob.GetSymbol(in_b.GetAtomicNum())
                )
                assert (
                    out_bonded == in_bonded
                ), 'different {} bonding'.format(bstr)
                if in_bonded:
                    if in_bond.IsAromatic():
                        assert (
                            in_bond.IsAromatic() == out_bond.IsAromatic()
                        ), 'different {} bond aromaticity'.format(bstr)
                    else: # allow different kekule structures
                        assert (
                            in_bond.GetBondOrder() == out_bond.GetBondOrder()
                        ), 'different {} bond orders'.format(bstr)

        # check whether correct num hydrogens were added
        n_in = in_mol.NumAtoms()
        n_out = out_mol.NumAtoms()
        assert n_out == n_in, 'different num atoms ({} vs {})'.format(
            n_out, n_in
        )

if False:
    def test_make_mol(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)

        out_mol, _, _ = adder.make_mol(struct)

        in_mol.AddHydrogens()
        in_mol = Molecule(adder.convert_ob_mol_to_rd_mol(in_mol))

        assert in_mol.n_atoms == out_mol.n_atoms, 'different num atoms'
        assert in_mol.to_smi() == out_mol.to_smi(), 'different molecules'
        assert in_mol.aligned_rmsd(out_mol) < 1e-5, 'RMSD too high'
