import sys, os, pytest
from numpy import isclose

sys.path.insert(0, '.')
import liGAN.molecules as mols
from liGAN.molecules import ob, Molecule
from liGAN.atom_types import Atom, AtomTyper
from liGAN.bond_adding import BondAdder, get_max_valences


test_sdf_files = [
    'data/O_2_0_0.sdf',
    'data/N_2_0_0.sdf',
    'data/C_2_0_0.sdf',
    'data/benzene.sdf',
    'data/neopentane.sdf',
    'data/sulfone.sdf', #TODO reassign guanidine double bond
    'data/ATP.sdf',
]


prop_ranges = {
    Atom.atomic_num: [6, 7, 8, 15, 16],
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
        assert n_out == n_in, \
            'different num heavy atoms ({} vs {})'.format(n_out, n_in)
    else:
        n_in = in_mol.NumAtoms()
        n_out = out_mol.NumAtoms()
        assert n_out == n_in, \
            'different num atoms ({} vs {})'.format(n_out, n_in)

    return zip(
        iter_atoms(in_mol, omit_h),
        iter_atoms(out_mol, omit_h),
    )


def write_pymol(visited_mols, in_mol):
    pymol_file = 'tests/TEST_' + in_mol.name + '.pymol'
    with open(pymol_file, 'w') as f:
        f.write('load {}\n'.format(write_mols(visited_mols, in_mol)))
        f.write('load {}\n'.format(write_mols(visited_mols, in_mol, 'a')))
        f.write('load {}\n'.format(write_mols(visited_mols, in_mol, 'e')))
        f.write('load {}\n'.format(write_mols(visited_mols, in_mol, 'h')))
        f.write('show_as nb_spheres\n')
        f.write('show sticks\n')
        f.write('util.cbam\n')


def write_mols(visited_mols, in_mol, mode=None):
    write_mols = visited_mols + [in_mol]
    write_mols = [mols.copy_ob_mol(m) for m in write_mols]
    hyb_map = {1:6, 2:7, 3:8}
    hyd_map = {0:1, 1:6, 2:7, 3:8, 4:9}

    for m in write_mols: # make sure we don't reassign these
        m.SetAromaticPerceived(True)
        m.SetHybridizationPerceived(True)

    if mode == 'a': # show bond aromaticity as bond order
        for m in write_mols:
            for a in ob.OBMolAtomIter(m):
                a.SetAtomicNum(1 + 5*a.IsAromatic())
            for b in ob.OBMolBondIter(m):
                b.SetBondOrder(1 + b.IsAromatic())
    
    elif mode == 'e': # show hybridization as element
        for m in write_mols:
            for a in ob.OBMolAtomIter(m):
                a.SetAtomicNum(hyb_map.get(a.GetHyb(), 1))

    elif mode == 'h': # show implicit H count as element
        for m in write_mols:
            for a in ob.OBMolAtomIter(m):
                a.SetAtomicNum(hyd_map[a.GetImplicitHCount()])

    elif mode is not None:
        raise ValueError(mode)

    mol_name = in_mol.name
    if mode:
        mol_name += '_' + mode

    mol_file = 'tests/TEST_{}.sdf'.format(mol_name)
    mols.write_ob_mols_to_sdf_file(mol_file, write_mols)
    return mol_file


class TestBondAdding(object):

    @pytest.fixture(params=[
        #[Atom.h_acceptor, Atom.h_donor],
        #[Atom.h_acceptor, Atom.h_donor, Atom.formal_charge],
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
        mol.AddHydrogens() # this is needed to determine donor/acceptor
        mol.name = os.path.splitext(os.path.basename(sdf_file))[0]
        return mol

    def test_init(self, adder):
        pass

    def test_make_ob_mol(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)
        out_mol, _ = struct.to_ob_mol()

        for i, o in iter_atom_pairs(in_mol, out_mol, typer.omit_h):
            assert o.GetAtomicNum() == i.GetAtomicNum(), 'different elements'
            assert o.GetVector() == i.GetVector(), 'different coordinates'

    def test_add_within_distance(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)
        out_mol, atoms = struct.to_ob_mol()
        adder.add_within_distance(out_mol, atoms, struct)

        for a_i, a_o in iter_atom_pairs(in_mol, out_mol, typer.omit_h):
            for b_i, b_o in iter_atom_pairs(in_mol, out_mol, typer.omit_h):

                in_bonded = bool(in_mol.GetBond(a_i, b_i))
                out_bonded = bool(out_mol.GetBond(a_o, b_o))
                bond_str = '{}-{}'.format(
                    ob.GetSymbol(a_i.GetAtomicNum()),
                    ob.GetSymbol(b_i.GetAtomicNum())
                )
                if in_bonded: # all input bonds should be present in output
                    assert out_bonded, 'missing ' + bond_str + ' bond'

    def test_set_min_h_counts(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)
        out_mol, atoms = struct.to_ob_mol()
        adder.add_within_distance(out_mol, atoms, struct)
        adder.set_min_h_counts(out_mol, atoms, struct)

        for i, o in iter_atom_pairs(in_mol, out_mol, typer.omit_h):
            # all H donors should have at least one hydrogen
            assert o.GetImplicitHCount() >= i.IsHbondDonor(), \
                'H donor has no hydrogen(s)'

    def test_set_formal_charges(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)
        out_mol, atoms = struct.to_ob_mol()
        adder.add_within_distance(out_mol, atoms, struct)
        adder.set_min_h_counts(out_mol, atoms, struct)
        adder.set_formal_charges(out_mol, atoms, struct)

        for i, o in iter_atom_pairs(in_mol, out_mol, typer.omit_h):
            assert o.GetFormalCharge() == i.GetFormalCharge(), \
                'incorrect formal charge'

    def test_remove_bad_valences(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)
        out_mol, atoms = struct.to_ob_mol()
        adder.add_within_distance(out_mol, atoms, struct)
        adder.set_min_h_counts(out_mol, atoms, struct)
        adder.set_formal_charges(out_mol, atoms, struct)
        adder.remove_bad_valences(out_mol, atoms, struct)
        max_vals = get_max_valences(atoms)

        for o in iter_atoms(out_mol, typer.omit_h):
            assert o.GetExplicitValence() <= max_vals.get(o.GetIdx(), 1), \
                'invalid valence'

    def test_remove_bad_geometry(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)
        out_mol, atoms = struct.to_ob_mol()
        adder.add_within_distance(out_mol, atoms, struct)
        adder.set_min_h_counts(out_mol, atoms, struct)
        adder.set_formal_charges(out_mol, atoms, struct)
        adder.remove_bad_valences(out_mol, atoms, struct)
        adder.remove_bad_geometry(out_mol)

    def test_set_aromaticity(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)
        out_mol, atoms = struct.to_ob_mol()
        adder.add_within_distance(out_mol, atoms, struct)
        adder.set_min_h_counts(out_mol, atoms, struct)
        adder.set_formal_charges(out_mol, atoms, struct)
        adder.remove_bad_valences(out_mol, atoms, struct)
        adder.remove_bad_geometry(out_mol)
        adder.set_aromaticity(out_mol, atoms, struct)

        for i, o in iter_atom_pairs(in_mol, out_mol, typer.omit_h):
            assert o.IsAromatic() == i.IsAromatic(), 'different aromaticity'

    def test_add_bonds(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)
        out_mol, atoms = struct.to_ob_mol()
        out_mol, visited_mols = adder.add_bonds(out_mol, atoms, struct)

        write_pymol(visited_mols, in_mol)
        for t in struct.atom_types:
            print(t)

        # check bonds between atoms in typed structure
        for in_a, out_a in iter_atom_pairs(in_mol, out_mol, typer.omit_h):
            for in_b, out_b in iter_atom_pairs(in_mol, out_mol, typer.omit_h):

                in_bond = in_mol.GetBond(in_a, in_b)
                out_bond = out_mol.GetBond(out_a, out_b)
                bstr = '{}-{}'.format(
                    ob.GetSymbol(in_a.GetAtomicNum()),
                    ob.GetSymbol(in_b.GetAtomicNum())
                )
                assert (
                    bool(out_bond) == bool(in_bond)
                ), 'different {} bond presence'.format(bstr)
                if in_bond and out_bond:
                    if in_bond.IsAromatic(): # allow diff kekule structures
                        assert out_bond.IsAromatic(), \
                            'different {} bond aromaticity'.format(bstr)
                    else: # mols should have same bond orders
                        assert (
                            in_bond.GetBondOrder() == out_bond.GetBondOrder()
                        ), 'different {} bond orders'.format(bstr)

        # check whether correct num hydrogens were added
        n_in = in_mol.NumAtoms()
        n_out = out_mol.NumAtoms()
        assert n_out == n_in, 'different num atoms ({} vs {})'.format(
            n_out, n_in
        )

    def test_convert_mol(self, in_mol):
        mol = Molecule.from_ob_mol(in_mol)
        out_mol = mol.to_ob_mol()

    def test_make_mol(self, adder, typer, in_mol):
        add_struct = typer.make_struct(in_mol)
        out_mol, _, _ = adder.make_mol(add_struct)
        in_mol = Molecule.from_ob_mol(in_mol)

        n_in = in_mol.n_atoms
        n_out = out_mol.n_atoms
        assert n_out == n_in, \
            'different num atoms ({} vs {})'.format(n_out, n_in)

        mols.Chem.SanitizeMol(in_mol)
        mols.Chem.SanitizeMol(out_mol)
        morgan_sim = mols.get_rd_mol_similarity(out_mol, in_mol, 'morgan')
        assert out_mol.to_smi() == in_mol.to_smi(), \
            'different SMILES strings ({:.3f})'.format(morgan_sim)

        rmsd = out_mol.aligned_rmsd(in_mol)
        assert rmsd < 1.0, 'RMSD too high ({})'.format(rmsd)
