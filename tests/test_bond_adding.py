import sys, os, pytest
import numpy as np
from numpy import isclose

sys.path.insert(0, '.')
import liGAN.molecules as mols
from liGAN.molecules import ob, Molecule
from liGAN.atom_types import Atom, AtomTyper
from liGAN.bond_adding import BondAdder, get_max_valences, reachable, compare_bonds


test_sdf_files = [
    'tests/input/O_2_0_0.sdf',
    'tests/input/N_2_0_0.sdf',
    'tests/input/C_2_0_0.sdf',
    'tests/input/benzene.sdf',
    'tests/input/neopentane.sdf',
    #'tests/input/sulfone.sdf', #TODO reassign guanidine double bond
    'tests/input/ATP.sdf',
    #'tests/input/buckyball.sdf', # takes a long time
    'tests/input/4fic_C_0UL.sdf',
    #'tests/input/3el8_B_rec_4fic_0ul_lig_tt_docked_13.sdf.gz',
]

test_typer_fns = ['oadc']


def iter_atoms(ob_mol, explicit_h=True):
    '''
    Iterate over atoms in ob_mol,
    optionally omitting hydrogens.
    '''
    for atom in ob.OBMolAtomIter(ob_mol):
        if not explicit_h and atom.GetAtomicNum() == 1:
            continue
        yield atom


def iter_atom_pairs(in_mol, out_mol, explicit_h=False):
    '''
    Iterate over pairs of atoms in in_mol and
    out_mol, optionally omitting hydrogens.
    '''
    if not explicit_h:
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
        iter_atoms(in_mol, explicit_h),
        iter_atoms(out_mol, explicit_h),
    )


def write_ob_pymol(visited_mols, in_mol):
    write_pymol(visited_mols, in_mol, mol_type='ob')


def write_rd_pymol(visited_mols, in_mol):
    write_pymol(visited_mols, in_mol, mol_type='rd')


def write_pymol(visited_mols, in_mol, mol_type):
    if mol_type == 'ob':
        mol_name = in_mol.name
        write_mols = write_ob_mols
    else:
        mol_name = in_mol.info['ob_mol'].name
        write_mols = write_rd_mols
    pymol_file = f'tests/output/TEST_{mol_type}_{mol_name}.pymol'
    with open(pymol_file, 'w') as f:
        write = lambda mode: write_mols(
            mol_name, visited_mols + [in_mol], mode
        )
        f.write('load {}\n'.format(write(None)))
        f.write('load {}\n'.format(write('o')))
        f.write('load {}\n'.format(write('e')))
        f.write('load {}\n'.format(write('n')))
        f.write('load {}\n'.format(write('a')))
        f.write('load {}\n'.format(write('d')))
        f.write('show_as nb_spheres\n')
        f.write('show sticks\n')
        f.write('util.cbam\n')
        f.write('color white, (name H)\n')
        f.write('color red, (name Md)\n')
        f.write('color orange, (name No)\n')
        f.write('color yellow, (name Lr)\n')
        f.write('color green, (name Rf)\n')
        f.write('color blue, (name Db)\n')
        f.write('color magenta, (name Sg)\n')


def write_rd_mols(mol_name, rd_mols, mode=None):

    # color molecule by atomic properties by setting
    #  the element based on the property value
    value_map = {0:1, 1:101, 2:102, 3:103, 4:104, 5:105, 6:106}

    write_mols = [mols.Molecule(m) for m in rd_mols]

    for m in write_mols:
        for a in m.GetAtoms():

            if mode == 'o': # aromaticity
                a.SetAtomicNum(value_map[a.GetIsAromatic()])

            elif mode == 'e': # hybridization
                a.SetAtomicNum(value_map[a.GetHybridization()])

            elif mode == 'n': # implicit H count
                a.SetAtomicNum(value_map[a.GetNumExplicitHs()])

            elif mode == 'a': # hydrogen acceptor
                a.SetAtomicNum(value_map[False])

            elif mode == 'd': # hydrogen donor
                a.SetAtomicNum(value_map[False])

            elif mode is not None:
                raise ValueError(mode)

        if mode == 'o': # aromatic bonds
            for b in m.GetBonds():
                if b.GetIsAromatic():
                    b.SetBondType(mols.Chem.BondType.DOUBLE)
                else:
                    b.SetBondType(mols.Chem.BondType.SINGLE)

    if mode:
        mol_name += '_' + mode

    mol_file = 'tests/output/TEST_rd_{}.sdf'.format(mol_name)
    mols.write_rd_mols_to_sdf_file(mol_file, write_mols)
    return mol_file


def write_ob_mols(mol_name, ob_mols, mode=None):

    # color molecule by atomic properties by setting
    #  the element based on the property value
    value_map = {0:1, 1:101, 2:102, 3:103, 4:104, 5:105, 6:106}

    write_mols = [mols.copy_ob_mol(m) for m in ob_mols]

    for m in write_mols:
        for a in ob.OBMolAtomIter(m):

            if mode == 'o': # aromaticity
                a.SetAtomicNum(value_map[a.IsAromatic()])

            elif mode == 'e': # hybridization
                a.SetAtomicNum(value_map[mols.ob_hyb_to_rd_hyb(a)])

            elif mode == 'n': # implicit H count
                a.SetAtomicNum(value_map[a.GetImplicitHCount()])

            elif mode == 'a': # hydrogen acceptor
                a.SetAtomicNum(value_map[a.IsHbondAcceptor()])

            elif mode == 'd': # hydrogen donor
                a.SetAtomicNum(value_map[a.IsHbondDonor()])

            elif mode is not None:
                raise ValueError(mode)

        if mode == 'o': # aromatic bonds
            for b in ob.OBMolBondIter(m):
                b.SetBondOrder(1 + b.IsAromatic())
    if mode:
        mol_name += '_' + mode

    mol_file = f'tests/output/TEST_ob_{mol_name}.sdf'
    mols.write_ob_mols_to_sdf_file(mol_file, write_mols)
    return mol_file


def test_add_bond():
    '''
    Test basic OBMol bond adding methods.
    '''
    # create an empty molecule
    mol = ob.OBMol()
    assert mol.NumAtoms() == 0, mol.NumAtoms()
    assert mol.NumBonds() == 0, mol.NumBonds()

    # add two atoms to the molecule
    a = mol.NewAtom()
    b = mol.NewAtom()
    assert mol.NumAtoms() == 2, mol.NumAtoms()
    assert mol.NumBonds() == 0, mol.NumBonds()

    # OB uses 1-based atom indexing
    assert a.GetIdx() == 1, a.GetIdx()
    assert b.GetIdx() == 2, b.GetIdx()
    assert mol.GetAtom(1) == a
    assert mol.GetAtom(2) == b

    # add a bond between the atoms
    assert mol.AddBond(1, 2, 1, 0)
    assert mol.GetBond(1, 2) and mol.GetBond(2, 1)
    assert mol.GetBond(a, b) and mol.GetBond(b, a)

    # try adding the same bond again
    assert not mol.AddBond(1, 2, 1, 0)
    assert not mol.AddBond(2, 1, 1, 0)

    # check that bond comparison holds
    assert compare_bonds(mol.GetBond(1, 2), mol.GetBond(2, 1))
    assert compare_bonds(mol.GetBond(a, b), mol.GetBond(b, a))


@pytest.fixture(params=[10, 50])
def dense(request):
    '''
    An OBMol where every pair of atoms
    is bonded with some probability.
    '''
    n_atoms = request.param

    p = 0.9
    bonds = np.random.choice(
        [0, 1], size=(n_atoms, n_atoms), p=[1-p, p]
    ) * (1 - np.eye(n_atoms))

    mol, atoms = mols.make_ob_mol(
        coords=np.random.normal(0, 10, (n_atoms, 3)),
        types=np.ones((n_atoms, 1)),
        bonds=bonds,
        typer=AtomTyper(
            prop_funcs=[Atom.atomic_num],
            prop_ranges=[[6]],
            radius_func=Atom.cov_radius,
            explicit_h=False,
            device='cpu'
        )
    )
    mols.write_ob_mols_to_sdf_file(
        'tests/output/TEST_dense_{}.sdf'.format(n_atoms), [mol]
    )
    return mol


def test_highly_fused_rings():
    ob_mol = mols.read_ob_mols_from_file('tests/input/buckyball.sdf', 'sdf')[0]
    mols.Molecule.from_ob_mol(ob_mol)


def test_reachable_basic():
    '''
    D--E
    | /
    |/\
    F  A--B
       | /
       |/
       C
    '''
    mol = ob.OBMol()

    def new_atom(x, y, z):
        # caution: i is the *id*, not the idx
        atom = mol.NewAtom(mol.NumAtoms())
        atom.SetAtomicNum(6)
        atom.SetVector(x, y, z)
        return atom

    def add_bond(a1, a2):
        mol.AddBond(a1.GetIdx(), a2.GetIdx(), 1, 0)

    a = new_atom(0, 0, 0)
    b = new_atom(1, 0, 0)
    with pytest.raises(AssertionError): # not bonded
        reachable(a, b) or reachable(b, a)

    add_bond(a, b)
    assert not reachable(a, b) and not reachable(b, a)

    c = new_atom(0, 2, 0)
    add_bond(b, c)
    assert not reachable(b, c) and not reachable(c, b)
    with pytest.raises(AssertionError):
        reachable(a, c) or reachable(c, a)

    add_bond(a, c) # cycle formed
    assert reachable(a, b) and reachable(b, a)
    assert reachable(b, c) and reachable(c, b)
    assert reachable(a, c) and reachable(c, a)

    d = new_atom(0, 0, 3)
    add_bond(a, d) # add unreachable group
    assert not reachable(a, d) and not reachable(d, a)

    e = new_atom(1, 0, 3)
    f = new_atom(0, 2, 3)
    add_bond(d, e)
    add_bond(d, f)
    add_bond(f, e) # form new cycle
    assert reachable(d, e) and reachable(e, d)
    assert reachable(d, f) and reachable(f, d)
    assert reachable(f, e) and reachable(e, f)

    #a.SetAtomicNum(8)
    #a.GetBond(d).SetBondOrder(2)
    #mols.write_ob_mols_to_sdf_file('tests/output/TEST_reachable.sdf', [mol])

    # atoms connecting the two cycles should be unreachable
    assert not reachable(a, d) and not reachable(d, a)


def test_reachable_recursion(dense):
    '''
    Test the recursive function that decides
    if two atoms are reachable without using
    the bond between them.
    '''
    f = lambda n: f(n-1) if n > 0 else True
    assert f(100)
    with pytest.raises(RecursionError):
        f(1000)

    # the first and last atoms should take a
    #   a pathologically long time to reach
    atom_a = dense.GetAtom(1)
    atom_b = dense.GetAtom(dense.NumAtoms())
    assert atom_a and atom_b, (atom_a, atom_b)
    assert dense.GetBond(atom_a, atom_b), 'not bonded'
    assert reachable(atom_a, atom_b), 'not reachable'


class TestBondAdding(object):

    @pytest.fixture(params=test_typer_fns)
    def typer(self, request):
        prop_funcs = request.param
        radius_func = lambda x: 1
        return AtomTyper.get_typer(prop_funcs, radius_func, device='cpu')

    @pytest.fixture
    def adder(self):
        return BondAdder(debug=True)

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

        for i, o in iter_atom_pairs(in_mol, out_mol, typer.explicit_h):
            assert o.GetAtomicNum() == i.GetAtomicNum(), 'different elements'
            assert o.GetVector() == i.GetVector(), 'different coordinates'

    def test_add_within_distance(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)
        out_mol, atoms = struct.to_ob_mol()
        adder.add_within_distance(out_mol, atoms, struct)

        for a_i, a_o in iter_atom_pairs(in_mol, out_mol, typer.explicit_h):
            for b_i, b_o in iter_atom_pairs(in_mol, out_mol, typer.explicit_h):

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

        for i, o in iter_atom_pairs(in_mol, out_mol, typer.explicit_h):
            # all H donors should have at least one hydrogen
            if typer.explicit_h:
                assert o.GetImplicitHCount() == 0, \
                    'explicit H donor has implicit H(s)'
            else:
                assert o.GetImplicitHCount() >= i.IsHbondDonor(), \
                    'implicit H donor has no implicit H(s)'

    def test_set_formal_charges(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)
        ob_mol, atoms = struct.to_ob_mol()
        adder.disable_perception(ob_mol)
        adder.set_formal_charges(ob_mol, atoms, struct)
        for i, o in iter_atom_pairs(in_mol, ob_mol, typer.explicit_h):
            assert o.GetFormalCharge() == i.GetFormalCharge(), \
                'incorrect formal charge'

    def test_remove_bad_valences(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)
        ob_mol, atoms = struct.to_ob_mol()
        adder.add_within_distance(ob_mol, atoms, struct)
        adder.remove_bad_valences(ob_mol, atoms, struct)
        max_vals = get_max_valences(atoms)
        for o in iter_atoms(ob_mol, typer.explicit_h):
            assert o.GetExplicitValence() <= max_vals.get(o.GetIdx(), 1), \
                'invalid valence'

    def test_remove_bad_geometry(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)
        ob_mol, atoms = struct.to_ob_mol()
        adder.add_within_distance(ob_mol, atoms, struct)
        adder.remove_bad_geometry(ob_mol)

    def test_set_aromaticity(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)
        ob_mol, atoms = struct.to_ob_mol()
        adder.disable_perception(ob_mol)
        adder.set_aromaticity(ob_mol, atoms, struct)
        for i, o in iter_atom_pairs(in_mol, ob_mol, typer.explicit_h):
            assert o.IsAromatic() == i.IsAromatic(), 'different aromaticity'

    def test_add_bonds(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)
        ob_mol, atoms = struct.to_ob_mol()
        ob_mol, visited_mols = adder.add_bonds(ob_mol, atoms, struct)
        add_struct = typer.make_struct(ob_mol)

        write_ob_pymol(visited_mols, in_mol)
        for t1, t2 in zip(struct.atom_types, add_struct.atom_types):
            print(t1, '\t', t2)

        # check bonds between atoms in typed structure
        for in_a, out_a in iter_atom_pairs(in_mol, ob_mol, typer.explicit_h):
            for in_b, out_b in iter_atom_pairs(in_mol, ob_mol, typer.explicit_h):

                in_bond = in_mol.GetBond(in_a, in_b)
                out_bond = ob_mol.GetBond(out_a, out_b)
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
        n_out = ob_mol.NumAtoms()
        assert n_out == n_in, 'different num atoms ({} vs {})'.format(
            n_out, n_in
        )

    def test_convert_mol(self, in_mol):
        rd_mol = Molecule.from_ob_mol(in_mol)
        out_mol = rd_mol.to_ob_mol()
        in_smi = mols.ob_mol_to_smi(in_mol)
        out_smi = mols.ob_mol_to_smi(out_mol)
        ob_sim = mols.get_ob_smi_similarity(out_smi, in_smi)
        assert out_smi == in_smi, \
            'different SMILES strings ({:.3f})'.format(ob_sim)

    def test_make_mol(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)
        out_mol, add_struct, visited_mols = adder.make_mol(struct)
        in_mol = Molecule.from_ob_mol(in_mol)

        n_atoms_diff = (in_mol.n_atoms - out_mol.n_atoms)
        elem_diff = (struct.elem_counts - add_struct.elem_counts).abs().sum()
        prop_diff = (struct.prop_counts - add_struct.prop_counts).abs().sum()

        assert n_atoms_diff == 0, \
            'different num atoms ({})'.format(n_atoms_diff)

        for t1, t2 in zip(struct.atom_types, add_struct.atom_types):
            print(t1, '\t', t2)

        assert elem_diff == 0, \
            'different element counts ({})'.format(elem_diff)

        assert prop_diff == 0, \
            'different property counts ({})'.format(prop_diff)

        out_valid, out_reason = out_mol.validate()
        assert out_valid, 'out_mol ' + out_reason

        in_valid, in_reason = in_mol.validate()
        assert in_valid, 'in_mol ' + in_reason

        print(len(visited_mols))
        write_rd_pymol(visited_mols, in_mol)

        in_smi = in_mol.to_smi()
        out_smi = out_mol.to_smi()
        rd_sim = mols.get_rd_mol_similarity(out_mol, in_mol, 'rdkit')
        ob_sim = mols.get_ob_smi_similarity(out_smi, in_smi)
        assert out_smi == in_smi, \
            'different SMILES strings ({:.3f} {:.3f})'.format(ob_sim, rd_sim)

        rmsd = out_mol.aligned_rmsd(in_mol)
        assert rmsd < 1.0, 'RMSD too high ({})'.format(rmsd)

    def test_uff_minimize(self, adder, typer, in_mol):
        struct = typer.make_struct(in_mol)
        out_mol, add_struct, visited_mols = adder.make_mol(struct)
        out_mol.validate()
        out_mol_min = out_mol.uff_minimize()
        in_mol = Molecule.from_ob_mol(in_mol)
        in_mol.validate()
        in_mol_min = in_mol.uff_minimize()
        write_rd_pymol(visited_mols + [out_mol_min, in_mol_min], in_mol)
