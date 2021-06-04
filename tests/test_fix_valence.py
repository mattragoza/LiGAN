import sys, os, pytest
from numpy import isclose

sys.path.insert(0, '.')
from liGAN import molecules as mols
from liGAN.molecules import ob
from liGAN.bond_adding import get_max_valences


def write_aromatic(sdf_file, visited_mols):
    write_mols = [mols.copy_ob_mol(m) for m in visited_mols]

    for m in write_mols: # make sure we don't reassign
        m.SetAromaticPerceived(True)

    # show aromaticity as element and bond order
    for m in write_mols:
        for a in ob.OBMolAtomIter(m):
            a.SetAtomicNum(1 + 5*a.IsAromatic())
        for b in ob.OBMolBondIter(m):
            b.SetBondOrder(1 + b.IsAromatic())

    mols.write_ob_mols_to_sdf_file(sdf_file, write_mols)


test_sdf_files = [
    'data/bad_mols/bad_valence0.sdf.gz',
    'data/bad_mols/bad_valence1.sdf.gz',
    'data/bad_mols/bad_valence2.sdf.gz',
    'data/bad_mols/bad_valence3.sdf.gz',
    'data/bad_mols/bad_valence4.sdf.gz',
    'data/bad_mols/bad_valence5.sdf.gz',
    'data/bad_mols/bad_valence6.sdf.gz',
    'data/bad_mols/bad_valence7.sdf.gz',
    'data/bad_mols/bad_valence8.sdf.gz',
    'data/bad_mols/bad_valence9.sdf.gz',
]


@pytest.fixture(params=test_sdf_files)
def bad_mol_file(request):
    return request.param


def fix_valence(ob_mol, atom, val_diff):

    if atom.IsInRing():
        assert False, 'RING'

    elif atom.GetAtomicNum() == 7:
        assert False, 'NITROGEN'
        atom.SetFormalCharge(val_diff)

    else:
        assert False, 'OTHER'


def test_fix_valence(bad_mol_file):

    ob_mol = mols.read_ob_mols_from_file(bad_mol_file, 'sdf')[0]
    assert ob_mol.NumAtoms() > 0, 'no atoms'
    atoms = list(ob.OBMolAtomIter(ob_mol))
    visited_mols = [mols.copy_ob_mol(ob_mol)]

    max_vals = get_max_valences(atoms)
    for a in atoms:
        val_diff = a.GetExplicitValence() - max_vals.get(a.GetIdx(), 1)
        if val_diff > 0:
            fix_valence(ob_mol, a, val_diff)

    assert ob.OBKekulize(ob_mol), 'failed to kekulize'
    visited_mols.append(mols.copy_ob_mol(ob_mol))

    write_aromatic('tests/output/TEST_' + bad_mol_file, visited_mols)

    ob_mol = mols.Molecule.from_ob_mol(ob_mol)
    ob_mol.sanitize()