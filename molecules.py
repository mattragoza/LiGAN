import gzip
import numpy as np
from rdkit import Chem, Geometry
from openbabel import openbabel as ob
from openbabel import pybel


def make_rd_mol(xyz, c, bonds, channels):

    rd_mol = Chem.RWMol()

    for c_ in c:
        atomic_num = channels[c_].atomic_num
        rd_atom = Chem.Atom(atomic_num)
        rd_mol.AddAtom(rd_atom)

    n_atoms = rd_mol.GetNumAtoms()
    rd_conf = Chem.Conformer(n_atoms)

    for i, (x, y, z) in enumerate(xyz):
        rd_conf.SetAtomPosition(i, (x, y, z))

    rd_mol.AddConformer(rd_conf)

    if np.any(bonds):
        n_bonds = 0
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                if bonds[i,j]:
                    rd_mol.AddBond(i, j, Chem.BondType.SINGLE)
                    n_bonds += 1

    return rd_mol


def read_rd_mols_from_sdf_file(sdf_file):
    if sdf_file.endswith('.gz'):
        f = gzip.open(sdf_file)
        suppl = Chem.ForwardSDMolSupplier(f)
    else:
        suppl = Chem.SDMolSupplier(sdf_file)
    return [mol for mol in suppl]


def write_rd_mols_to_sdf_file(sdf_file, mols, name=''):
    '''
    Write a list of rdkit molecules to a file
    or io stream in sdf format.
    '''
    writer = Chem.SDWriter(sdf_file)
    for mol in mols:
        if name:
            mol.SetProp('_Name', name)
        writer.write(mol)
    writer.close()


def make_ob_mol(xyz, c, bonds, channels):
    '''
    Return an OpenBabel molecule from an array of
    xyz atom positions, channel indices, a bond matrix,
    and a list of atom type channels.
    '''
    ob_mol = ob.OBMol()

    n_atoms = 0
    for (x, y, z), c_ in zip(xyz, c):
        atomic_num = channels[c_].atomic_num
        atom = ob_mol.NewAtom()
        atom.SetAtomicNum(atomic_num)
        atom.SetVector(x, y, z)
        n_atoms += 1

    if np.any(bonds):
        n_bonds = 0
        for i in range(n_atoms):
            atom_i = ob_mol.GetAtom(i)
            for j in range(i+1, n_atoms):
                atom_j = ob_mol.GetAtom(j)
                if bonds[i,j]:
                    bond = ob_mol.NewBond()
                    bond.Set(n_bonds, atom_i, atom_j, 1, 0)
                    n_bonds += 1
    return ob_mol


def write_ob_mols_to_sdf_file(sdf_file, mols):
    conv = ob.OBConversion()
    if sdf_file.endswith('.gz'):
        conv.SetOutFormat('sdf.gz')
    else:
        conv.SetOutFormat('sdf')
    for i, mol in enumerate(mols):
        if i == 0:
            conv.WriteFile(mol, sdf_file)
        else:
            conv.Write(mol)
    conv.CloseOutFile()


def rd_mol_to_ob_mol(rd_mol, confId=0):

    ob_mol = ob.OBMol()
    rd_conf = rd_mol.GetConformer(confId)

    for i, rd_atom in enumerate(rd_mol.GetAtoms()):
        atomic_num = rd_atom.GetAtomicNum()
        rd_coords = rd_conf.GetAtomPosition(i)
        x = rd_coords.x
        y = rd_coords.y
        z = rd_coords.z
        ob_atom = ob_mol.NewAtom()
        ob_atom.SetAtomicNum(atomic_num)
        ob_atom.SetAromatic(rd_atom.GetIsAromatic())
        ob_atom.SetVector(x, y, z)

    for k, rd_bond in enumerate(rd_mol.GetBonds()):
        i = rd_bond.GetBeginAtomIdx()+1
        j = rd_bond.GetEndAtomIdx()+1
        bond_type = rd_bond.GetBondType()
        bond_flags = 0
        if bond_type == Chem.BondType.AROMATIC:
            bond_order = 1
            bond_flags |= ob.OB_AROMATIC_BOND
        elif bond_type == Chem.BondType.SINGLE:
            bond_order = 1
        elif bond_type == Chem.BondType.DOUBLE:
            bond_order = 2
        elif bond_type == Chem.BondType.TRIPLE:
            bond_order = 3
        ob_mol.AddBond(i, j, bond_order, bond_flags)

    return ob_mol


def ob_mol_to_rd_mol(ob_mol):

    n_atoms = ob_mol.NumAtoms()
    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(n_atoms)

    for ob_atom in ob.OBMolAtomIter(ob_mol):
        rd_atom = Chem.Atom(ob_atom.GetAtomicNum())
        rd_atom.SetIsAromatic(ob_atom.IsAromatic())
        #TODO copy format charge
        i = rd_mol.AddAtom(rd_atom)
        ob_coords = ob_atom.GetVector()
        x = ob_coords.GetX()
        y = ob_coords.GetY()
        z = ob_coords.GetZ()
        rd_coords = Geometry.Point3D(x, y, z)
        rd_conf.SetAtomPosition(i, rd_coords)

    rd_mol.AddConformer(rd_conf)

    for ob_bond in ob.OBMolBondIter(ob_mol):
        i = ob_bond.GetBeginAtomIdx()-1
        j = ob_bond.GetEndAtomIdx()-1
        bond_order = ob_bond.GetBondOrder()
        if ob_bond.IsAromatic():
            bond_type = Chem.BondType.AROMATIC
        elif bond_order == 1:
            bond_type = Chem.BondType.SINGLE
        elif bond_order == 2:
            bond_type = Chem.BondType.DOUBLE
        elif bond_order == 3:
            bond_type = Chem.BondType.TRIPLE
        else:
            raise Exception('unknown bond order {}'.format(bond_order))
        rd_mol.AddBond(i, j, bond_type)

    return rd_mol
