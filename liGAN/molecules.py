import sys, gzip, traceback, time
from functools import lru_cache
from collections import Counter
import numpy as np

from openbabel import openbabel as ob
from openbabel import pybel

import rdkit
from rdkit import Chem, Geometry, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED, Crippen
from rdkit.Chem.Fingerprints import FingerprintMols

from SA_Score import sascorer
from NP_Score import npscorer

from .common import catch_exception


class Molecule(Chem.RWMol):
    '''
    A 3D molecular structure.

    This is a subclass of an RDKit molecule with methods
    for evaluating validity and minimizing with UFF.
    '''
    def __init__(self, rd_mol, **info):

        if hasattr(rd_mol, 'info'): # copy over
            new_info, info = info, rd_mol.info
            info.update(new_info)

        super().__init__(rd_mol)
        self.info = info
    
    @classmethod
    def from_struct(cls, struct):
        return cls(
            make_rd_mol(struct.xyz, struct.c, struct.bonds, struct.channels)
        )

    @classmethod
    def from_ob_mol(cls, ob_mol):
        return cls(ob_mol_to_rd_mol(ob_mol))

    @classmethod
    def from_sdf(cls, sdf_file, sanitize=True, idx=0):
        return cls(
            read_rd_mols_from_sdf_file(sdf_file, sanitize=sanitize)[idx]
        )

    @classmethod
    def from_pdb(cls, pdb_file, sanitize=True):
        return cls(
            read_rd_mol_from_pdb_file(pdb_file, sanitize=sanitize)
        )

    def to_sdf(self, sdf_file, name='', kekulize=True):
        write_rd_mol_to_sdf_file(sdf_file, self, name, kekulize)

    def to_smi(self):
        return get_smiles_string(self)

    def to_ob_mol(self):
        return rd_mol_to_ob_mol(self)

    @property
    def n_atoms(self):
        return self.GetNumAtoms()

    @property
    def n_hydros(self):
        return self.GetNumAtoms() - self.GetNumHeavyAtoms()

    @property
    def n_frags(self):
        return len(Chem.GetMolFrags(self))

    @property
    def center(self):
        # return heavy atom centroid
        mask = [a.GetAtomicNum() != 1 for a in self.GetAtoms()]
        return self.GetConformer(0).GetPositions()[mask].mean(axis=0)

    def translate(self, xyz):
        dx, dy, dz = xyz
        rd_conf = self.GetConformer(0)
        for i in range(rd_conf.GetNumAtoms()):
            x, y, z = rd_conf.GetAtomPosition(i)
            rd_conf.SetAtomPosition(i, (x+dx, y+dy, z+dz)) # must be float64

    def aligned_rmsd(self, mol):
        return get_rd_mol_rmsd(self, mol)

    def sanitize(self):
        return Chem.SanitizeMol(self)

    def uff_minimize(self):
        '''
        Minimize molecular geometry using UFF.
        The minimization results are stored in
        the info attribute of the returned mol.
        '''
        t_start = time.time()
        min_mol, E_init, E_min, error = uff_minimize_rd_mol(self)
        rmsd = min_mol
        return Molecule(
            min_mol,
            E_init=E_init,
            E_min=E_min,
            min_rmsd=get_rd_mol_rmsd(min_mol, self),
            min_error=error,
            min_time=time.time() - t_start,
        )


def make_rd_mol(coords, types, bonds, typer):
    '''
    Create an RWMol from numpy arrays of coords
    and types, optional bonds, and an AtomTyper.
    No atomic properties other than the elements
    and coordinates are set.
    '''
    rd_mol = Chem.RWMol()

    for type_vec in types:
        atom_type = typer.get_atom_type(type_vec)
        rd_atom = Chem.Atom(atom_type.atomic_num)
        rd_mol.AddAtom(rd_atom)

    n_atoms = rd_mol.GetNumAtoms()
    rd_conf = Chem.Conformer(n_atoms)

    for i, coord in enumerate(coords):
        x, y, z = [float(c) for c in coord]
        rd_conf.SetAtomPosition(i, (x, y, z)) # must be float64

    rd_mol.AddConformer(rd_conf)

    if bonds is not None and np.any(bonds):
        n_bonds = 0
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                if bonds[i,j]:
                    rd_mol.AddBond(i, j, Chem.BondType.SINGLE)
                    n_bonds += 1

    return rd_mol


def read_rd_mol_from_pdb_file(pdb_file, sanitize=True):
    return Chem.MolFromPDBFile(pdb_file, sanitize=sanitize)


def read_rd_mols_from_sdf_file(sdf_file, sanitize=True):
    if sdf_file.endswith('.gz'):
        f = gzip.open(sdf_file)
        suppl = Chem.ForwardSDMolSupplier(f, sanitize=sanitize)
    else:
        suppl = Chem.SDMolSupplier(sdf_file, sanitize=sanitize)
    return [Molecule(mol) for mol in suppl]


def write_rd_mol_to_sdf_file(sdf_file, mol, name='', kekulize=True):
    return write_rd_mols_to_sdf_file(sdf_file, [mol], name, kekulize)


def write_rd_mols_to_sdf_file(sdf_file, mols, name='', kekulize=True):
    '''
    Write a list of rdkit molecules to a file
    or io stream in sdf format.
    '''
    writer = Chem.SDWriter(sdf_file)
    writer.SetKekulize(kekulize)
    for mol in mols:
        if name:
            mol.SetProp('_Name', name)
        writer.write(mol)
    writer.close()


def make_ob_mol(coords, types, bonds, typer):
    '''
    Create an OBMol from numpy arrays of coords
    and types, optional bonds, and an AtomTyper.
    No atomic properties other than the elements
    and coordinates are set.

    Also returns a list of the created atoms in
    the same order as struct, which is needed for
    other methods that add hydrogens and change
    the indexing of atoms in the molecule.
    '''
    ob_mol = ob.OBMol()
    ob_mol.BeginModify()

    atoms = []
    for coord, type_vec in zip(coords, types):
        atom = ob_mol.NewAtom()

        x, y, z = [float(c) for c in coord]
        atom.SetVector(x, y, z)

        atom_type = typer.get_atom_type(type_vec)
        atom.SetAtomicNum(atom_type.atomic_num)
        atoms.append(atom)

    n_atoms = len(atoms)

    if bonds is not None and np.any(bonds):
        n_bonds = 0
        for i in range(n_atoms):
            atom_i = ob_mol.GetAtom(i)
            for j in range(i+1, n_atoms):
                atom_j = ob_mol.GetAtom(j)
                if bonds[i,j]:
                    bond = ob_mol.NewBond()
                    bond.Set(n_bonds, atom_i, atom_j, 1, 0)
                    n_bonds += 1

    ob_mol.EndModify()
    return ob_mol, atoms


def read_ob_mols_from_file(mol_file, in_format):
    ob_conv = ob.OBConversion()
    ob_conv.SetInFormat(in_format)
    ob_mol = ob.OBMol()
    not_at_end = ob_conv.ReadFile(ob_mol, mol_file)
    ob_mols = [ob_mol]
    while not_at_end:
        ob_mol = ob.OBMol()
        not_at_end = ob_conv.Read(ob_mol)
        ob_mols.append(ob_mol)
    return ob_mols


def set_ob_conv_opts(ob_conv, options):
    for o in options:
        ob_conv.AddOption(o, ob_conv.OUTOPTIONS)


def write_ob_mols_to_sdf_file(sdf_file, ob_mols, options='h'):
    ob_conv = ob.OBConversion()
    if sdf_file.endswith('.gz'):
        ob_conv.SetOutFormat('sdf.gz')
    else:
        ob_conv.SetOutFormat('sdf')
    set_ob_conv_opts(ob_conv, options)
    for i, ob_mol in enumerate(ob_mols):
        if i == 0:
            ob_conv.WriteFile(ob_mol, sdf_file)
        else:
            ob_conv.Write(ob_mol)
    ob_conv.CloseOutFile()


def ob_mol_count_elems(ob_mol):
    return Counter(a.GetAtomicNum() for a in ob.OBMolAtomIter(ob_mol))


def ob_mol_to_smi(ob_mol, options='cnh'):
    ob_conv = ob.OBConversion()
    ob_conv.SetOutFormat('smi')
    set_ob_conv_opts(ob_conv, options)
    return ob_conv.WriteString(ob_mol).rstrip()


def ob_mol_delete_bonds(ob_mol):
    ob_mol = ob.OBMol(ob_mol)
    for bond in ob.OBMolBondIter(ob_mol):
        ob_mol.DeleteBond(bond)
    return ob_mol


def ob_mol_to_rd_mol(ob_mol):
    '''
    Convert an OBMol to an RWMol, copying
    over the elements, coordinates, formal
    charges, bonds and aromaticity.
    '''
    n_atoms = ob_mol.NumAtoms()
    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(n_atoms)

    for ob_atom in ob.OBMolAtomIter(ob_mol):

        rd_atom = Chem.Atom(ob_atom.GetAtomicNum())
        rd_atom.SetFormalCharge(ob_atom.GetFormalCharge())
        rd_atom.SetIsAromatic(ob_atom.IsAromatic())
        rd_atom.SetNumExplicitHs(ob_atom.GetImplicitHCount())
        rd_atom.SetNoImplicit(True) # don't use rdkit valence model
        idx = rd_mol.AddAtom(rd_atom)

        rd_coords = Geometry.Point3D(
            ob_atom.GetX(), ob_atom.GetY(), ob_atom.GetZ()
        )
        rd_conf.SetAtomPosition(idx, rd_coords)

    rd_mol.AddConformer(rd_conf)

    for ob_bond in ob.OBMolBondIter(ob_mol):

        # OB uses 1-indexing, rdkit uses 0
        i = ob_bond.GetBeginAtomIdx() - 1
        j = ob_bond.GetEndAtomIdx() - 1

        bond_order = ob_bond.GetBondOrder()
        if bond_order == 1:
            bond_type = Chem.BondType.SINGLE
        elif bond_order == 2:
            bond_type = Chem.BondType.DOUBLE
        elif bond_order == 3:
            bond_type = Chem.BondType.TRIPLE
        else:
            raise Exception('unknown bond order {}'.format(bond_order))

        rd_mol.AddBond(i, j, bond_type)
        rd_bond = rd_mol.GetBondBetweenAtoms(i, j)
        rd_bond.SetIsAromatic(ob_bond.IsAromatic())

    return rd_mol


def rd_mol_to_ob_mol(rd_mol):
    '''
    Convert an RWMol to an OBMol, copying
    over the elements, coordinates, formal
    charges, bonds and aromaticity.
    '''
    ob_mol = ob.OBMol()
    ob_mol.BeginModify()
    rd_conf = rd_mol.GetConformer(0)

    for idx, rd_atom in enumerate(rd_mol.GetAtoms()):

        ob_atom = ob_mol.NewAtom()
        ob_atom.SetAtomicNum(rd_atom.GetAtomicNum())
        ob_atom.SetFormalCharge(rd_atom.GetFormalCharge())
        ob_atom.SetAromatic(rd_atom.GetIsAromatic())
        ob_atom.SetImplicitHCount(rd_atom.GetNumExplicitHs())

        rd_coords = rd_conf.GetAtomPosition(idx)
        ob_atom.SetVector(rd_coords.x, rd_coords.y, rd_coords.z)

    for rd_bond in rd_mol.GetBonds():

        # OB uses 1-indexing, rdkit uses 0
        i = rd_bond.GetBeginAtomIdx() + 1
        j = rd_bond.GetEndAtomIdx() + 1

        bond_type = rd_bond.GetBondType()
        if bond_type == Chem.BondType.SINGLE:
            bond_order = 1
        elif bond_type == Chem.BondType.DOUBLE:
            bond_order = 2
        elif bond_type == Chem.BondType.TRIPLE:
            bond_order = 3
        else:
            raise Exception('unknown bond type {}'.format(bond_type))

        ob_mol.AddBond(i, j, bond_order)
        ob_bond = ob_mol.GetBond(i, j)
        ob_bond.SetAromatic(rd_bond.GetIsAromatic())

    ob_mol.EndModify()
    return ob_mol


def get_rd_mol_validity(rd_mol):
    '''
    A molecule is considered valid iff it has at
    least one atom, all atoms are connected in a
    single fragment, and it raises no errors when
    passed to rdkit.Chem.SanitizeMol, indicating
    valid valences and successful kekulization.
    '''
    n_atoms = rd_mol.GetNumAtoms()
    n_frags = len(Chem.GetMolFrags(rd_mol))
    try:
        Chem.SanitizeMol(rd_mol)
        error = None
    except Chem.MolSanitizeException as e:
        error = str(e)
    valid = (n_atoms > 0 and n_frags == 1 and error is None)
    return n_atoms, n_frags, error, valid


# molecular weight and logP are defined for invalid molecules
# logP is log water-octanol partition coefficient
#   where hydrophilic molecules have logP < 0
get_rd_mol_weight = Descriptors.MolWt
get_rd_mol_logP = Chem.Crippen.MolLogP

# QED, SAS, and NPS are likely undefined for invalid molecules
# QED does not require calcImplicitHs, but SAS and NPS do
# SAS can raise RingInfo not initialized on invalid molecules
get_rd_mol_QED = catch_exception(
    Chem.QED.default, Chem.MolSanitizeException
)
get_rd_mol_SAS = catch_exception(
    sascorer.calculateScore, Chem.MolSanitizeException
)

# mol RMSD is undefined b/tw diff molecules (even if same type counts)
# also note that GetBestRMS(mol1, mol2) aligns mol1 to mol2
get_rd_mol_rmsd  = catch_exception(AllChem.GetBestRMS, RuntimeError)


@lru_cache(maxsize=1)
def get_NPS_model():
    '''
    Read NPS scoring model on first call,
    and cache it for subsequent calls.
    '''
    return npscorer.readNPModel()


def get_rd_mol_NPS(rd_mol):
    return npscorer.scoreMol(rd_mol, get_NPS_model())


def get_smiles_string(rd_mol):
    return Chem.MolToSmiles(rd_mol, canonical=True, isomericSmiles=False)


#@catch_exception(exc_type=RuntimeError)
def get_rd_mol_similarity(rd_mol1, rd_mol2, fingerprint):

    if fingerprint == 'morgan':
        # this can raise RingInfo not initialized even when valid??
        fgp1 = AllChem.GetMorganFingerprintAsBitVect(rd_mol1, 2, 1024)
        fgp2 = AllChem.GetMorganFingerprintAsBitVect(rd_mol2, 2, 1024)

    elif fingerprint == 'rdkit':
        fgp1 = Chem.Fingerprints.FingerprintMols.FingerprintMol(rd_mol1)
        fgp2 = Chem.Fingerprints.FingerprintMols.FingerprintMol(rd_mol2)

    elif fingerprint == 'maccs':
        fgp1 = AllChem.GetMACCSKeysFingerprint(rd_mol1)
        fgp2 = AllChem.GetMACCSKeysFingerprint(rd_mol2)

    return DataStructs.TanimotoSimilarity(fgp1, fgp2)


@catch_exception(exc_type=SyntaxError)
def get_ob_smi_similarity(smi1, smi2):
    fgp1 = pybel.readstring('smi', smi1).calcfp()
    fgp2 = pybel.readstring('smi', smi2).calcfp()
    return fgp1 | fgp2


def uff_minimize_rd_mol(rd_mol, max_iters=10000):
    '''
    Attempt to minimize rd_mol with UFF.
    Returns min_mol, E_init, E_final, error
    '''
    E_init = E_final = np.nan

    if rd_mol.GetNumAtoms() == 0:
        return rd_mol, E_init, E_final, 'No atoms'

    try: # initialize molecule and force field
        rd_mol = Chem.AddHs(rd_mol, addCoords=True)
        uff = AllChem.UFFGetMoleculeForceField(rd_mol, confId=0)
        uff.Initialize()
        E_init = uff.CalcEnergy()

    except Chem.rdchem.AtomValenceException:
        return rd_mol, E_init, E_final, 'Invalid valence'

    except Chem.rdchem.KekulizeException:
        return rd_mol, E_init, E_final, 'Failed to kekulize'

    except Exception as e:
        if 'getNumImplicitHs' in str(e):
            return rd_mol, E_init, E_final, 'No implicit valence'
        if 'bad params pointer' in str(e):
            return rd_mol, E_init, E_final, 'Invalid atom type'
        print('UFF1 exception')
        write_rd_mol_to_sdf_file('badmol_uff1.sdf', rd_mol, kekulize=False)
        raise e

    try: # minimize molecule with force field
        result = uff.Minimize(maxIts=max_iters)
        E_final = uff.CalcEnergy()
        return (
            rd_mol, E_init, E_final, 'Not converged' if result else None
        )

    except RuntimeError as e:
        print('UFF2 exception')
        write_rd_mol_to_sdf_file(
            'badmol_uff2.sdf', rd_mol, kekulize=True
        )
        traceback.print_exc(file=sys.stdout)
        return rd_mol, E_init, np.nan, str(e)


@catch_exception(exc_type=SyntaxError)
def get_rd_mol_uff_energy(rd_mol): # TODO do we need to add H for true mol?
    rd_mol = Chem.AddHs(rd_mol, addCoords=True)
    uff = AllChem.UFFGetMoleculeForceField(rd_mol, confId=0)
    return uff.CalcEnergy()
