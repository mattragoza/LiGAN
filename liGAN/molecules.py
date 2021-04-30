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

    @property
    def n_atoms(self):
        return self.GetNumAtoms()

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


def make_rd_mol(xyz, c, bonds, channels):

    rd_mol = Chem.RWMol()

    for c_ in c:
        atomic_num = channels[c_].atomic_num
        rd_atom = Chem.Atom(atomic_num)
        rd_mol.AddAtom(rd_atom)

    n_atoms = rd_mol.GetNumAtoms()
    rd_conf = Chem.Conformer(n_atoms)

    for i, (x, y, z) in enumerate(xyz):
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


def rd_mol_to_ob_mol(rd_mol):

    ob_mol = ob.OBMol()
    rd_conf = rd_mol.GetConformer(0)

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


@catch_exception(exc_type=RuntimeError)
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


def set_rd_mol_aromatic(rd_mol, c, channels):

    # get aromatic carbon channels
    aroma_c_channels = set()
    for i, channel in enumerate(channels):
        if 'AromaticCarbon' in channel.name:
            aroma_c_channels.add(i)

    # get aromatic carbon atoms
    aroma_c_atoms = set()
    for i, c_ in enumerate(c):
        if c_ in aroma_c_channels:
            aroma_c_atoms.add(i)

    # make aromatic rings using channel info
    rings = Chem.GetSymmSSSR(rd_mol)
    for ring_atoms in rings:
        ring_atoms = set(ring_atoms)
        if len(ring_atoms & aroma_c_atoms) == 0: #TODO test < 3 instead, and handle heteroatoms
            continue
        if (len(ring_atoms) - 2)%4 != 0:
            continue
        for bond in rd_mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            if atom1.GetIdx() in ring_atoms and atom2.GetIdx() in ring_atoms:
                atom1.SetIsAromatic(True)
                atom2.SetIsAromatic(True)
                bond.SetBondType(Chem.BondType.AROMATIC)


def connect_rd_mol_frags(rd_mol):

    # try to connect fragments by adding min distance bonds
    frags = Chem.GetMolFrags(rd_mol)
    n_frags = len(frags)
    if n_frags > 1:

        nax = np.newaxis
        xyz = rd_mol.GetConformer(0).GetPositions()
        dist2 = ((xyz[nax,:,:] - xyz[:,nax,:])**2).sum(axis=2)

        pt = Chem.GetPeriodicTable()
        while n_frags > 1:

            frag_map = {ai: fi for fi, f in enumerate(frags) for ai in f}
            frag_idx = np.array([frag_map[i] for i in range(rd_mol.GetNumAtoms())])
            diff_frags = frag_idx[nax,:] != frag_idx[:,nax]

            can_bond = []
            for a in rd_mol.GetAtoms():
                n_bonds = sum(b.GetBondTypeAsDouble() for b in a.GetBonds())
                max_bonds = pt.GetDefaultValence(a.GetAtomicNum())
                can_bond.append(n_bonds < max_bonds)

            can_bond = np.array(can_bond)
            can_bond = can_bond[nax,:] & can_bond[:,nax]

            cond_dist2 = np.where(diff_frags & can_bond & (dist2<25), dist2, np.inf)

            if not np.any(np.isfinite(cond_dist2)):
                break # no possible bond meets the conditions

            a1, a2 = np.unravel_index(cond_dist2.argmin(), dist2.shape)
            rd_mol.AddBond(int(a1), int(a2), Chem.BondType.SINGLE)
            try:
                rd_mol.UpdatePropertyCache() # update explicit valences
            except:
                pass

            frags = Chem.GetMolFrags(rd_mol)
            n_frags = len(frags)
