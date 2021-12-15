import sys, os, gzip, traceback, time, tempfile, shlex
from subprocess import Popen, PIPE
from functools import lru_cache
from collections import Counter
import numpy as np
import scipy as sp

from openbabel import openbabel as ob
from openbabel import pybel

import rdkit
from rdkit import Chem, Geometry, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED, Crippen
from rdkit.Chem.Fingerprints import FingerprintMols
sys.path.append(Chem.RDConfig.RDContribDir)
from SA_Score import sascorer
from NP_Score import npscorer

from .common import catch_exception

try:
    GNINA_CMD = os.environ["GNINA_CMD"]
except KeyError:
    # Default path to gnina
    GNINA_CMD = '/net/pulsar/home/koes/dkoes/local/bin/gnina'

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
    def from_smi(cls, smi, sanitize=True):
        return cls(Chem.MolFromSmiles(smi, sanitize=sanitize))
    
    @classmethod
    def from_struct(cls, struct):
        return cls(
            make_rd_mol(struct.xyz, struct.c, struct.bonds, struct.channels)
        )

    @classmethod
    def from_ob_mol(cls, ob_mol):
        return cls(ob_mol_to_rd_mol(ob_mol), ob_mol=ob_mol)

    @classmethod
    def from_sdf(cls, sdf_file, sanitize=True, idx=0):
        mol = read_rd_mols_from_sdf_file(sdf_file, sanitize=sanitize)[idx]
        return cls(mol, src_file=sdf_file, src_idx=idx, **mol.GetPropsAsDict())

    @classmethod
    def all_from_sdf(cls, sdf_file, sanitize=True):
        mols = read_rd_mols_from_sdf_file(sdf_file, sanitize=sanitize)
        return [
            cls(mol, src_file=sdf_file, src_idx=idx, **mol.GetPropsAsDict())
                for idx, mol in enumerate(mols)
        ]

    @classmethod
    def from_pdb(cls, pdb_file, sanitize=True):
        mol = read_rd_mol_from_pdb_file(pdb_file, sanitize=sanitize)
        return cls(mol, src_file=pdb_file, **mol.GetPropsAsDict())

    def to_sdf(self, sdf_file, name='', kekulize=True):
        write_rd_mol_to_sdf_file(sdf_file, self, name, kekulize)

    def to_pdb(self, pdb_file, name=''):
        write_rd_mol_to_pdb_file(pdb_file, self, name)

    def to_smi(self):
        return get_smiles_string(self)

    def to_ob_mol(self):
        return rd_mol_to_ob_mol(self)

    @property
    def n_atoms(self):
        return self.GetNumAtoms()

    @property
    def n_bonds(self):
        return self.GetNumBonds()

    @property
    def n_hydros(self):
        return self.GetNumAtoms() - self.GetNumHeavyAtoms()

    @property
    def n_frags(self):
        return len(Chem.GetMolFrags(self))

    @property
    def atoms(self):
        return [self.GetAtomWithIdx(i) for i in range(self.n_atoms)]

    @property
    def bonds(self):
        return [self.GetBondWithIdx(i) for i in range(self.n_bonds)]

    @property
    def coords(self):
        return self.GetConformer().GetPositions()

    @property
    def center(self):
        # return heavy atom centroid
        not_h = [a.GetAtomicNum() != 1 for a in self.atoms]
        return self.coords[not_h].mean(axis=0)

    def translate(self, xyz):
        dx, dy, dz = xyz
        rd_conf = self.GetConformer(0)
        for i in range(rd_conf.GetNumAtoms()):
            x, y, z = rd_conf.GetAtomPosition(i)
            rd_conf.SetAtomPosition(i, (x+dx, y+dy, z+dz)) # must be float64

    def aligned_rmsd(self, mol):
        # aligns self to mol
        return get_rd_mol_rmsd(self, mol, align=True)

    def sanitize(self):
        return Chem.SanitizeMol(self)

    def add_hs(self):
        return type(self)(Chem.AddHs(self, addCoords=True))

    def remove_hs(self):
        return type(self)(Chem.RemoveHs(self, sanitize=False))

    def validate(self):
        if self.n_atoms == 0:
            return False, 'No atoms'
        if self.n_frags > 1:
            return False, 'Multiple fragments'
        try:
            self.sanitize()
            return True, 'Valid molecule'
        except Chem.AtomValenceException:
            return False, 'Invalid valence'
        except (Chem.AtomKekulizeException, Chem.KekulizeException):
            return False, 'Failed to kekulize'

    def get_pocket(self, *args, **kwargs):
        return get_rd_mol_pocket(self, *args, **kwargs)

    def uff_minimize(self, *args, **kwargs):
        '''
        Minimize molecular geometry using UFF.
        The minimization results are stored in
        the info attribute of the returned mol.
        '''
        t_start = time.time()
        min_mol, E_init, E_min, error = \
            uff_minimize_rd_mol(self, *args, **kwargs)
        rmsd = min_mol
        return Molecule(
            min_mol,
            E_init=E_init,
            E_min=E_min,
            min_rmsd=get_rd_mol_rmsd(self, min_mol, align=False),
            min_error=error,
            min_time=time.time() - t_start,
        )

    def gnina_minimize(self, *args, **kwargs):
        '''
        Minimize receptor-ligand pose with gnina.
        The minimization results are stored in 
        the info attribute of the returned mol.
        '''
        return gnina_minimize_rd_mol(self, *args, **kwargs)


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
        with gzip.open(sdf_file) as f:
            suppl = Chem.ForwardSDMolSupplier(f, sanitize=sanitize)
            return [Molecule(mol) for mol in suppl]
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
    use_gzip = (
        isinstance(sdf_file, str) and sdf_file.endswith('.gz')
    )
    if use_gzip:
        sdf_file = gzip.open(sdf_file, 'wt')
    writer = Chem.SDWriter(sdf_file)
    writer.SetKekulize(kekulize)
    for mol in mols:
        if name:
            mol.SetProp('_Name', name)
        writer.write(mol)
    writer.close()
    if use_gzip:
        sdf_file.close()


def write_rd_mol_to_pdb_file(pdb_file, mol, name=''):
    use_gzip = (
        isinstance(pdb_file, str) and pdb_file.endswith('.gz')
    )
    if use_gzip:
        pdb_file = gzip.open(pdb_file, 'wt')
    writer = Chem.PDBWriter(pdb_file)
    mol.SetProp('_Name', name)
    writer.write(mol)
    writer.close()
    if use_gzip:
        pdb_file.close()


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

    if bonds is not None:
        for i, atom_i in enumerate(atoms):
            for j, atom_j in enumerate(atoms):
                if bonds[i,j]:
                    ob_mol.AddBond(
                        atom_i.GetIdx(), atom_j.GetIdx(), 1, 0
                    )

    ob_mol.EndModify()
    return ob_mol, atoms


def copy_ob_mol(ob_mol):
    copy_mol = ob.OBMol(ob_mol)
    assert copy_mol.HasAromaticPerceived() == ob_mol.HasAromaticPerceived()
    assert copy_mol.HasHybridizationPerceived() == ob_mol.HasHybridizationPerceived()
    for a, b in zip(ob.OBMolAtomIter(ob_mol), ob.OBMolAtomIter(copy_mol)):
        assert a.GetImplicitHCount() == b.GetImplicitHCount()
    return copy_mol



def read_ob_mols_from_file(mol_file, in_format=None, n_mols=None, add_h=False):
    assert os.path.isfile(mol_file), mol_file + ' does not exist'
    if in_format is None:
        in_format = mol_file.split('.', 1)[1]
    ob_conv = ob.OBConversion()
    ob_conv.SetInFormat(in_format)
    ob_mol = ob.OBMol()
    not_at_end = ob_conv.ReadFile(ob_mol, mol_file)
    ob_mols = [ob_mol]
    while not_at_end and (n_mols is None or len(ob_mols) < n_mols):
        ob_mol = ob.OBMol()
        not_at_end = ob_conv.Read(ob_mol)
        if add_h:
            ob_mol.AddHydrogens()
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


def ob_mol_center(ob_mol):
    assert ob_mol.NumAtoms() > 0
    x, y, z, n = 0
    for a in ob.OBMolAtomIter(ob_mol):
        x += a.GetX()
        y += a.GetY()
        z += a.GetZ()
        n += 1
    return (x/n, y/n, z/n)


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


def ob_hyb_to_rd_hyb(ob_atom):
    '''
    Get the rdkit hybridization state
    for an openbabel atom. See below:

            OpenBabel   RDkit
    unk     0           0
    s       1           1
    sp      1           2
    sp2     2           3
    sp3     3           4
    sp3d    4           5
    sp3d2   5           6
    other   6+          7
    '''
    ob_hyb = ob_atom.GetHyb()
    rd_hybs = Chem.HybridizationType
    if 1 < ob_hyb < 6:
        return rd_hybs.values[ob_hyb+1]
    elif ob_hyb == 1: # s or sp
        if ob_atom.GetAtomicNum() > 4:
            return rd_hybs.SP
        else: # no p orbitals
            return rd_hybs.S
    elif ob_hyb == 0:
        return rd_hybs.UNSPECIFIED
    else:
        return rd_hybs.OTHER


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
        rd_atom.SetHybridization(ob_hyb_to_rd_hyb(ob_atom))

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

    Chem.GetSSSR(rd_mol) # initialize ring info
    rd_mol.UpdatePropertyCache(strict=False) # compute valence

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


def get_rd_mol_rmsd(rd_mol1, rd_mol2, align=False):

    # NOTE rd_mol RMSD is undefined b/tw diff molecules
    #   (even if they have the same type counts)

    # GetBestRMS(mol1, mol2) aligns mol1 to mol2
    #   if we don't want this, copy mol1 first
    if not align:
        rd_mol1 = Chem.RWMol(rd_mol1)

    try:
        return AllChem.GetBestRMS(rd_mol1, rd_mol2)
    except RuntimeError:
        #RuntimeError: No sub-structure match found 
        #  between the reference and probe mol
        return np.nan


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
def get_rd_mol_similarity(rd_mol1, rd_mol2, fingerprint='rdkit'):

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


def get_rd_mol_pocket(rec_mol, lig_mol, max_dist=8):
    '''
    Return a molecule containing only
    the residues from rec_mol that are
    within max_dist of lig_mol.
    '''
    # get distances b/tw rec and lig atoms
    lig_coords = lig_mol.coords
    rec_coords = rec_mol.coords
    dist = sp.spatial.distance.cdist(lig_coords, rec_coords)

    # indexes of atoms in rec_mol that are
    #   within max_dist of an atom in lig_mol
    pocket_atom_idxs = set(np.nonzero((dist < max_dist))[1])

    # determine pocket residues
    pocket_res_ids = set()
    for i in pocket_atom_idxs:
        atom = rec_mol.GetAtomWithIdx(int(i))
        res_id = get_rd_atom_res_id(atom)
        pocket_res_ids.add(res_id)

    # copy mol and delete atoms
    pkt_mol = Molecule(rec_mol, src_mol=rec_mol)
    for atom in list(pkt_mol.GetAtoms()):
        res_id = get_rd_atom_res_id(atom)
        if res_id not in pocket_res_ids:
            pkt_mol.RemoveAtom(atom.GetIdx())

    pkt_mol.sanitize()
    return pkt_mol


def get_rd_atom_res_id(rd_atom):
    '''
    Return an object that uniquely
    identifies the residue that the
    atom belongs to in a given PDB.
    '''
    res_info = rd_atom.GetPDBResidueInfo()
    return (
        res_info.GetChainId(),
        res_info.GetResidueNumber()
    )


def uff_minimize_rd_mol(lig_mol, rec_mol=None, n_iters=200, n_tries=2):
    '''
    Attempt to minimize rd_mol with UFF.
    If rec_mol is provided, minimize in
    the context of the fixed receptor.

    Returns (min_mol, E_init, E_final, error).
    '''
    lig_mol = Chem.RWMol(lig_mol)
    if lig_mol.GetNumAtoms() == 0:
        return lig_mol, np.nan, np.nan, 'No atoms'

    E_init = np.nan
    E_final = np.nan
    error = None

    # regenerate hydrogen coords with rdkit
    lig_mol = Chem.AddHs(
        Chem.RemoveHs(lig_mol, updateExplicitCount=True, sanitize=False),
        explicitOnly=True,
        addCoords=True
    )

    if rec_mol: # combine into complex
        uff_mol = Chem.CombineMols(rec_mol, lig_mol)
    else: # just use the ligand
        uff_mol = lig_mol

    try:
        Chem.SanitizeMol(uff_mol)
    except Chem.AtomValenceException:
        error = 'Invalid valence'
    except (Chem.AtomKekulizeException, Chem.KekulizeException):
        error = 'Failed to kekulize'

    if error:
        return lig_mol, E_init, E_final, error

    try:
        # initialize force field
        uff = AllChem.UFFGetMoleculeForceField(
            uff_mol, confId=0, ignoreInterfragInteractions=False
        )
        uff.Initialize()

        # get the initial energy
        E_init = uff.CalcEnergy()

    except Exception as e:
        if 'getNumImplicitHs' in str(e):
            return lig_mol, E_init, E_final, 'No implicit valence'
        if 'bad params pointer' in str(e):
            return lig_mol, E_init, E_final, 'Invalid atom type'
        print('UFF1 exception')
        write_rd_mol_to_sdf_file('badmol_uff1.sdf', uff_mol, kekulize=False)
        raise e

    if not error and n_tries * n_iters > 0:

        if rec_mol: # fix receptor atoms
            for i in range(rec_mol.GetNumAtoms()):
                uff.AddFixedPoint(i)
        try:
            # minimize with force field
            converged = False
            while n_tries > 0 and not converged:
                print('.', end='', flush=True)
                converged = not uff.Minimize(maxIts=n_iters)
                n_tries -= 1
            print(flush=True)

            # get the final energy
            E_final = uff.CalcEnergy()
            if not converged:
                error = 'Not converged'

        except RuntimeError as e:
            print('UFF2 exception')
            error = str(e)
            write_rd_mol_to_sdf_file(
                'badmol_uff2.sdf', uff_mol, kekulize=True
            )
            traceback.print_exc(file=sys.stdout)

        if rec_mol:
            # copy minimized coords back to ligand
            coords = uff_mol.GetConformer().GetPositions()
            lig_conf = lig_mol.GetConformer()
            for i, xyz in enumerate(coords[-lig_mol.GetNumAtoms():]):
                lig_conf.SetAtomPosition(i, xyz)

    return lig_mol, E_init, E_final, error


def gnina_minimize_rd_mol(lig_mol, rec_mol):
    '''
    Minimize lig_mol wrt rec_mol using gnina.
    gnina is run as a subprocess, which needs
    the mols to be present on disk. Check for
    src_file or out_file, else use temp_file.
    '''
    if lig_mol.n_atoms == 0:
        return Molecule(Chem.RWMol(lig_mol), error='No atoms')

    def get_temp_file():
        with tempfile.NamedTemporaryFile() as f:
            return f.name + '.sdf.gz'

    def get_mol_as_file(mol):

        if 'src_file' in mol.info:
            return mol.info['src_file']
        elif 'out_file' in mol.info:
            return mol.info['out_file']
        else:
            tmp_file = get_temp_file()
            mol.to_sdf(tmp_file, kekulize=False)
            return tmp_file

    rec_file = get_mol_as_file(rec_mol)
    lig_file = get_mol_as_file(lig_mol)
    out_file = get_temp_file()
    assert os.path.isfile(lig_file), 'lig file does not exist'

    cmd = f'{GNINA_CMD} --minimize -r {rec_file} -l {lig_file} ' \
        f'--autobox_ligand {lig_file} -o {out_file}'

    error = None
    last_stdout = ''
    proc = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
    for c in iter(lambda: proc.stdout.read(1), b''): 
        sys.stdout.buffer.write(c)
        last_stdout += c.decode()
        if c == b'*' or c == b'\n': # progress bar or new line
            sys.stdout.flush()
            if last_stdout.startswith('WARNING'):
                error = last_stdout
            last_stdout = ''

    stderr = proc.stderr.read().decode()
    for stderr_line in stderr.split('\n'):
        if stderr_line.startswith('CUDNN Error'):
            error = stderr_line            

    print('GNINA STDERR', file=sys.stderr)
    print(stderr, file=sys.stderr)
    print('END GNINA STDERR', file=sys.stderr)

    try: # get top-ranked pose according to gnina
        out_mol = Molecule.from_sdf(out_file, idx=0, sanitize=False)
    except IndexError:
        out_mol = Molecule(Chem.RWMol(lig_mol))
        if not error:
            error = stderr

    out_mol.info['error'] = error
    return out_mol
