import sys, gzip, traceback
import numpy as np

from openbabel import openbabel as ob
from openbabel import pybel

import rdkit
from rdkit import Chem, Geometry, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED, Crippen
from rdkit.Chem.Fingerprints import FingerprintMols

from SA_Score import sascorer
from NP_Score import npscorer
nps_model = npscorer.readNPModel()


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


def catch_exc(func, exc=Exception, default=np.nan):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except exc as e:            
            return default
    return wrapper


get_rd_mol_weight = catch_exc(Descriptors.MolWt)
get_rd_mol_logP = catch_exc(Chem.Crippen.MolLogP)
get_rd_mol_QED = catch_exc(Chem.QED.default)
get_rd_mol_SAS = catch_exc(sascorer.calculateScore)
get_rd_mol_NPS = catch_exc(npscorer.scoreMol)
get_aligned_rmsd  = catch_exc(AllChem.GetBestRMS)


@catch_exc
def get_smiles_string(rd_mol):
    return Chem.MolToSmiles(rd_mol, canonical=True, isomericSmiles=False)


@catch_exc
def get_rd_mol_similarity(rd_mol1, rd_mol2, fingerprint):

    if fingerprint == 'morgan':
        fgp1 = AllChem.GetMorganFingerprintAsBitVect(rd_mol1, 2, 1024)
        fgp2 = AllChem.GetMorganFingerprintAsBitVect(rd_mol2, 2, 1024)

    elif fingerprint == 'rdkit':
        fgp1 = Chem.Fingerprints.FingerprintMols.FingerprintMol(rd_mol1)
        fgp2 = Chem.Fingerprints.FingerprintMols.FingerprintMol(rd_mol2)

    elif fingerprint == 'maccs':
        fgp1 = AllChem.GetMACCSKeysFingerprint(rd_mol1)
        fgp2 = AllChem.GetMACCSKeysFingerprint(rd_mol2)

    return DataStructs.TanimotoSimilarity(fgp1, fgp2)


@catch_exc
def get_ob_smi_similarity(smi1, smi2):
    fgp1 = pybel.readstring('smi', smi1).calcfp()
    fgp2 = pybel.readstring('smi', smi2).calcfp()
    return fgp1 | fgp2


def uff_minimize_rd_mol(rd_mol, max_iters=10000):
    if rd_mol.GetNumAtoms() == 0:
        return Chem.RWMol(rd_mol), np.nan, np.nan, None
    try:
        rd_mol_H = Chem.AddHs(rd_mol, addCoords=True)
        ff = AllChem.UFFGetMoleculeForceField(rd_mol_H, confId=0)
        ff.Initialize()
        E_init = ff.CalcEnergy()
        try:
            res = ff.Minimize(maxIts=max_iters)
            E_final = ff.CalcEnergy()
            rd_mol = Chem.RemoveHs(rd_mol_H, sanitize=False)
            if res == 0:
                e = None
            else:
                e = RuntimeError('minimization not converged')
            return rd_mol, E_init, E_final, e
        except RuntimeError as e:
            w = Chem.SDWriter('badmol.sdf')
            w.write(rd_mol)
            w.close()
            print("NumAtoms",rd_mol.GetNumAtoms())
            traceback.print_exc(file=sys.stdout)
            return Chem.RWMol(rd_mol), E_init, np.nan, e
    except Exception as e:
        print("UFF Exception")
        traceback.print_exc(file=sys.stdout)
        return Chem.RWMol(rd_mol), np.nan, np.nan, None


@catch_exc
def get_rd_mol_uff_energy(rd_mol): # TODO do we need to add H for true mol?
    rd_mol_H = Chem.AddHs(rd_mol, addCoords=True)
    ff = AllChem.UFFGetMoleculeForceField(rd_mol_H, confId=0)
    return ff.CalcEnergy()


def get_rd_mol_validity(rd_mol):
    n_frags = len(Chem.GetMolFrags(rd_mol))
    try:
        Chem.SanitizeMol(rd_mol)
        error = None
    except Exception as e:
        error = e
    valid = n_frags == 1 and error is None
    return n_frags, error, valid


@catch_exc
def get_gpu_usage(gpu_id=0):
    return getGPUs()[gpu_id].memoryUtil


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