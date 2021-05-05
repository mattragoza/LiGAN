import sys, os, pickle
from openbabel import openbabel as ob
from openbabel import pybel
from rdkit import Chem, Geometry
import numpy as np
from scipy.spatial.distance import pdist, squareform

from .atom_types import Atom
from .molecules import ob_mol_to_rd_mol, Molecule


class BondAdder(object):
    '''
    An algorithm for constructing a valid molecule
    from a structure of atomic coordinates and types.

    First, it converts the struture to OBAtoms and
    tries to maintain as many of the atomic proper-
    ties defined by the atom types as possible.

    Next, it add bonds to the atoms, using the atom
    properties and coordinates as constraints.
    '''
    def __init__(
        self,
        min_bond_len=0.01,
        max_bond_len=4.0,
        max_bond_stretch=0.45,
        min_bond_angle=45,
    ):
        self.min_bond_len = min_bond_len
        self.max_bond_len = max_bond_len

        self.max_bond_stretch = max_bond_stretch
        self.min_bond_angle = min_bond_angle

    def make_ob_mol(self, struct):
        '''
        Create an OBMol from AtomStruct that has the
        same elements and coordinates. No other atomic
        properties are set and bonds are not added.

        Also returns a list of the created atoms in
        the same order as struct, which is needed for
        other methods that add hydrogens and change
        the indexing of atoms in the molecule.
        '''
        ob_mol = ob.OBMol()
        ob_mol.BeginModify()

        atoms = []
        for coord, type_vec in zip(struct.coords, struct.types):         
            ob_atom = ob_mol.NewAtom()

            x, y, z = [float(c) for c in coord]
            ob_atom.SetVector(x, y, z)

            atom_type = struct.typer.get_atom_type(type_vec)
            ob_atom.SetAtomicNum(atom_type.atomic_num)
            atoms.append(ob_atom)

        #self.set_atom_properties(ob_mol, atoms, struct)
        ob_mol.EndModify()
        return ob_mol, atoms

    def set_aromaticity(self, ob_mol, atoms, struct):
        '''
        Set aromaticiy of atoms based on their atom
        types. Aromatic atoms are also marked as
        sp2 hybridization.
        '''
        for ob_atom, atom_type in zip(atoms, struct.atom_types):

            if 'aromatic' in atom_type._fields:
                if atom_type.aromatic:
                    ob_atom.SetAromatic(True)
                    ob_atom.SetHyb(2)
                else:
                    ob_atom.SetAromatic(False)

        if 'aromatic' in atom_type._fields:
            ob_mol.SetAromaticPerceived(True)

    def set_min_h_counts(self, ob_mol, atoms, struct):
        '''
        Set atoms to have at least one H if they are
        hydrogen bond donors, and the exact number
        of Hs specified by their atom type, if it is
        available.
        '''
        assert not ob_mol.HasHydrogensAdded()

        for ob_atom, atom_type in zip(atoms, struct.atom_types):

            if 'h_degree' in atom_type._fields:
                ob_atom.SetImplicitHCount(atom_type.h_degree)

            elif 'h_donor' in atom_type._fields:
                if atom_type.h_donor and ob_atom.GetImplicitHCount() == 0:
                    ob_atom.SetImplicitHCount(1)

    def connect_the_dots(self, ob_mol, atoms, struct, visited_mols):
        '''
        Custom implementation of ConnectTheDots. This is similar to
        OpenBabel's version, but is more willing to make long bonds 
        to keep the molecule connected.

        It also attempts to respect atom type information from struct.
        Note that atoms and struct need to correspond in their order.

        Assumes no hydrogens or existing bonds.
        '''
        if len(atoms) == 0:
            return

        ob_mol.BeginModify()

        # add all bonds between all atom pairs in a certain distance range
        self.add_within_distance(ob_mol, atoms, struct)
        visited_mols.append(ob.OBMol(ob_mol))

        # remove bonds to atoms that are above their allowed valence
        self.remove_bad_valences(ob_mol, atoms, struct)
        visited_mols.append(ob.OBMol(ob_mol))

        # remove bonds whose lengths or angles are excessively distorted
        self.remove_bad_geometry(ob_mol)
        visited_mols.append(ob.OBMol(ob_mol))

        ob_mol.EndModify() # mtr22- this causes a seg fault if omitted

    def add_within_distance(self, ob_mol, atoms, struct):

        # just do n^2 comparisons, worry about efficiency later
        coords = np.array([(a.GetX(), a.GetY(), a.GetZ()) for a in atoms])
        dists = squareform(pdist(coords))

        # add bonds between every atom pair within a certain distance
        for i, atom_a in enumerate(atoms):
            for j, atom_b in enumerate(atoms):
                if i >= j: # avoid redundant checks
                    continue

                # if distance is between min and max bond length
                if self.min_bond_len < dists[i,j] < self.max_bond_len:

                    # add bond, checking whether it should be aromatic
                    flag = 0
                    if (
                        'aromatic' in struct.atom_types[i]
                        and struct.atom_types[i].aromatic
                        and struct.atom_types[j].aromatic
                    ):
                        flag = ob.OB_AROMATIC_BOND

                    ob_mol.AddBond(
                        atom_a.GetIdx(), atom_b.GetIdx(), 1, flag
                    )

    def remove_bad_valences(self, ob_mol, atoms, struct):

        # get max valence of the atom types
        max_vals = get_max_valences(atoms, struct)

        # remove any impossible bonds between halogens (mtr22- and hydrogens)
        for bond in ob.OBMolBondIter(ob_mol):
            atom_a = bond.GetBeginAtom()
            atom_b = bond.GetEndAtom()
            if (
                max_vals[atom_a.GetIdx()] == 1 and
                max_vals[atom_b.GetIdx()] == 1
            ):
                ob_mol.DeleteBond(bond)

        # removing bonds causing larger-than-permitted valences
        # prioritize atoms with lowest max valence, since they tend
        # to introduce the most problems with reachability (e.g O)

        atom_info = sort_atoms_by_valence(atoms, max_vals)
        for max_val, rem_val, atom in atom_info:

            if atom.GetExplicitValence() <= max_val:
                continue
            # else, the atom could have an invalid valence
            # so check whether we can remove a bond

            bond_info = sort_bonds_by_stretch(ob.OBAtomBondIter(atom))
            for bond_stretch, bond_len, bond in bond_info:
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()

                # check whether valences are not permitted (this could
                # have changed since the call to sort_atoms_by_valence)
                if atom1.GetExplicitValence() > max_vals[atom1.GetIdx()] or \
                    atom2.GetExplicitValence() > max_vals[atom2.GetIdx()]:
            
                    if reachable(atom1, atom2): # don't fragment the molecule
                        ob_mol.DeleteBond(bond)

                    # if the current atom now has a permitted valence,
                    # break and let other atoms choose next bonds to remove
                    if atom.GetExplicitValence() <= max_vals[atom.GetIdx()]:
                        break

    def remove_bad_geometry(self, ob_mol):

        # eliminate geometrically poor bonds
        bond_info = sort_bonds_by_stretch(ob.OBMolBondIter(ob_mol))
        for bond_stretch, bond_len, bond in bond_info:

            # can we remove this bond without disconnecting the molecule?
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()

            # as long as we aren't disconnecting, let's remove things
            # that are excessively far away (0.45 from ConnectTheDots)
            # get bonds to be less than max allowed
            # also remove tight angles, as done in openbabel
            if (bond_stretch > self.max_bond_stretch
                or forms_small_angle(atom1, atom2, self.min_bond_angle)
                or forms_small_angle(atom2, atom1, self.min_bond_angle)
            ):
                if reachable(atom1, atom2): # don't fragment the molecule
                    ob_mol.DeleteBond(bond)

    def add_bonds(self, ob_mol, atoms, struct):

        visited_mols = [ob.OBMol(ob_mol)]

        if len(atoms) == 0:
            return ob_mol, visited_mols

        ob_mol.BeginModify()

        # add all bonds between atom pairs within a distance range
        self.add_within_distance(ob_mol, atoms, struct)
        visited_mols.append(ob.OBMol(ob_mol))

        # set minimum H counts to determine hyper valency
        #   but don't make them explicit yet to avoid issues
        #   with bond adding/removal (i.e. ignore H bonds)
        self.set_min_h_counts(ob_mol, atoms, struct)
        visited_mols.append(ob.OBMol(ob_mol))

        # remove bonds to atoms that are above their allowed valence
        #   with priority towards removing highly stretched bonds
        self.remove_bad_valences(ob_mol, atoms, struct)
        visited_mols.append(ob.OBMol(ob_mol))

        # remove bonds whose lengths/angles are excessively distorted
        self.remove_bad_geometry(ob_mol)
        visited_mols.append(ob.OBMol(ob_mol))

        # segfault if EndModify() not before PerceiveBondOrders()
        #   but it also resets H coords, so AddHydrogens() after
        #   and it clears most flags, except AromaticPerceived()
        ob_mol.EndModify()

        # need to make implicit Hs explicit before PerceiveBondOrders()
        #   since it fills in the remaining valence with multiple bonds
        # AND need to set aromatic before making implicit Hs explicit
        #   because it uses the hybridization to create H coords
        self.set_aromaticity(ob_mol, atoms, struct)
        visited_mols.append(ob.OBMol(ob_mol))
        ob_mol.AddHydrogens()

        # use geometry to fill empty valences with double/triple bonds
        ob_mol.PerceiveBondOrders()
        visited_mols.append(ob.OBMol(ob_mol))

        return ob_mol, visited_mols

    if False: # TODO integrate from here

        self.set_atom_properties(ob_mol, atoms, struct)
        visited_mols.append(ob.OBMol(ob_mol))

        for ob_atom in ob.OBMolAtomIter(ob_mol):
            ob.OBAtomAssignTypicalImplicitHydrogens(ob_atom)
        self.set_atom_properties(ob_mol, atoms, struct)
        visited_mols.append(ob.OBMol(ob_mol))

        ob_mol.AddHydrogens()
        self.set_atom_properties(ob_mol, atoms, struct)
        visited_mols.append(ob.OBMol(ob_mol))

        # make rings all aromatic if majority of carbons are aromatic
        for ring in ob.OBMolRingIter(ob_mol):
            if 5 <= ring.Size() <= 6:
                carbon_cnt = 0
                aromatic_c_cnt = 0
                for ai in ring._path:
                    a = ob_mol.GetAtom(ai)
                    if a.GetAtomicNum() == 6:
                        carbon_cnt += 1
                        if a.IsAromatic():
                            aromatic_c_cnt += 1
                if aromatic_c_cnt >= carbon_cnt/2 and aromatic_c_cnt != ring.Size():
                    #set all ring atoms to be aromatic
                    for ai in ring._path:
                        a = ob_mol.GetAtom(ai)
                        a.SetAromatic(True)

        # bonds must be marked aromatic for smiles to match
        for bond in ob.OBMolBondIter(ob_mol):
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if a1.IsAromatic() and a2.IsAromatic():
                bond.SetAromatic(True)

        visited_mols.append(ob.OBMol(ob_mol))
        return ob_mol, visited_mols

    def convert_ob_mol_to_rd_mol(self, ob_mol, struct=None):
        '''
        Convert OBMol to RDKit mol, fixing up issues.
        '''
        ob_mol.DeleteHydrogens() # mtr22- don't we want to keep these?

        n_atoms = ob_mol.NumAtoms()
        rd_mol = Chem.RWMol()
        rd_conf = Chem.Conformer(n_atoms)

        for ob_atom in ob.OBMolAtomIter(ob_mol):
            rd_atom = Chem.Atom(ob_atom.GetAtomicNum())
            #TODO copy formal charge
            if ob_atom.IsAromatic() and ob_atom.IsInRing() and ob_atom.MemberOfRingSize() <= 6:
                # don't commit to being aromatic unless rdkit will be okay
                # with the ring status
                # (this can happen if the atoms aren't fit well enough)
                rd_atom.SetIsAromatic(True)
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
            if bond_order == 1:
                rd_mol.AddBond(i, j, Chem.BondType.SINGLE)
            elif bond_order == 2:
                rd_mol.AddBond(i, j, Chem.BondType.DOUBLE)
            elif bond_order == 3:
                rd_mol.AddBond(i, j, Chem.BondType.TRIPLE)
            else:
                raise Exception('unknown bond order {}'.format(bond_order))

            if ob_bond.IsAromatic():
                bond = rd_mol.GetBondBetweenAtoms (i,j)
                bond.SetIsAromatic(True)

        rd_mol = Chem.RemoveHs(rd_mol, sanitize=False)
        #mtr22- didn't we just previously delete hydrogens in OB?

        pt = Chem.GetPeriodicTable()
        #if double/triple bonds are connected to hypervalent atoms, decrement the order

        positions = rd_mol.GetConformer().GetPositions()
        nonsingles = []
        for bond in rd_mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE or bond.GetBondType() == Chem.BondType.TRIPLE:
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                dist = np.linalg.norm(positions[i]-positions[j])
                nonsingles.append((dist,bond))
        nonsingles.sort(reverse=True, key=lambda t: t[0])

        for (d,bond) in nonsingles:
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()

            if calc_valence(a1) > pt.GetDefaultValence(a1.GetAtomicNum()) or \
               calc_valence(a2) > pt.GetDefaultValence(a2.GetAtomicNum()):
                btype = Chem.BondType.SINGLE
                if bond.GetBondType() == Chem.BondType.TRIPLE:
                    btype = Chem.BondType.DOUBLE
                bond.SetBondType(btype)

        for atom in rd_mol.GetAtoms():
            #set nitrogens with 4 neighbors to have a charge
            if atom.GetAtomicNum() == 7 and atom.GetDegree() == 4:
                atom.SetFormalCharge(1)

        rd_mol = Chem.AddHs(rd_mol,addCoords=True)

        positions = rd_mol.GetConformer().GetPositions()
        center = np.mean(positions[np.all(np.isfinite(positions),axis=1)],axis=0)
        for atom in rd_mol.GetAtoms():
            i = atom.GetIdx()
            pos = positions[i]
            if not np.all(np.isfinite(pos)):
                #hydrogens on C fragment get set to nan (shouldn't, but they do)
                rd_mol.GetConformer().SetAtomPosition(i,center)

        try:
            Chem.SanitizeMol(rd_mol,Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE)
        except: # mtr22 - don't assume mols will pass this
            pass
            # dkoes - but we want to make failures as rare as possible and should debug them
            m = pybel.Molecule(ob_mol)
            if not os.path.isdir('badmols'):
                os.mkdir('badmols')
            i = np.random.randint(1000000)
            outname = 'badmols/badmol%d.sdf'%i
            print("WRITING", outname, file=sys.stderr)
            m.write('sdf',outname,overwrite=True)
            if struct:
                pickle.dump(struct,open('badmols/badmol%d.pkl'%i,'wb'))

        #but at some point stop trying to enforce our aromaticity -
        #openbabel and rdkit have different aromaticity models so they
        #won't always agree.  Remove any aromatic bonds to non-aromatic atoms
        for bond in rd_mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if bond.GetIsAromatic():
                if not a1.GetIsAromatic() or not a2.GetIsAromatic():
                    bond.SetIsAromatic(False)
            elif a1.GetIsAromatic() and a2.GetIsAromatic():
                bond.SetIsAromatic(True)

        return rd_mol

    def make_mol(self, struct):
        '''
        Create a Molecule from an AtomStruct with added bonds,
        trying to maintain the same atom types.

        TODO- separation of concerns
            should move as much of the "fixing up/validifying"
            code into separate methods, completely separate
            from the initial conversion of struct on ob_mol
            and the later conversion of ob_mol to rd_mol

            how best to do this given the OB and RDkit have
            different aromaticity models/other functions?
        '''
        ob_mol, atoms = self.make_ob_mol(struct)
        ob_mol, visited_mols = self.add_bonds(ob_mol, atoms, struct)
        rd_mol = Molecule(self.convert_ob_mol_to_rd_mol(ob_mol))
        add_struct = struct.typer.make_struct(rd_mol.to_ob_mol())
        visited_mols = [
            Molecule(ob_mol_to_rd_mol(m)) for m in visited_mols
        ] + [rd_mol]
        return rd_mol, add_struct, visited_mols


def calc_valence(rd_atom):
    '''
    Can call GetExplicitValence before sanitize,
    but need to know this to fix up the molecule
    to prevent sanitization failures.
    '''
    val = 0
    for bond in rd_atom.GetBonds():
        val += bond.GetBondTypeAsDouble()
    return val


def reachable_r(atom_a, atom_b, visited_bonds):
    '''
    Recursive helper for determining whether
    atom_a is reachable from atom_b without
    using the bond between them.
    '''
    for nbr in ob.OBAtomAtomIter(atom_a):
        bond = atom_a.GetBond(nbr).GetIdx()
        if bond not in visited_bonds:
            visited_bonds.add(bond)
            if nbr == atom_b:
                return True
            elif reachable_r(nbr, atom_b, visited_bonds):
                return True
    return False


def reachable(atom_a, atom_b):
    '''
    Return true if atom b is reachable from a
    without using the bond between them.
    '''
    if atom_a.GetExplicitDegree() == 1 or atom_b.GetExplicitDegree() == 1:
        return False # this is the _only_ bond for one atom

    # otherwise do recursive traversal
    visited_bonds = set([atom_a.GetBond(atom_b).GetIdx()])
    return reachable_r(atom_a, atom_b, visited_bonds)


def forms_small_angle(atom_a, atom_b, cutoff=45):
    '''
    Return whether bond between atom_a and atom_b
    is part of a small angle with a neighbor of a
    only.
    '''
    for nbr in ob.OBAtomAtomIter(atom_a):
        if nbr != atom_b:
            degrees = atom_b.GetAngle(atom_a, nbr)
            if degrees < cutoff:
                return True
    return False


def sort_bonds_by_stretch(bonds):
    '''
    Return bonds sorted by their distance
    from the optimal covalent bond length.
    '''
    bond_info = []
    for bond in bonds:

        # compute how far away from optimal we are
        atomic_num1 = bond.GetBeginAtom().GetAtomicNum()
        atomic_num2 = bond.GetEndAtom().GetAtomicNum()
        ideal_bond_len = (
            ob.GetCovalentRad(atomic_num1) +
            ob.GetCovalentRad(atomic_num2)
        )
        bond_len = bond.GetLength()
        stretch = np.abs(bond_len - ideal_bond_len) # mtr22- take abs
        bond_info.append((stretch, bond_len, bond))

    # sort bonds from most to least stretched
    bond_info.sort(reverse=True, key=lambda t: (t[0], t[1]))
    return bond_info


def get_max_valences(atoms, struct):

    # determine max allowed valences
    max_vals = {}
    for i, ob_atom in enumerate(atoms):

        # set max valance to the smallest allowed by either openbabel
        # or rdkit, since we want the molecule to be valid for both
        # (rdkit is usually lower, mtr22- specifically for N, 3 vs 4)
        atomic_num = ob_atom.GetAtomicNum()
        max_val = min(
            ob.GetMaxBonds(atomic_num),
            Chem.GetPeriodicTable().GetDefaultValence(atomic_num)
        )
        atom_type = struct.typer.get_atom_type(struct.types[i])

        if Atom.formal_charge in struct.typer:
            max_val += atom_type.formal_charge #mtr22- is this correct?

        if Atom.h_degree in struct.typer:
            max_val -= atom_type.h_degree # leave room for hydrogen

        elif Atom.h_donor in struct.typer:
            if atom_type.h_donor:
                max_val -= 1  # leave room for hydrogen (mtr22- how many?)

        max_vals[ob_atom.GetIdx()] = max_val

    return max_vals


def sort_atoms_by_valence(atoms, max_vals):
    '''
    Return atoms sorted by their explicit 
    valence and difference from maximum
    allowed valence.
    '''
    atom_info = []
    for atom in atoms:
        max_val = max_vals[atom.GetIdx()]
        rem_val = max_val - atom.GetExplicitValence()
        atom_info.append((max_val, rem_val, atom))

        # mtr22- should we sort by rem_val first instead?
        # doesn't this mean that we will always choose to
        # remove a bond to a low max-valence atom, even if it
        # has remaining valence, and even if there are higher
        # max-valence atom that have less remaining valence?

    # sort atoms from least to most remaining valence
    atom_info.sort(key=lambda t: (t[0], t[1]))
    return atom_info
