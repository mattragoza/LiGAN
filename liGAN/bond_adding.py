from openbabel import openbabel as ob
from rdkit import Chem
import numpy as np
from scipy.spatial.distance import pdist, squareform

from .atom_types import Atom


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

    def set_atom_properties(self, ob_mol, atoms, struct):
        '''
        Set atom properties to match the atom types.
        Keep doing this to beat openbabel over the
        head with what we want to happen. (fixup)
        '''
        types = struct.types
        typer = struct.typer
        aromatic_perceived = False

        for ob_atom, type_vec in zip(atoms, types):

            atom_type = typer.get_atom_type(type_vec)

            if Atom.aromatic in typer:
                aromatic_perceived = True
                if atom_type.aromatic:
                    ob_atom.SetAromatic(True)
                    ob_atom.SetHyb(2)
                else:
                    ob_atom.SetAromatic(False)

            h_degree = None

            if Atom.h_degree in typer: # explicit number of h bonds
                h_degree = atom_type.h_degree

            elif Atom.h_donor in typer and Atom.h_acceptor in typer:

                if atom_type.h_donor: # at least one h bond, maybe more

                    # if there are no explicit h bonds,
                    if ob_atom.GetExplicitDegree() == ob_atom.GetHvyDegree():

                        # if it's nitrogen with one heavy atom bond,
                        if (
                            ob_atom.GetHvyDegree() == 1 and
                            ob_atom.GetAtomicNum() == 7
                        ):
                            h_degree = 2
                        else:
                            h_degree = 1

                elif atom_type.h_acceptor: # note the else, i.e. not a donor
                    h_degree = 0

            if h_degree is not None:
                ob_atom.SetImplicitHCount(h_degree)

        ob_mol.SetAromaticPerceived(aromatic_perceived)

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

        ob_mol.EndModify()

    def add_within_distance(self, ob_mol, atoms, struct):

        # just do n^2 comparisons, worry about efficiency later
        coords = np.array([(a.GetX(), a.GetY(), a.GetZ()) for a in atoms])
        dists = squareform(pdist(coords))
        atom_types = struct.get_atom_types()

        # add bonds between every atom pair within a certain distance
        for i, atom_a in enumerate(atoms):
            for j, atom_b in enumerate(atoms[i+1:]):

                # if distance is between min and max bond length
                if self.min_bond_len < dists[i,j] < self.max_bond_len:

                    # add bond, checking whether it should be aromatic
                    flag = 0
                    if (Atom.aromatic in struct.typer
                        and atom_types[i].aromatic
                        and atom_types[j].aromatic
                    ):
                        flag = ob.OB_AROMATIC_BOND

                    ob_mol.AddBond(
                        atom_a.GetIdx(), atom_b.GetIdx(), 1, flag
                    )

    def remove_bad_valences(self, ob_mol, atoms, struct):

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

            bond_info = sort_bonds_by_stretch(ob.OBAtomBondIter(atom))
            for bond_stretch, bond_len, bond in bond_info:

                # can we remove this bond without disconnecting the molecule?
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()

                # check whether valences are permitted
                if atom1.GetExplicitValence() > max_vals[atom1.GetIdx()] or \
                    atom2.GetExplicitValence() > max_vals[atom2.GetIdx()]:
            
                    if reachable(atom1, atom2): # don't fragment the molecule
                        ob_mol.DeleteBond(bond)

                    # if the current atom is under the permitted valence,
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

    def make_ob_mol(self, struct):
        '''
        Create an OBMol from AtomStruct that attempts
        to maintain the properties of the atom types.

        Also adds bonds to the molecule based on the
        distance, valence, and geometry of the atoms.
        '''
        ob_mol = ob.OBMol()
        ob_mol.BeginModify()
        visited_mols = []

        # need to maintain a list of original atoms since
        # this function can add hydrogens to the molecule
        atoms = []

        for coord, type_vec in zip(struct.coords, struct.types):         
            ob_atom = ob_mol.NewAtom()

            x, y, z = [float(c) for c in coord]
            ob_atom.SetVector(x, y, z)

            atom_type = struct.typer.get_atom_type(type_vec)
            ob_atom.SetAtomicNum(atom_type.atomic_num)
            atoms.append(ob_atom)

        self.set_atom_properties(ob_mol, atoms, struct)
        visited_mols.append(ob.OBMol(ob_mol))

        self.connect_the_dots(ob_mol, atoms, struct, visited_mols)
        self.set_atom_properties(ob_mol, atoms, struct)
        visited_mols.append(ob.OBMol(ob_mol))

        ob_mol.EndModify()

        ob_mol.AddPolarHydrogens() #make implicits explicit
        visited_mols.append(ob.OBMol(ob_mol))

        ob_mol.PerceiveBondOrders()
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

        # check if the atom types of the molecule are the same
        return ob_mol, struct.typer.make_struct(ob_mol), visited_mols


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
        # (rdkit is usually lower)
        atomic_num = ob_atom.GetAtomicNum()
        max_val = min(
            ob.GetMaxBonds(atomic_num),
            Chem.GetPeriodicTable().GetDefaultValence(atomic_num)
        )
        if Atom.h_donor in struct.typer:
            if struct.typer.get_atom_type(struct.types[i]).h_donor:
                max_val -= 1  # leave room for hydrogen (mtr22- how many?)

        max_vals[ob_atom.GetIdx()] = max_val

    return max_vals


def sort_atoms_by_valence(atoms, max_vals):
    '''
    Return atoms sorted by their difference
    from their maximum valence.
    '''
    atom_info = []
    for atom in atoms:
        max_val = max_vals[atom.GetIdx()]
        rem_val = max_val - atom.GetExplicitValence()
        atom_info.append((max_val, rem_val, atom)) # sort by rem_val first

    # sort atoms from least to most remaining valence
    atom_info.sort(key=lambda t: (t[0], t[1]))
    return atom_info
