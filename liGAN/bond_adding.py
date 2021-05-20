import sys, os, pickle
from openbabel import openbabel as ob
from openbabel import pybel
from rdkit import Chem, Geometry
import numpy as np
from scipy.spatial.distance import pdist, squareform

from .atom_types import Atom
from .molecules import ob_mol_to_rd_mol, Molecule, copy_ob_mol


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
        debug=False,
    ):
        self.min_bond_len = min_bond_len
        self.max_bond_len = max_bond_len

        self.max_bond_stretch = max_bond_stretch
        self.min_bond_angle = min_bond_angle

        self.debug = debug

    def set_aromaticity(self, ob_mol, atoms, struct):
        '''
        Set aromaticiy of atoms based on their atom
        types. Aromatic atoms are also marked as
        having sp2 hybridization. Bonds are set as
        aromatic iff both atoms are aromatic.
        '''
        if Atom.aromatic not in struct.typer:
            return False

        # set this flag to ensure that openbabel doesn't
        # reassign aromaticity when accessing IsAromatic()
        # and also copies aromaticity when copying mol
        ob_mol.SetAromaticPerceived(True)

        for ob_atom, atom_type in zip(atoms, struct.atom_types):

            if atom_type.aromatic:
                ob_atom.SetAromatic(True)
                ob_atom.SetHyb(2)
            else:
                ob_atom.SetAromatic(False)

        for bond in ob.OBMolBondIter(ob_mol):
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            bond.SetAromatic(a1.IsAromatic() and a2.IsAromatic())

        return True

    def set_formal_charges(self, ob_mol, atoms, struct):
        '''
        Set formal charge on atoms based on their
        atom type, if it is available.
        '''
        if Atom.formal_charge not in struct.typer:
            return False

        for ob_atom, atom_type in zip(atoms, struct.atom_types):
            ob_atom.SetFormalCharge(atom_type.formal_charge)

        return True

    def set_min_h_counts(self, ob_mol, atoms, struct):
        '''
        Set atoms to have at least one H if they are
        hydrogen bond donors, or the exact number of
        Hs specified by their atom type, if it is
        available.
        '''
        # this ensures that explicit coords are created for the
        # implicit Hs we add here when AddHydrogens() is called
        ob_mol.SetHydrogensAdded(False)

        for ob_atom, atom_type in zip(atoms, struct.atom_types):

            if struct.typer.explicit_h:
                continue

            if 'h_degree' in atom_type._fields:
                ob_atom.SetImplicitHCount(atom_type.h_degree)

            elif 'h_donor' in atom_type._fields:
                if atom_type.h_donor and ob_atom.GetImplicitHCount() == 0:
                    ob_atom.SetImplicitHCount(1)

    def add_within_distance(self, ob_mol, atoms, struct):

        # just do n^2 comparisons, worry about efficiency later
        coords = np.array([(a.GetX(), a.GetY(), a.GetZ()) for a in atoms])
        dists = squareform(pdist(coords))

        # add bonds between every atom pair within a certain distance
        for i, atom_a in enumerate(atoms):
            for j, atom_b in enumerate(atoms):
                if i >= j: # avoid redundant checks
                    continue

                # if distance is between min and max bond length,
                if self.min_bond_len < dists[i,j] < self.max_bond_len:

                    # add single bond
                    ob_mol.AddBond(atom_a.GetIdx(), atom_b.GetIdx(), 1)

    def remove_bad_valences(self, ob_mol, atoms, struct):

        # get max valence of the atoms
        max_vals = get_max_valences(atoms)

        # remove any impossible bonds between halogens (mtr22- and hydrogens)
        for bond in ob.OBMolBondIter(ob_mol):
            atom_a = bond.GetBeginAtom()
            atom_b = bond.GetEndAtom()
            if (
                max_vals.get(atom_a.GetIdx(), 1) == 1 and
                max_vals.get(atom_b.GetIdx(), 1) == 1
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
            # so check whether we can modify a bond

            bond_info = sort_bonds_by_stretch(ob.OBAtomBondIter(atom))
            for bond_stretch, bond_len, bond in bond_info:

                # do the atoms involved in this bond have bad valences?
                #   since we are modifying the valences in the loop, this
                #   could have changed since calling sort_atoms_by_valence

                a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
                max_val_diff = max( # by how much are the valences over?
                    a1.GetExplicitValence() - max_vals.get(a1.GetIdx(), 1),
                    a2.GetExplicitValence() - max_vals.get(a2.GetIdx(), 1)
                )
                if max_val_diff > 0:

                    bond_order = bond.GetBondOrder()
                    if bond_order > max_val_diff: # decrease bond order
                        bond.SetBondOrder(bond_order - max_val_diff)

                    elif reachable(a1, a2): # don't fragment the molecule
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

    def fill_rem_valences(self, ob_mol, atoms, struct):
        '''
        Fill empty valences with hydrogens up to the
        typical amount allowed by their atom type, and
        remaining valences with higher bond orders.
        '''
        # if this flag is present, then new hydrogens are not added
        ob_mol.SetHydrogensAdded(False)

        # get max valence of the atoms
        max_vals = get_max_valences(atoms)

        for ob_atom, atom_type in zip(atoms, struct.atom_types):
            assert ob_atom.GetImplicitHCount() == 0
            
            if struct.typer.explicit_h:
                continue

            max_val = max_vals.get(ob_atom.GetIdx(), 1)

            if 'h_degree' in atom_type._fields:
                # this should have already been set by set_min_h_counts
                h_degree = ob_atom.GetTotalDegree() - ob_atom.GetHvyDegree()
                assert h_degree == atom_type.h_degree

            elif ob_atom.GetExplicitValence() < max_val:

                # this uses explicit valence and formal charge
                # and it only ever INCREASES hydrogens, since it
                # never sets implicit H to a negative value
                ob.OBAtomAssignTypicalImplicitHydrogens(ob_atom)

        max_vals = get_max_valences(atoms)

        atom_info = sort_atoms_by_valence(atoms, max_vals)
        for max_val, rem_val, atom in reversed(atom_info):

            if atom.GetExplicitValence() >= max_val:
                continue
            # else, the atom could have an empty valence
            # so check whether we can augment a bond,
            # prioritizing bonds that are too short

            bond_info = sort_bonds_by_stretch(ob.OBAtomBondIter(atom))
            for bond_stretch, bond_len, bond in reversed(bond_info):

                # do the atoms involved in this bond have empty valences?
                #   since we are modifying the valences in the loop, this
                #   could have changed since calling sort_atoms_by_valence

                a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
                min_val_diff = min( # by how much are the valences under?
                    max_vals.get(a1.GetIdx(), 1) - a1.GetExplicitValence(),
                    max_vals.get(a2.GetIdx(), 1) - a2.GetExplicitValence()
                )
                if min_val_diff > 0: # increase bond order

                    bond_order = bond.GetBondOrder()
                    bond.SetBondOrder(bond_order + min_val_diff)

                    # if the current atom now has its preferred valence,
                    # break and let other atoms choose next bonds to augment
                    if atom.GetExplicitValence() == max_vals[atom.GetIdx()]:
                        break

    def add_bonds(self, ob_mol, atoms, struct):

        # track each step of bond adding
        visited_mols = []

        def visit_mol(mol, msg):
            visited_mols.append(copy_ob_mol(mol))
            if self.debug:
                print(len(visited_mols), msg)

        visit_mol(ob_mol, 'initial struct')

        if len(atoms) == 0: # nothing to do
            return ob_mol, visited_mols

        ob_mol.BeginModify() # why do we even need this?

        # add all bonds between atom pairs within a distance range
        self.add_within_distance(ob_mol, atoms, struct)
        visit_mol(ob_mol, 'add_within_distance')

        # set minimum H counts to determine hyper valency
        #   but don't make them explicit yet to avoid issues
        #   with bond adding/removal (i.e. ignore H bonds)
        self.set_min_h_counts(ob_mol, atoms, struct)
        visit_mol(ob_mol, 'set_min_h_counts')

        # set formal charge to correctly determine allowed valences
        # remove bonds to atoms that are above their allowed valence
        #   with priority towards removing highly stretched bonds
        self.set_formal_charges(ob_mol, atoms, struct)
        self.remove_bad_valences(ob_mol, atoms, struct)
        visit_mol(ob_mol, 'remove_bad_valences')

        # remove bonds whose lengths/angles are excessively distorted
        self.remove_bad_geometry(ob_mol)
        visit_mol(ob_mol, 'remove_bad_geometry')

        # NOTE the next section is important, but not intuitive,
        #   and the order of operations deserves explanation:
        # need to AddHydrogens() before PerceiveBondOrders()
        #   bc it fills remaining EXPLICIT valence with bonds
        # need to EndModify() before PerceiveBondOrders()
        #   otherwise you get a segmentation fault
        # need to AddHydrogens() after EndModify()
        #   because EndModify() resets hydrogen coords

        # need to set_aromaticity() before AND after EndModify()
        #   otherwise aromatic atom types are missing

        # hybridization and aromaticity are perceived in PBO()
        #   but the flags are both are turned off at the end
        #   which causes perception to be triggered again when
        #   calls to GetHyb() or IsAromatic() are made

        self.set_aromaticity(ob_mol, atoms, struct)
        visit_mol(ob_mol, 'set_aromaticity')

        ob_mol.EndModify()
        ob_mol.AddHydrogens()
        ob_mol.PerceiveBondOrders()
        self.set_aromaticity(ob_mol, atoms, struct)
        visit_mol(ob_mol, 'perceive_bond_orders')

        # try to fix higher bond orders that cause bad valences
        #   if hybrid flag is not set, then can alter hybridization
        self.remove_bad_valences(ob_mol, atoms, struct)
        visit_mol(ob_mol, 'remove_bad_valences')

        # fill remaining valences with h bonds,
        #   up to typical num allowed by the atom types
        #   and the rest with increased bond orders
        self.fill_rem_valences(ob_mol, atoms, struct)
        ob_mol.AddHydrogens()
        visit_mol(ob_mol, 'fill_rem_valences')

        return ob_mol, visited_mols

    def post_process_rd_mol(self, rd_mol, struct=None):
        '''
        Convert OBMol to RDKit mol, fixing up issues.
        '''
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
        Create a Molecule from an AtomStruct with added
        bonds, trying to maintain the same atom types.
        '''
        ob_mol, atoms = struct.to_ob_mol()
        ob_mol, visited_mols = self.add_bonds(ob_mol, atoms, struct)
        add_mol = Molecule.from_ob_mol(ob_mol)
        add_struct = struct.typer.make_struct(add_mol.to_ob_mol())
        visited_mols = [
            Molecule.from_ob_mol(m) for m in visited_mols
        ] + [add_mol]
        return add_mol, add_struct, visited_mols


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


def sort_bonds_by_stretch(bonds, absolute=True):
    '''
    Return bonds sorted by their distance
    from the optimal covalent bond length,
    and their actual bond length, with the
    most stretched and longest bonds first.
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
        stretch = bond_len - ideal_bond_len
        if absolute:
            stretch = np.abs(stretch)
        bond_info.append((stretch, bond_len, bond))

    # sort bonds from most to least stretched
    bond_info.sort(reverse=True, key=lambda t: (t[0], t[1]))
    return bond_info


def count_nbrs_of_elem(atom, atomic_num):
    count = 0
    for nbr in ob.OBAtomAtomIter(atom):
        if nbr.GetAtomicNum() == atomic_num:
            count += 1
    return count


def get_max_valences(atoms):

    # determine max allowed valences
    pt = Chem.GetPeriodicTable()
    max_vals = {}
    for i, ob_atom in enumerate(atoms):

        # set max valance to the smallest allowed by either openbabel
        # or rdkit, since we want the molecule to be valid for both
        # (rdkit is usually lower, mtr22- specifically for N, 3 vs 4)

        # mtr22- since we are assessing validity with rdkit,
        # we should try to use the rdkit valence model here
        # which allows multiple valences for certain elements
        # refer to rdkit.Chem.Atom.calcExplicitValence

        atomic_num = ob_atom.GetAtomicNum()

        # get default valence of isoelectronic element
        iso_atomic_num = atomic_num - ob_atom.GetFormalCharge()
        max_val = pt.GetDefaultValence(iso_atomic_num)

        # check for common functional groups
        if atomic_num == 15: # phosphate
            if count_nbrs_of_elem(ob_atom, 8) >= 4:
                max_val = 5

        elif atomic_num == 16: # sulfone
            if count_nbrs_of_elem(ob_atom, 8) >= 2:
                max_val = 6

        max_val -= ob_atom.GetImplicitHCount()

        max_vals[ob_atom.GetIdx()] = max_val

    return max_vals


def sort_atoms_by_valence(atoms, max_vals):
    '''
    Return atoms sorted by their maximum
    allowed valence and remaining valence,
    with the most valence-constrained and
    hyper-valent atoms sorted first.
    '''
    atom_info = []
    for atom in atoms:
        max_val = max_vals[atom.GetIdx()]
        rem_val = max_val - atom.GetExplicitValence()
        atom_info.append((max_val, rem_val, atom))

    # sort atoms from least to most remaining valence
    atom_info.sort(key=lambda t: (t[0], t[1]))
    return atom_info
