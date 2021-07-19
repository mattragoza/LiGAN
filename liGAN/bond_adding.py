import sys, os, pickle
from openbabel import openbabel as ob
from openbabel import pybel
from rdkit import Chem, Geometry
import numpy as np
from scipy.spatial.distance import pdist, squareform

from . import molecules as mols
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

    def disable_perception(self, ob_mol):
        '''
        Set flags that prevent openbabel perception
        of hybridization and aromaticity from being
        triggered when the properties are accessed.
        '''
        ob_mol.SetHybridizationPerceived(True)
        ob_mol.SetAromaticPerceived(True)

    def set_hybridization(self, ob_mol, atoms, struct):
        '''
        Set hybridization of atoms in two passes.
        First, perceive the hybridization state
        using openbabel, then turn off perception.
        Next, set all aromatic atoms to sp2.
        '''
        # turn on perception
        ob_mol.SetHybridizationPerceived(False)

        # trigger perception
        for ob_atom in atoms:
            ob_atom.GetHyb()

        # turn off perception
        ob_mol.SetHybridizationPerceived(True)

        # set all aromatic atoms to sp2
        if Atom.aromatic in struct.typer:
            for ob_atom, atom_type in zip(atoms, struct.atom_types):
                if atom_type.aromatic:
                    ob_atom.SetHyb(2)

    def set_aromaticity(self, ob_mol, atoms, struct):
        '''
        Use openbabel to perceive aromaticity, or
        set it based on atom types, if available.
        Set bonds as aromatic iff they are between
        aromatic atoms in a ring.
        '''
        if Atom.aromatic not in struct.typer:
            
            # turn on perception
            ob_mol.SetAromaticPerceived(False)

            # trigger perception
            for ob_atom in atoms:
                ob_atom.IsAromatic()

        else: # set aromaticity based on atom types
            for ob_atom, atom_type in zip(atoms, struct.atom_types):
                ob_atom.SetAromatic(bool(atom_type.aromatic))

        # turn off perception
        ob_mol.SetAromaticPerceived(True)

        # set bonds between aromatic ring atoms as aromatic
        for bond in ob.OBMolBondIter(ob_mol):
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if bond.IsInRing():
                bond.SetAromatic(a1.IsAromatic() and a2.IsAromatic())

    def set_formal_charges(self, ob_mol, atoms, struct):
        '''
        Set formal charge on atoms based on their
        atom type, if it is available.
        '''
        if Atom.formal_charge in struct.typer:
            for ob_atom, atom_type in zip(atoms, struct.atom_types):
                ob_atom.SetFormalCharge(atom_type.formal_charge)

    def set_min_h_counts(self, ob_mol, atoms, struct):
        '''
        Set atoms to have at least the minimum number
        of Hs required by their atom type. Does not
        remove Hs, and any added Hs are implicit.
        '''
        for ob_atom, atom_type in zip(atoms, struct.atom_types):

            if struct.typer.explicit_h:
                # all Hs should already be explicit,
                #   though possibly not bonded yet
                continue

            # get current hydrogen count
            h_count = Atom.h_count(ob_atom)

            # get count required by atom type
            if Atom.h_count in struct.typer:
                min_h_count = atom_type.h_count

            elif Atom.h_donor in struct.typer:
                min_h_count = 1 if atom_type.h_donor else 0

            if h_count < min_h_count:
                ob_atom.SetImplicitHCount(min_h_count - h_count)

    def add_within_distance(self, ob_mol, atoms, struct):
        '''
        Add bonds between every pair of atoms
        that are within a certain distance.
        '''
        # just do n^2 comparisons, worry about efficiency later
        coords = np.array([(a.GetX(), a.GetY(), a.GetZ()) for a in atoms])
        dists = squareform(pdist(coords))

        # for every pairs of atoms in ob_mol,
        for i, atom_a in enumerate(atoms):
            for j, atom_b in enumerate(atoms):
                if i >= j: # avoid redundant checks
                    continue

                # if they are within min and max bond length,
                if self.min_bond_len < dists[i,j] < self.max_bond_len:

                    # add a single bond between the atoms
                    ob_mol.AddBond(atom_a.GetIdx(), atom_b.GetIdx(), 1)

    def remove_bad_valences(self, ob_mol, atoms, struct):
        '''
        Remove hypervalent bonds without fragmenting
        the molecule, and prioritize stretched bonds.
        Also remove bonds between halogens/hydrogens.
        '''
        # get max valence of the atoms
        max_vals = get_max_valences(atoms)

        # remove any bonds between halogens or hydrogens
        for bond in ob.OBMolBondIter(ob_mol):
            atom_a = bond.GetBeginAtom()
            atom_b = bond.GetEndAtom()
            if (
                max_vals.get(atom_a.GetIdx(), 1) == 1 and
                max_vals.get(atom_b.GetIdx(), 1) == 1
            ):
                ob_mol.DeleteBond(bond)

        # remove bonds causing larger-than-permitted valences
        #   prioritize atoms with lowest max valence, since they
        #   place the hardest constraint on reachability (e.g O)

        atom_info = sort_atoms_by_valence(atoms, max_vals)
        for max_val, rem_val, atom in atom_info:

            if atom.GetExplicitValence() <= max_val:
                continue
            # else, the atom could have an invalid valence
            #   so check whether we can modify a bond

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

        # deleting bonds resets this flag
        ob_mol.SetHybridizationPerceived(True)

    def remove_bad_geometry(self, ob_mol):
        '''
        Remove bonds with excessive stretch or angle strain
        without fragmenting the molecule, and prioritizing
        the most stretch bonds.
        '''
        # eliminate geometrically poor bonds
        bond_info = sort_bonds_by_stretch(ob.OBMolBondIter(ob_mol))
        for bond_stretch, bond_len, bond in bond_info:

            # can we remove this bond without disconnecting the molecule?
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()

            # as long as we aren't disconnecting, let's remove things
            #   that are excessively far away (0.45 from ConnectTheDots)
            # get bonds to be less than max allowed
            # also remove tight angles, as done in openbabel
            if (bond_stretch > self.max_bond_stretch
                or forms_small_angle(atom1, atom2, self.min_bond_angle)
                or forms_small_angle(atom2, atom1, self.min_bond_angle)
            ):
                if reachable(atom1, atom2): # don't fragment the molecule
                    ob_mol.DeleteBond(bond)

        # deleting bonds resets this flag
        ob_mol.SetHybridizationPerceived(True)

    def fill_rem_valences(self, ob_mol, atoms, struct):
        '''
        Fill empty valences with hydrogens up to the
        amount expected by the atom type, or a typical
        amount according to openbabel, and then fill
        remaining empty valences with higher bond orders.
        '''
        max_vals = get_max_valences(atoms)

        for ob_atom, atom_type in zip(atoms, struct.atom_types):
          
            if struct.typer.explicit_h:
                # all Hs should already be present
                continue

            max_val = max_vals.get(ob_atom.GetIdx(), 1)

            if Atom.h_count in struct.typer:
                # this should have already been set
                #   by set_min_h_counts, but whatever
                h_count = Atom.h_count(ob_atom)
                if h_count < atom_type.h_count:
                    n = ob_atom.GetImplicitHCount()
                    ob_atom.SetImplicitHCount(n + atom_type.h_count - h_count)

            elif ob_atom.GetExplicitValence() < max_val:
                # this uses explicit valence and formal charge,
                #   and only ever INCREASES hydrogens, since it
                #   never sets implicit H to a negative value
                # but it does overwrite the existing value, so
                #   we need to save it beforehand and then add
                n = ob_atom.GetImplicitHCount()
                ob.OBAtomAssignTypicalImplicitHydrogens(ob_atom)
                n += ob_atom.GetImplicitHCount()
                ob_atom.SetImplicitHCount(n)

        # these have possibly changed
        max_vals = get_max_valences(atoms)

        # now increment bond orders to fill remaining valences
        atom_info = sort_atoms_by_valence(atoms, max_vals)
        for max_val, rem_val, atom in reversed(atom_info):

            if atom.GetExplicitValence() >= max_val:
                continue
            # else, the atom could have an empty valence
            #   so check whether we can augment a bond,
            #   prioritizing bonds that are too short

            bond_info = sort_bonds_by_stretch(ob.OBAtomBondIter(atom))
            for bond_stretch, bond_len, bond in reversed(bond_info):

                if bond.GetBondOrder() >= 3:
                    continue # don't go above triple

                # do the atoms involved in this bond have empty valences?
                #   since we are modifying the valences in the loop, this
                #   could have changed since calling sort_atoms_by_valence

                a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
                min_val_diff = min( # by how much are the valences under?
                    max_vals.get(a1.GetIdx(), 1) - a1.GetExplicitValence(),
                    max_vals.get(a2.GetIdx(), 1) - a2.GetExplicitValence()
                )
                if min_val_diff > 0: # increase bond order

                    bond_order = bond.GetBondOrder() # don't go above triple
                    bond.SetBondOrder(min(bond_order + min_val_diff, 3))

                    # if the current atom now has its preferred valence,
                    #   break and let other atoms choose next bonds to augment
                    if atom.GetExplicitValence() == max_vals[atom.GetIdx()]:
                        break

    def make_h_explicit(self, ob_mol, atoms):
        '''
        Make implicit hydrogens into
        explicit hydrogens and set
        their hybridization state.
        '''
        # hydrogens are not added if this flag is set
        ob_mol.SetHydrogensAdded(False)
        ob_mol.AddHydrogens()

        for a in ob.OBMolAtomIter(ob_mol):
            if a.GetAtomicNum() == 1:
                a.SetHyb(1)

        # AddHydrogens() resets some flags
        self.disable_perception(ob_mol)

    def add_bonds(self, ob_mol, atoms, struct):

        # track each step of bond adding
        visited_mols = []

        def visit_mol(mol, msg):
            mol = copy_ob_mol(mol)
            visited_mols.append(mol)
            if self.debug:
                bmap = {1:'-', 2:'=', 3:'≡'}
                print(len(visited_mols), msg)
                assert (
                    mol.HasHybridizationPerceived() and 
                    mol.HasAromaticPerceived()
                ), 'perception is on'
                return
                for a in ob.OBMolAtomIter(mol):
                    print('   ', (
                        a.GetAtomicNum(),
                        a.IsAromatic(),
                        a.GetHyb(),
                        a.GetImplicitHCount()
                    ), end=' ')
                    for b in ob.OBAtomBondIter(a):
                        print('({}{}{})'.format(
                            b.GetBeginAtomIdx(),
                            bmap[b.GetBondOrder()],
                            b.GetEndAtomIdx()
                        ), end=' ')
                    print()

        if len(atoms) == 0: # nothing to do
            return ob_mol, visited_mols

        # by default, openbabel tries to perceive
        #   aromaticity and hybridization when you
        #   first access those properties, but it
        #   can be disabled by setting flags
        # here, we will prefer to use atom type info
        #   and only enable perception when needed
        self.disable_perception(ob_mol)
        visit_mol(ob_mol, 'initial struct')

        # add all bonds between atom pairs within a distance range
        self.add_within_distance(ob_mol, atoms, struct)
        visit_mol(ob_mol, 'add_within_distance')

        # set minimum H counts to determine hyper valency
        #   but don't make them explicit yet to avoid issues
        #   with bond adding/removal (i.e. ignore bonds to H)
        self.set_min_h_counts(ob_mol, atoms, struct)
        visit_mol(ob_mol, 'set_min_h_counts')

        # set formal charge to correctly determine allowed valences
        self.set_formal_charges(ob_mol, atoms, struct)
        visit_mol(ob_mol, 'set_formal_charges')

        # remove bonds to atoms that are above their allowed valence
        #   with priority towards removing highly stretched bonds
        self.remove_bad_valences(ob_mol, atoms, struct)
        visit_mol(ob_mol, 'remove_bad_valences')

        # remove bonds with excessively distorted lengths/angles
        self.remove_bad_geometry(ob_mol)
        visit_mol(ob_mol, 'remove_bad_geometry')

        # need to make_h_explicit() before PerceiveBondOrders()
        #   bc it fills remaining EXPLICIT valence with bonds
        # need to set_hybridization() before make_h_explicit()
        #   so that it generates the correct H coordinates

        self.set_hybridization(ob_mol, atoms, struct)
        visit_mol(ob_mol, 'set_hybridization')

        self.set_aromaticity(ob_mol, atoms, struct)
        visit_mol(ob_mol, 'set_aromaticity')

        self.make_h_explicit(ob_mol, atoms)
        visit_mol(ob_mol, 'make_h_explicit')

        # hybridization and aromaticity are perceived in PBO()
        #   but the flags both are cleared at the end
        # so we have to disable perception again, and then
        #   re-apply previous methods
        ob_mol.PerceiveBondOrders()
        self.disable_perception(ob_mol)
        visit_mol(ob_mol, 'perceive_bond_orders')

        self.set_min_h_counts(ob_mol, atoms, struct) # maybe removed by PBO?
        self.set_hybridization(ob_mol, atoms, struct)
        self.set_aromaticity(ob_mol, atoms, struct)
        self.make_h_explicit(ob_mol, atoms)
        visit_mol(ob_mol, 'recover_from_pbo')

        # try to fix higher bond orders that cause bad valences
        #   if hybrid flag is not set, then can alter hybridization
        self.remove_bad_valences(ob_mol, atoms, struct)
        visit_mol(ob_mol, 'remove_bad_valences')

        # fill remaining valences with explicit Hs,
        #   up to the num expected by the atom types,
        #   and fill the rest with increased bond orders
        self.fill_rem_valences(ob_mol, atoms, struct)
        visit_mol(ob_mol, 'fill_rem_valences')

        self.make_h_explicit(ob_mol, atoms)
        visit_mol(ob_mol, 'make_h_explicit')

        # final cleanup for validity
        for a in ob.OBMolAtomIter(ob_mol):
            if not a.IsInRing():
                a.SetAromatic(False)

        return ob_mol, visited_mols   

    def post_process_rd_mol(self, rd_mol, struct=None):
        '''
        Convert OBMol to RDKit mol, fixing up issues.
        '''
        pt = Chem.GetPeriodicTable()
        # if double/triple bonds are connected to hypervalent atoms,
        #   decrement the order

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

    def make_mol(self, struct, visited=True):
        '''
        Create a Molecule from an AtomStruct with added
        bonds, trying to maintain the same atom types.
        '''
        # convert struct to ob_mol with minimal processing
        ob_mol, atoms = struct.to_ob_mol()

        # add bonds and hydrogens, maintaining atomic properties
        ob_mol, visited_mols = self.add_bonds(ob_mol, atoms, struct)

        # convert ob_mol to rd_mol with minimal processing
        add_mol = Molecule.from_ob_mol(ob_mol)

        # convert output mol back to struct, to see if types match
        add_struct = struct.typer.make_struct(ob_mol)

        if visited:
            visited_mols = [
                Molecule.from_ob_mol(m) for m in visited_mols
            ] + [add_mol]

            return add_mol, add_struct, visited_mols
        else:
            return add_mol, add_struct

    def make_batch(self, structs):

        add_mols, add_structs = [], []
        for struct in structs:
            add_mol, add_struct = self.make_mol(struct, visited=False)
            add_mols.append(add_mol)
            add_structs.append(add_struct)

        return add_mols, add_structs


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


def compare_bonds(b1, b2):
    '''
    Return whether two OB bonds have the
    the same begin and end atom indices,
    assuming they're in the same mol.
    '''
    return (
        b1.GetBeginAtomIdx() == b2.GetBeginAtomIdx() and
        b1.GetEndAtomIdx() == b2.GetEndAtomIdx()
    ) or (
        b1.GetBeginAtomIdx() == b2.GetEndAtomIdx() and
        b1.GetEndAtomIdx() == b2.GetBeginAtomIdx()
    )


def reachable_r(curr_atom, goal_atom, avoid_bond, visited_atoms):
    '''
    Recursive helper for determining whether
    goal_atom is reachable from curr_atom
    without using avoid_bond.
    '''
    for nbr_atom in ob.OBAtomAtomIter(curr_atom):
        curr_bond = curr_atom.GetBond(nbr_atom)
        nbr_atom_idx = nbr_atom.GetIdx()
        if not compare_bonds(curr_bond, avoid_bond) and nbr_atom_idx not in visited_atoms:
            visited_atoms.add(nbr_atom_idx)
            if nbr_atom == goal_atom:
                return True
            elif reachable_r(nbr_atom, goal_atom, avoid_bond, visited_atoms):
                return True
    return False


def reachable(atom_a, atom_b):
    '''
    Return whether atom_b is reachable from atom_a
    without using the bond between them, i.e. whether
    the bond can be removed without fragmenting the
    molecule (because the bond is part of a ring).
    '''
    assert atom_a.GetBond(atom_b), 'atoms must be bonded'

    if atom_a.GetExplicitDegree() == 1 or atom_b.GetExplicitDegree() == 1:
        return False # this is the _only_ bond for one atom

    # otherwise do recursive traversal
    return reachable_r(
        curr_atom=atom_a,
        goal_atom=atom_b,
        avoid_bond=atom_a.GetBond(atom_b),
        visited_atoms={atom_a.GetIdx()}
    )


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
