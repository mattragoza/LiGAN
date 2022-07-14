import sys, os
import ligan

_, mol_file = sys.argv
mol = ligan.molecules.Molecule.from_sdf(mol_file, sanitize=False)
mol.sanitize()

