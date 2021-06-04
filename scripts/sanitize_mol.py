import sys, os
import liGAN

_, mol_file = sys.argv
mol = liGAN.molecules.Molecule.from_sdf(mol_file, sanitize=False)
mol.sanitize()

