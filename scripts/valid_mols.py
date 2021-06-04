import sys, os
from rdkit import Chem
sys.path.append('.')
from liGAN.molecules import Molecule

_, data_file, data_root = sys.argv

print(data_file, file=sys.stderr)
f = open(data_file)

n_valid, n_mols = 0, 0
for line in f:
    mol_src = line.split(' ')[4]
    mol = Molecule.from_sdf(data_root + '/' + mol_src, sanitize=False)
    valid, reason = mol.validate()
    if valid:
        print(line, end='', file=sys.stdout)
        n_valid += 1
    else:
        print(mol_src, reason, file=sys.stderr)
    n_mols += 1

f.close()
print('n_valid =', n_valid, file=sys.stderr)
print('n_invalid =', n_mols-n_valid, file=sys.stderr)
print('n_mols =', n_mols, file=sys.stderr)
