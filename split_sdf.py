import sys, os
from collections import defaultdict
from rdkit import Chem

_, in_file, out_dir = sys.argv

assert os.path.isfile(in_file)
assert os.path.isdir(out_dir)

print('Splitting ' + in_file)

pose_count = defaultdict(int)
for rd_mol in Chem.SDMolSupplier(in_file):
    lig_name = rd_mol.GetProp('_Name')
    pose_index = pose_count[lig_name]
    out_base = '{}_{}.sdf'.format(lig_name, pose_index)
    out_file = os.path.join(out_dir, out_base)
    Chem.SDWriter(out_file).write(rd_mol)
    print('Wrote ' + out_file)
    pose_count[lig_name] += 1
