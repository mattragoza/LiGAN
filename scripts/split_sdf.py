import sys, os
from collections import defaultdict

sys.path.append('..')
from liGAN import molecules as m


def split_sdf(sdf_file):
    '''
    Split an sdf file into several files
    that each contain one molecular pose.
    '''
    assert os.path.isfile(sdf_file), sdf_file + ' does not exist'
    print('Splitting', sdf_file)
    in_dir, in_base = os.path.split(sdf_file)
    mol_name = in_base.split('.', 1)[0]
    pose_count = defaultdict(int)
    for mol in m.read_rd_mols_from_sdf_file(sdf_file, sanitize=False):
        #mol_name = mol.GetProp('_Name')
        pose_index = pose_count[mol_name]
        out_base = '{}_{}.sdf.gz'.format(mol_name, pose_index)
        out_file = os.path.join(in_dir, out_base)
        m.write_rd_mol_to_sdf_file(out_file, mol, name=mol_name, kekulize=True)
        print('\tWriting', out_file)
        pose_count[mol_name] += 1


def find_and_split_sdf(sdf_file):
    '''
    Given the name of a single-pose sdf file,
    find and split the multi-pose sdf file.
    '''
    if os.path.isfile(sdf_file):
        print('Found', sdf_file)
        return
    # need to find and split multi-pose file
    in_prefix = sdf_file.split('.', 1)[0] # strip file extension
    in_prefix = in_prefix.rsplit('_', 1)[0] # strip pose index
    multi_sdf_file = in_prefix + '.sdf.gz'
    split_sdf(multi_sdf_file)
    assert os.path.isfile(sdf_file), sdf_file + ' was not created'


if __name__ == '__main__':
    _, data_file, data_root = sys.argv
    with open(data_file) as f:
        lines = f.readlines()
    n_lines = len(lines)
    for i, line in enumerate(lines):
        pct = 100*(i+1)/n_lines
        print(f'[{pct:.2f}%] ', end='')
        sdf_file = os.path.join(data_root, line.split()[4])
        find_and_split_sdf(sdf_file)
    print('[100.00%] Done')

