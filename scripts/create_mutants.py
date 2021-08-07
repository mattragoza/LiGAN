import os
# requires pymol rotkit package for mutate command

data_root = os.environ['CROSSDOCK_ROOT']
rec_file = data_root + '/AROK_MYCTU_1_176_0/1zyu_A_rec.pdb'
rec_name = '1zyu_A_rec'

residues = [
    (11, 'PRO'),
    (132, 'LEU'),
    (120, 'LEU'),
    (119, 'LEU'),
    (118, 'PRO'),
    (49, 'PHE'),
    (45, 'ILE'),
    (57, 'PHE'),
    (34, 'ASP'),
    (58, 'ARG'),
    (61, 'GLU'),
    (79, 'GLY'),
    (80, 'GLY'),
    (81, 'GLY'),
    (136, 'ARG'),
]

# create a mapping from charged residues to
#   oppositely charged residues of similar size
charge_map = {
    'ARG': 'GLU',
    'HIS': 'ASP',
    'LYS': 'GLU',
    'ASP': 'LYS',
    'GLU': 'LYS',
}

# mutate each pocket residue individually
for res_idx, res_name in residues:

    # make a mutant with the residue set to alanine
    mut_name = f'{rec_name}_mut_{res_idx}_{res_name}_ALA'
    print(f'copy {mut_name}, {rec_name}')
    print(f'mutate {mut_name}, {res_idx}, ALA')
    print(f'save {mut_name}.pdb, {mut_name}')

    if res_name in charge_map:

        # make another mutant with the opposite charge
        opp_res = charge_map[res_name]
        mut_name = f'{rec_name}_mut_{res_idx}_{res_name}_{opp_res}'
        print(f'copy {mut_name}, {rec_name}')
        print(f'mutate {mut_name}, {res_idx}, {opp_res}')
        print(f'save {mut_name}.pdb, {mut_name}')

# create one final mutant with ALL charges flipped
mut_name = f'{rec_name}_mut_all_charges'
print(f'copy {mut_name}, {rec_name}')
for res_idx, res_name in residues:
    if res_name in charge_map:
        opp_res = charge_map[res_name]
        print(f'mutate {mut_name}, {res_idx}, {opp_res}')

print(f'save {mut_name}.pdb, {mut_name}')
