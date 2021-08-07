import pandas as pd

# read in crossdocked2020 dataset
data_file = 'data/it2_tt_0_train0.types'
data_root = '/net/pulsar/home/koes/paf46_shared/PocketomeGenCross_Output'
data_cols = [
    'low_rmsd',
    'true_aff',
    'xtal_rmsd',
    'rec_src',
    'lig_src',
    'vina_aff'
]
data = pd.read_csv(data_file, sep=' ', names=data_cols, index_col=False)
#data['rec_src'] = data['rec_src'].str.replace('_0.gninatypes', '.pdb')
#data['lig_src'] = data['lig_src'].str.replace('.gninatypes', '.sdf.gz')

# create columns for the pocket, receptor-ligand pair, and ligand name
data['pocket'] = data['rec_src'].map(lambda x: x.split('/')[0])
data['rec_lig'] = data['lig_src'].map(lambda x: x.split('_lig')[0])
data['lig_name'] = data['rec_lig'].map(lambda x: x.rsplit('_', 1)[-1])

# get the lowest RMSD pose of each receptor-ligand pair
#   and filter any that don't have a pose < 2 RMSD
min_data = data.loc[data.groupby('rec_lig')['xtal_rmsd'].idxmin()]
min_data = min_data[min_data['low_rmsd'].astype(bool)]

#min_data['valid'] = min_data['lig_src'].map(
#    lambda x: liGAN.molecules.Molecule.from_sdf(data_root + '/' + x, sanitize=False).validate()[0]
#)
#min_data = min_data[min_data['valid']]

# merge the two data frames on the pocket and ligand name
#   the result will have every possible mapping from receptor-
#   ligand pose to different receptor-ligand pose, such that
#   the receptors have the same pocket, the ligands are the
#   same molecule, and the second pose has the lowest RMSD
# this could be a 10-20x expansion of the dataset, plus it
#   has additional string columns- could be 50 GB...
merge_data = data.merge(min_data,on=['pocket','lig_name'],suffixes=['','_min'])

save_cols = [
    'low_rmsd_min', 'true_aff_min', 'xtal_rmsd_min',
    'rec_src', 'lig_src', 'rec_src_min', 'lig_src_min',
    'vina_aff_min', 'low_rmsd', 'true_aff', 'xtal_rmsd', 'vina_aff'
]
out_file = 'data/it2_tt_0_cond_train0.types'
merge_data[save_cols].to_csv(out_file, index=False, header=False, sep=' ', float_format='%.5f')
