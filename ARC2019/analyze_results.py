import sys, os, re, glob
import numpy as np
import pandas as pd

sys.path.append('.')
from generate import read_gninatypes_file
from atom_types import get_default_lig_channels
import results

import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster', font_scale=1.25)
sns.color_palette('bright')


def uses_covalent_radius(model_name):
	print(model_name)
	try:
		data_model_file = glob.glob(model_name + '/data*.model')[0]
	except IndexError:
		return False
	with open(data_model_file) as f:
		buf = f.read()
	return 'use_covalent_radius: true' in buf


def read_out_files(out_files, model_type):
	dfs = []
	for out_file in out_files:
		try:
			df = pd.read_csv(out_file, sep=' ')
			model_name = re.match(r'(.+)\.lowrmsd\.0\.all\.training_output', os.path.basename(out_file)).group(1)
			if uses_covalent_radius(model_name):
				continue
			df['model_name'] = model_name
			df['Model type'] = model_type
			df.rename(columns=dict(
				test_loss='Gen. L2 loss',
				test_y_loss='Gen. L2 loss',
				test_aff_loss='K-L divergence',
				gen_L2_loss='Gen. L2 loss',
				gen_adv_log_loss='Gen. GAN loss',
				gen_kldiv_loss='K-L divergence',
				disc_log_loss='Disc. GAN loss',
				iteration='Iteration'
			), inplace=True)
			dfs.append(df)
		except (pd.io.common.EmptyDataError, IOError) as e:
			print(out_file, e)
			continue
	return dfs

#################################### CORRELATION MATRIX ####################################

if False:
	def read_structs(data_file, data_root):
		channels = get_default_lig_channels(use_covalent_radius=False)
		lig_xyzs = dict()
		lig_cs = dict()
		with open(data_file, 'r') as f:
			for line in f:
				lig_file = os.path.join(data_root, line.split()[3])
				lig_name = os.path.splitext(os.path.basename(lig_file))[0]
				lig_xyz, lig_c = read_gninatypes_file(lig_file, channels)
				lig_xyzs[lig_name] = lig_xyz
				lig_cs[lig_name] = lig_c
		return lig_xyzs, lig_cs

	data_file = 'data/lowrmsd.types'
	data_root = '/net/pulsar/home/koes/dkoes/PDBbind/refined-set'
	lig_xyzs, lig_cs = read_structs(data_file, data_root)
	lig_centers = {l: np.mean(xyz, axis=0) for l, xyz in lig_xyzs.items()}

	metric_df = pd.read_csv('test_gen_refactor/post_sample.gen_metrics', sep=' ')
	metric_df = metric_df.groupby('lig_name').mean()

	metric_df['lig_gen_dist/lig_norm'] = metric_df['lig_gen_dist'] / metric_df['lig_norm']

	metric_df['n_atoms'] = metric_df.index.map(lambda x: len(lig_cs[x]))
	metric_df['n_atom_types'] = metric_df.index.map(lambda x: len(set(lig_cs[x])))
	metric_df['max_atom_dist'] = metric_df.index.map(lambda x: max(np.linalg.norm(lig_xyzs[x] - lig_centers[x], axis=1)))

	# 10 random samples that fit on grid
	#print(metric_df[metric_df['max_atom_dist'] < 11.5/2].sample(10))

	metric_df['model_name'] = ' ' #'adam0_2_2_b_0.0_vl-le13_24_0.5_3_2l_32_2_1024_e_d11_24_3_1l_16_2_x'

	metric_df.rename(columns=dict(
		n_atoms='# atoms',
		max_atom_dist='Ligand radius',
		lig_norm='||True density||',
		lig_gen_norm='||Gen. density||',
		lig_gen_dist='Gen. L2 dist.',
		lig_gen_fit_dist='Fit L2 dist.',
		lig_gen_RMSD='Fit RMSD',
		lig_gen_mean_dist='Gen. diversity'
	), inplace=True)

	x = ['Gen. L2 dist.', 'Fit L2 dist.', 'Fit RMSD']
	results.plot_dist('ARC2019/metric_dist.png', metric_df, x=x, hue=None, n_cols=len(x), height=8, width=6)

	x = ['# atoms', 'Ligand radius', '||True density||', '||Gen. density||', 'Gen. L2 dist.', 'Fit L2 dist.', 'Fit RMSD']
	results.plot_corr('ARC2019/metric_corr.png', metric_df, x=x, y=x, height=4, width=4, despine=False)

	print('finished corr plot')

	#################################### TRAINING PLOTS ####################################

	ce_models = [
		'ce13_24_0.5_3_2_32_2_1024_e',
		'vce13_24_0.5_3_2_32_2_1024_e',
	]

	base_pattern = '/*.lowrmsd.0.all.training_output'
	ae_l2_files = glob.glob('ae13_24_0.5_3_2*_*_1024_e/' + base_pattern)
	ae_gan_files = glob.glob('adam0_*_vl-le13_24_0.5_3_2*_*_1024__d11_24_3_1l_16_2_x/' + base_pattern)
	ae_l2gan_files = glob.glob('adam0_*_vl-le13_24_0.5_3_2*_*_1024_e_d11_24_3_1l_16_2_x/' + base_pattern)

	ae_l2_dfs    = read_out_files(ae_l2_files, 'AE L2')
	ae_gan_dfs   = read_out_files(ae_gan_files, 'AE GAN')
	ae_l2gan_dfs = read_out_files(ae_l2gan_files, 'AE L2+GAN')

	print('{} AE L2 models'.format(len(ae_l2_dfs)))
	print('{} AE GAN models'.format(len(ae_gan_dfs)))
	print('{} AE L2+GAN models'.format(len(ae_l2gan_dfs)))
	df = pd.concat(ae_l2_dfs + ae_gan_dfs + ae_l2gan_dfs)

	df = df[df['Iteration'] <= 50000]

	avg_iters = 250
	df['Iteration'] = avg_iters * (df['Iteration']//avg_iters)

	df = df.set_index(['model_name', 'Iteration'])
	params = results.add_param_columns(df)

	df.rename(columns={
		'job_params.gen_model_params.encode_type': 'Encoder type',
		'job_params.gen_model_params.n_levels': '# levels',
		'job_params.gen_model_params.conv_per_level': 'Conv. per level',
		'job_params.gen_model_params.arch_options': 'Arch. options',
		'job_params.gen_model_params.n_filters': '# filters',
		'job_params.gen_model_params.width_factor': 'Width factor',
		'job_params.gen_model_params.loss_types': 'Loss function',
	}, inplace=True)

	y = ['Gen. L2 loss', 'Gen. GAN loss', 'Disc. GAN loss', 'K-L divergence']
	results.plot_lines('ARC2019/loss_lines.png', df,
		x='Iteration', y=y, hue='Model type', width=8, height=12, lgd_title=False,
		ylim=[(-100, 1100), (-0.4, 4.4), (-0.4, 4.4), (-400, 4400)])

	df = df.reset_index()
	for m in df['model_name'].unique():
		print(m)

#################################### ARCHITECTURE SEARCH ####################################

if False:
	base_pattern = '/*.lowrmsd.0.all.training_output'
	ae_l2_files = glob.glob('*ae13_24_0.5_*/' + base_pattern) + glob.glob('*l-le13_24_0.5_*/' + base_pattern)
	ce_l2_files = glob.glob('*ce13_24_0.5_*/' + base_pattern)

	ae_l2_dfs = read_out_files(ae_l2_files, 'AE L2')
	ce_l2_dfs = read_out_files(ce_l2_files, 'CE L2')

	print('{} AE L2 models'.format(len(ae_l2_dfs)))
	print('{} CE L2 models'.format(len(ce_l2_dfs)))
	df = pd.concat(ae_l2_dfs + ce_l2_dfs)

	df = df.set_index(['model_name', 'Iteration'])
	params = results.add_param_columns(df)


	df.rename(columns={
		'job_params.gen_model_params.encode_type': 'Encoder type',
		'job_params.gen_model_params.n_levels': '# conv. levels',
		'job_params.gen_model_params.conv_per_level': 'Conv. per level',
		'job_params.gen_model_params.arch_options': 'Arch. options',
		'job_params.gen_model_params.n_filters': '# filters',
		'job_params.gen_model_params.width_factor': 'Width factor',
		 'job_params.gen_model_params.n_latent': 'Latent dim.',
		'job_params.gen_model_params.loss_types': 'Loss function',
	}, inplace=True)

	df = df.reset_index()
	df = df.set_index('Iteration').loc[20000]

	df.loc[df['Encoder type'] == 'a', 'Encoder type'] = 'AE'
	df.loc[df['Encoder type'] == 'va', 'Encoder type'] = 'AE'
	df.loc[df['Encoder type'] == 'l-l', 'Encoder type'] = 'AE'
	df.loc[df['Encoder type'] == 'vl-l', 'Encoder type'] = 'AE'

	df.loc[df['Encoder type'] == 'c', 'Encoder type'] = 'CE'
	df.loc[df['Encoder type'] == 'vc', 'Encoder type'] = 'CE'
	df.loc[df['Encoder type'] == 'r-l', 'Encoder type'] = 'CE'
	df.loc[df['Encoder type'] == 'vr-l', 'Encoder type'] = 'CE'

	df.loc[df['Loss function'] == 'e', 'Loss function'] = 'L2'
	df.loc[df['Loss function'] == 'em','Loss function'] = 'L2'
	df.loc[df['Loss function'] == 'g','Loss function'] = 'GAN'
	df.loc[df['Loss function'] == 'eg','Loss function'] = 'L2+GAN'
	df.loc[df['Loss function'] == 'emg','Loss function'] = 'L2+GAN'


	df.to_csv('ARC2019/model_data.csv')

df = pd.read_csv('ARC2019/model_data.csv')

df = df[df['Encoder type'] == 'AE']
df = df[df['Loss function'] != 'GAN']

print(df)

x = ['# conv. levels', 'Conv. per level', '# filters', 'Width factor', 'Latent dim.', 'Loss function']
y = ['Gen. L2 loss'] #, 'Gen. GAN loss', 'Disc. GAN loss', 'K-L divergence']
results.plot_strips('ARC2019/loss_strips.png', df, x=x, y=y, hue='# conv. levels',
	width=8, height=8, jitter=0.1, n_cols=3)
