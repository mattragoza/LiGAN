import sys, os, re
import numpy as np
from collections import defaultdict
import multiprocessing
import caffe
caffe.set_mode_gpu()
import caffe_util
import generate

model_file = 'models/rvl-le13_24_0.5_3_3_32_2_1024_e.model'
model_name = 'abgan_rvl-le13_24_0.5_3_3_32_2_1024_e_disc2'
weights_file = '{}/{}.lowrmsd.0.0_gen_iter_20000.caffemodel'.format(model_name, model_name)
data_root = '/home/mtr22/PDBbind/refined-set/'
targets = ['2pwc']

lig_files = [data_root + '{}/{}_min.sdf'.format(t, t) for t in targets]
rec_files = [data_root + '{}/{}_rec.pdb'.format(t, t) for t in targets]
centers = [generate.get_center_from_sdf_file(l) for l in lig_files]
data_file = generate.get_temp_data_file(zip(rec_files, lig_files))

net_param = caffe_util.NetParameter.from_prototxt(model_file)
net_param.set_molgrid_data_source(data_file, '')

channels = generate.channel_info.get_default_channels(False, True, True)

net = caffe_util.Net.from_param(net_param, weights_file, phase=caffe.TEST)
grid_blobs = dict(rec=net.blobs['rec'], lig=net.blobs['lig'], lig_gen=net.blobs['lig_gen'])

net.forward(end='latent_concat')
net.blobs['lig_latent_std'] = 0.0
net.forward(start='lig_latent_noise')

dx_groups = {}
lig_gen_grids = dict()
for grid_name, grid_blob in grid_blobs.items():
	grid = np.array(grid_blob.data[0])
	dx_groups[grid_name] = generate.write_grids_to_dx_files(grid_name, grid, channels, np.zeros(3), 0.5)
	if grid_name == 'lig_gen':
		lig_gen_grids[grid_name] = grid

lig_gen_grid = lig_gen_grids['lig_gen']
wd = lambda x,n,r: generate.wiener_deconv_grids(x, channels, np.zeros(3), 0.5, 1.0, n, r)

grid_name, grid = 'lig_gen_A', 5.0*lig_gen_grid
dx_groups[grid_name] = generate.write_grids_to_dx_files(grid_name, grid, channels, np.zeros(3), 0.5)
lig_gen_grids[grid_name] = grid

fit_algos = dict(
	fit_A=dict(greedy=True,
			   bonded=True,
		       max_init_bond_E=0.5,
		       lambda_E=0.1,
		       radius_factor=1.2),

	fit_B=dict(greedy=True,
			   bonded=True,
		       max_init_bond_E=0.5,
		       lambda_E=0.2,
		       radius_factor=1.2),

	fit_C=dict(greedy=True,
			   bonded=True,
		       max_init_bond_E=0.5,
		       lambda_E=0.1,
		       radius_factor=1.3),
)

def do_fit(args):
	(grid_name, grid), (fit_name, fit_algo) = args
	xyz, c, bonds, loss = generate.fit_atoms_to_grids(grid, channels, centers[0], 0.5, np.inf, 1.0, 
		                                              verbose=2, all_iters=True, **fit_algo)
	elems = [[channels[i][1] for i in c_] for c_ in c]
	fit_file = '{}_{}.sdf'.format(grid_name, fit_name)
	generate.write_xyz_elems_bonds_to_sdf_file(fit_file, zip(xyz, elems, bonds)[-1:])
	return fit_file

args = [(g, f) for g in lig_gen_grids.items() for f in fit_algos.items()]
fit_files = multiprocessing.Pool(len(args)).map(do_fit, args)


generate.write_pymol_script('ATOM_FITTING.pymol', dx_groups, \
							rec_files + lig_files + fit_files, \
							2*centers + [centers[0] for _ in fit_files])
print 'done'
