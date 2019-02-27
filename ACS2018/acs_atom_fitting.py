import sys, os, re, shutil
import numpy as np
from collections import defaultdict
import multiprocessing
import caffe
caffe.set_mode_gpu()
import caffe_util
import generate

model_file = '../models/rvl-le13_24_0.5_3_3_32_2_1024_e.model'
model_name = 'abgan_rvl-le13_24_0.5_3_3_32_2_1024_e_disc2'
weights_file = '../{}/{}.lowrmsd.0.all_gen_iter_20000.caffemodel'.format(model_name, model_name)
data_root = '/home/mtr22/PDBbind/refined-set/'
targets = ['2avo']
['1ydt',
'4a7i',
'3l4w',
'1nvq',
'3oe4',
'3tzm',
'1lbf',
'3tmk',
'4pyx',
'4p6w']

lig_files = [data_root + '{}/{}_min.sdf'.format(t, t) for t in targets]
rec_files = [data_root + '{}/{}_rec.pdb'.format(t, t) for t in targets]
centers = [generate.get_center_from_sdf_file(l) for l in lig_files]
data_file = generate.get_temp_data_file(zip(rec_files, lig_files))

net_param = caffe_util.NetParameter.from_prototxt(model_file)
net_param.set_molgrid_data_source(data_file, '')

channels = generate.channel_info.get_default_channels(False, True, True)

net = caffe_util.Net.from_param(net_param, weights_file, phase=caffe.TEST)
grid_blobs = dict(lig_gen=net.blobs['lig_gen'])

net.forward(end='latent_concat')
net.blobs['lig_latent_std'] = 0.0
net.forward(start='lig_latent_noise')

for other_file in rec_files + lig_files:
	shutil.copyfile(other_file, os.path.basename(other_file))
rec_files = [os.path.basename(f) for f in rec_files]
lig_files = [os.path.basename(f) for f in lig_files]

dx_groups = []
lig_gen_grids = dict()
for grid_name, grid_blob in grid_blobs.items():
	grid = np.array(grid_blob.data[0])
	generate.write_grids_to_dx_files(grid_name, grid, channels, np.zeros(3), 0.5)
	dx_groups.append(grid_name)
	if grid_name == 'lig_gen':
		lig_gen_grids[grid_name] = grid

lig_gen_grid = lig_gen_grids['lig_gen']
wd = lambda x,n,r: generate.wiener_deconv_grids(x, channels, np.zeros(3), 0.5, 1.0, n, r)

grid_name, grid = 'lig_gen_A', 8.0*lig_gen_grid
generate.write_grids_to_dx_files(grid_name, grid, channels, np.zeros(3), 0.5)
lig_gen_grids[grid_name] = grid
dx_groups.append(grid_name)

fit_algos = dict(
	fit_A=dict(greedy=True,
			   bonded=True,
		       max_init_bond_E=0.2,
		       lambda_E=0.2,
		       radius_factor=1.33),
)

def do_fit(args):
	(grid_name, grid), (fit_name, fit_algo) = args
	xyz, c, bonds, loss = generate.fit_atoms_to_grids(grid, channels, centers[0], 0.5, np.inf, 1.0, 
		                                              verbose=2, all_iters=True, **fit_algo)
	elems = [[channels[i][1] for i in c_] for c_ in c]
	fit_file = '{}_{}.sdf'.format(grid_name, fit_name)
	generate.write_xyz_elems_bonds_to_sdf_file(fit_file, zip(xyz, elems, bonds))
	return fit_file

args = [(g, f) for g in lig_gen_grids.items() for f in fit_algos.items()]
fit_files = multiprocessing.Pool(len(args)).map(do_fit, args)

generate.write_pymol_script('ATOM_FITTING.pymol', dx_groups, \
							rec_files + lig_files + fit_files, \
							2*centers + [centers[0] for _ in fit_files])
print 'done'
