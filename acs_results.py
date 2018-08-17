import sys, os, re
import numpy as np
from collections import defaultdict
import caffe
caffe.set_mode_gpu()
import caffe_util
import generate

model_file = 'models/rvl-le13_24_0.5_3_3_32_2_1024_e.model'
model_name = 'abgan_rvl-le13_24_0.5_3_3_32_2_1024_e_disc2'
weights_file = '{}/{}.lowrmsd.0.0_gen_iter_20000.caffemodel'.format(model_name, model_name)
data_root = '/home/mtr22/PDBbind/refined-set/'
targets = ['2avo', '2pwc']

lig_files = [data_root + '{}/{}_min.sdf'.format(t, t) for t in targets]
rec_files = [data_root + '{}/{}_rec.pdb'.format(t, t) for t in targets]
centers = [generate.get_center_from_sdf_file(l) for l in lig_files]
data_file = generate.get_temp_data_file(zip(rec_files, lig_files))

net_param = caffe_util.NetParameter.from_prototxt(model_file)
net_param.set_molgrid_data_source(data_file, '')

channels = generate.channel_info.get_default_channels(False, True, True)

net = caffe_util.Net.from_param(net_param, weights_file, phase=caffe.TEST)
latent_blobs = dict(rec=net.blobs['rec_latent_fc'], lig=net.blobs['lig_latent_sample'])

net.forward(end='latent_concat')
net.blobs['lig_latent_std'] = 0.0
net.forward(start='lig_latent_noise')

latent_vecs = defaultdict(dict)
for i, target in enumerate(targets):
	for name, blob in latent_blobs.items():
		latent_vecs[target][name] = np.array(blob.data[i])

for name, blob in latent_blobs.items():
	latent_vecs['zero'][name] = np.zeros(blob.shape[1:])
	blob.data[...] = 0.0

for target in latent_vecs:
	for name in latent_vecs[target]:
		latent_vec = latent_vecs[target][name]
		print name, target, latent_vec.shape, np.linalg.norm(latent_vec)

grid_names = []
for rec_target in latent_vecs:
	for lig_target in latent_vecs:
		idx = len(grid_names)
		latent_blobs['rec'].data[idx,...] = latent_vecs[rec_target]['rec']
		latent_blobs['lig'].data[idx,...] = latent_vecs[lig_target]['lig']
		grid_names.append('REC_{}_LIG_{}'.format(rec_target, lig_target))
		print grid_names[-1]

net.forward(start='latent_concat')

dx_groups = {}
for i, grid_name in enumerate(grid_names):
	grid = net.blobs['lig_gen'].data[i]
	dx_groups[grid_name] = generate.write_grids_to_dx_files(grid_name, grid, channels, np.zeros(3), 0.5)

other_files = rec_files + lig_files
centers = centers + centers
generate.write_pymol_script('REC.pymol', dx_groups, other_files, centers)
