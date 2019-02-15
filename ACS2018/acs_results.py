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
targets = ['3dx1']

lig_files = [data_root + '{}/{}_min.sdf'.format(t, t) for t in targets]
rec_files = [data_root + '{}/{}_rec.pdb'.format(t, t) for t in targets]
centers = [generate.get_center_from_sdf_file(l) for l in lig_files]
data_file = generate.get_temp_data_file(zip(rec_files, lig_files))

net_param = caffe_util.NetParameter.from_prototxt(model_file)
net_param.set_molgrid_data_source(data_file, '')

channels = generate.channel_info.get_default_channels(False, True, True)

net = caffe_util.Net.from_param(net_param, weights_file, phase=caffe.TEST)
grid_blobs = dict(rec=net.blobs['rec'],
	              lig=net.blobs['lig'],
	              lig_gen=net.blobs['lig_gen'])

net.forward(end='latent_concat')
net.blobs['lig_latent_std'] = 0.0
net.forward(start='lig_latent_noise')

dx_groups = {}
for name, grid_blob in grid_blobs.items():
	grid_name = 'GEN_{}_{}'.format(targets[0], name)
	grid = np.array(grid_blob.data[0])
	dx_groups[grid_name] = generate.write_grids_to_dx_files(grid_name, grid, channels, np.zeros(3), 0.5)

other_files = rec_files + lig_files
centers = centers + centers
generate.write_pymol_script('GEN.pymol', dx_groups, other_files, centers)
