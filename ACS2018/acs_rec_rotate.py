import sys, os, re, shutil
import numpy as np
from collections import defaultdict, OrderedDict
import caffe
caffe.set_mode_gpu()
import caffe_util
import generate

model_file = '../models/rvl-le13_24_0.5_3_3_32_2_1024_e.model'
model_name = 'abgan_rvl-le13_24_0.5_3_3_32_2_1024_e_disc2'
weights_file = '../{}/{}.lowrmsd.0.0_gen_iter_20000.caffemodel'.format(model_name, model_name)
data_root = '/home/mtr22/PDBbind/refined-set/'
targets = ['1ai5']

lig_files = [data_root + '{}/{}_min.sdf'.format(t, t) for t in targets]
rec_files = [data_root + '{}/{}_rec.pdb'.format(t, t) for t in targets]
centers = [generate.get_center_from_sdf_file(l) for l in lig_files]
data_file = generate.get_temp_data_file(zip(rec_files, lig_files))

net_param = caffe_util.NetParameter.from_prototxt(model_file)
data_param = net_param.layer[1].molgrid_data_param
data_param.batch_rotate = True
data_param.batch_rotate_yaw = 2*np.pi/data_param.batch_size
net_param.set_molgrid_data_source(data_file, '')

channels = generate.channel_info.get_default_channels(False, True, True)

net = caffe_util.Net.from_param(net_param, weights_file, phase=caffe.TEST)

net.forward(end='latent_concat')
net.blobs['lig_latent_sample'].data[...] = 0.0
net.forward(start='latent_concat')

for other_file in rec_files + lig_files:
	shutil.copyfile(other_file, os.path.basename(other_file))
rec_files = [os.path.basename(f) for f in rec_files]
lig_files = [os.path.basename(f) for f in lig_files]

dx_groups = []
for i in range(data_param.batch_size):
	grid = net.blobs['lig_gen'].data[i]
	grid_name = 'ROTATE_{}'.format(i)
	generate.write_grids_to_dx_files(grid_name, grid, channels, np.zeros(3), 0.5)
	dx_groups.append(grid_name)

generate.write_pymol_script('ROTATE.pymol', dx_groups, rec_files + lig_files, 2*centers)
print('done')
