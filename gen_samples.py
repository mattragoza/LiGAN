import sys, os, re
import numpy as np
import scipy as sp
import caffe
caffe.set_mode_gpu()
import caffe_util
import generate as g


model_file = 'models/r-le13_24_0.5_2_1l_8_1_8_e.model'
model_name = os.path.splitext(os.path.basename(model_file))[0]
model_name = 'adam2_2_2_s_0.01_' + model_name + '_d11_24_2_1l_16_1_x'
weights_file = '{}/{}.1ai5.0.all_gen_iter_50000.caffemodel'.format(model_name, model_name)

data_root = '/net/pulsar/home/koes/dkoes/PDBbind/refined-set/'
lig_file = data_root + '1ai5/1ai5_min.sdf'
rec_file = data_root + '1ai5/1ai5_rec.pdb'
data_file = g.get_temp_data_file([(rec_file, lig_file)])

lig_mol = g.read_mols_from_sdf_file(lig_file)[0]
lig_mol.removeh()
center = g.get_mol_center(lig_mol)

net_param = caffe_util.NetParameter.from_prototxt(model_file)
net_param.set_molgrid_data_source(data_file, '')

channels = g.atom_types.get_default_lig_channels(use_covalent_radius=True)

net = caffe_util.Net.from_param(net_param, weights_file, phase=caffe.TRAIN)
blob_names = ['rec', 'lig', 'lig_gen']
n_samples = 4

net.forward()

grid_names = []
for i in range(n_samples):
    for blob_name in blob_names:
        grid_name = '1T_{}_{}'.format(blob_name, i)
        grid = np.array(net.blobs[blob_name].data[i])
        g.write_grids_to_dx_files(grid_name, grid, channels, center, 0.5)
        grid_names.append(grid_name)

g.write_pymol_script('1T.pymol', grid_names, [rec_file, lig_file], 2*[center])
