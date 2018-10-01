import sys, os, re
import numpy as np
import scipy as sp
import caffe
caffe.set_mode_gpu()
import caffe_util
import generate as g


def slerp(v0, v1, t):
    norm_v0 = np.linalg.norm(v0)
    norm_v1 = np.linalg.norm(v1)
    dot_v0_v1 = np.dot(v0, v1)
    cos_theta = dot_v0_v1 / (norm_v0 * norm_v1)
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)
    s0 = np.sin((1.0-t)*theta) / sin_theta
    s1 = np.sin(t*theta) / sin_theta
    return s0[:,np.newaxis] * v0[np.newaxis,:] \
         + s1[:,np.newaxis] * v1[np.newaxis,:]


def lerp(v0, v1, t):
    return (1.0-t)[:,np.newaxis] * v0[np.newaxis,:] \
               + t[:,np.newaxis] * v1[np.newaxis,:]


encode = 'l'
n_latent = 16
loss = ''
iter_ = 50

name = '{}{}{}'.format(encode, n_latent, loss)
model_file = 'models/{}-le13_24_0.5_2_1l_8_1_{}_{}.model'.format(encode, n_latent, loss)
model_name = 'adam2_2_2__0.01_{}-le13_24_0.5_2_1l_8_1_{}_{}_d11_24_2_1l_16_1_x'.format(encode, n_latent, loss)

data_root = '/net/pulsar/home/koes/dkoes/PDBbind/refined-set/'
rec_file = data_root + '1ai5/1ai5_rec.pdb'
lig_file = data_root + '1ai5/1ai5_min.sdf'
center = g.get_center_from_sdf_file(lig_file)
data_file = g.get_temp_data_file([(rec_file, lig_file)])
channels = g.channel_info.get_default_channels(rec=False, lig=True, use_covalent_radius=True)

weights_file = '{}/{}.1ai5.0.all_gen_iter_{}000.caffemodel'.format(model_name, model_name, iter_)
net_param = caffe_util.NetParameter.from_prototxt(model_file)
net_param.set_molgrid_data_source(data_file, '')
data_param = net_param.get_molgrid_data_param(caffe.TEST)
data_param.random_rotation = True

net = caffe_util.Net.from_param(net_param, weights_file, phase=caffe.TEST)

latent_name = '{}_latent_fc'.format(dict(r='rec', l='lig')[encode])
after_latent_name = 'lig_latent_defc'
net.forward(end=latent_name)
n_samples = n = net.blobs[latent_name].shape[0]
v0 = np.array(net.blobs[latent_name].data[0])
v1 = np.array(net.blobs[latent_name].data[1])
net.blobs[latent_name].data[0:n//2] = lerp(v0, v1, np.linspace(0, 1, n//2))
net.blobs[latent_name].data[n//2:n] = lerp(v1, v0, np.linspace(0, 1, n//2))
net.forward(start=after_latent_name)

dx_groups = {}
blob_names = ['lig_gen']
for j in range(n_samples):
    for blob_name in blob_names:
        grid_name = '{}_{}k_{}_{}'.format(name, iter_, blob_name, j)
        grid = np.array(net.blobs[blob_name].data[j])
        dx_groups[grid_name] = g.write_grids_to_dx_files(grid_name, grid, channels, center, 0.5)

pymol_file = '{}.pymol'.format(name)
other_files = [rec_file, lig_file]
g.write_pymol_script(pymol_file, dx_groups, other_files, [center, center])
