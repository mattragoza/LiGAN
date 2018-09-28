import sys, os, re
import numpy as np
import scipy as sp
import caffe
caffe.set_mode_gpu()
import caffe_util
import generate


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


model_file = 'models/vr-le13_12_0.5_1_2l_8_1_8_.model'
model_name = 'adam2_2_2__0.01_vr-le13_12_0.5_1_2l_8_1_8__d11_12_1_1l_16_1_x'
weights_file = '{}/{}.two_atoms.0.all_gen_iter_25000.caffemodel'.format(model_name, model_name)

lig_files = ['data/O_2_0_0.sdf']
rec_files = ['data/O_0_0_0.pdb']
centers = [np.zeros(3) for l in lig_files]
data_file = generate.get_temp_data_file(zip(rec_files, lig_files))

net_param = caffe_util.NetParameter.from_prototxt(model_file)
net_param.set_molgrid_data_source(data_file, '')

channels = generate.channel_info.get_default_channels(False, True, True)

net = caffe_util.Net.from_param(net_param, weights_file, phase=caffe.TEST)
grid_blobs = dict(rec=net.blobs['rec'],
                  lig=net.blobs['lig'],
                  lig_gen=net.blobs['lig_gen'])

net.forward()
net.blobs['rec_latent_mean'].data[...] = 0.0
net.blobs['rec_latent_std'].data[...] = 1.0
net.forward(start='rec_latent_noise')

n = 50
z = 0.5
v0 = np.array(net.blobs['rec_latent_sample'].data[0])
v1 = np.array(net.blobs['rec_latent_sample'].data[1])
v0 = z * v0 / np.linalg.norm(v0)
v1 = z * v1 / np.linalg.norm(v1)
net.blobs['rec_latent_sample'].data[0:n//2] = slerp(v0, v1, np.linspace(0, 1, n//2))
net.blobs['rec_latent_sample'].data[n//2:n] = slerp(v1, v0, np.linspace(0, 1, n//2))

net.forward(start='lig_latent_defc')

dx_groups = {}
for name, grid_blob in grid_blobs.items():
    if name == 'lig_gen':
        for i in range(n):
            grid_name = '2A_{}_z{}_{}'.format(name, z, i)
            grid = np.array(grid_blob.data[i])
            dx_groups[grid_name] = generate.write_grids_to_dx_files(grid_name, grid, channels, np.zeros(3), 0.5)
    else:
        grid_name = '2A_{}_z{}'.format(name, z)
        grid = np.array(grid_blob.data[0])
        dx_groups[grid_name] = generate.write_grids_to_dx_files(grid_name, grid, channels, np.zeros(3), 0.5)

other_files = rec_files + lig_files
centers = centers + centers
generate.write_pymol_script('2A.pymol', dx_groups, other_files, centers)
