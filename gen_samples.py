from __future__ import print_function, division
import sys, os, re, itertools
from collections import defaultdict
import numpy as np
import scipy as sp

import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', 80)

import caffe
caffe.set_mode_gpu()

import caffe_util
import generate as g


def permute(a):
    p = itertools.permutations(a)
    return np.array(list(p))


def fit_atoms_to_grid(grid, center, resolution, c, r):
    points = g.get_grid_points(grid.shape[1:], center, resolution)
    density = grid.reshape((grid.shape[0], -1)).T
    xyz = np.zeros((0, 3))
    for i, c_ in enumerate(c):
        max_d = 0.0
        for p, d in zip(points, density[:,c_]):
            if d > max_d:
                dist2 = ((p - xyz[c[:i] == c_])**2).sum(axis=1)
                if all(dist2 > r[c_]**2):
                    max_p = p
                    max_d = d
        xyz = np.append(xyz, max_p[np.newaxis,:], axis=0)
    xyz, density, loss = g.fit_atoms_by_GD(points, density, xyz, c, [], r, 1.5, np.inf, lr=0.05)
    return xyz, density.T.reshape(grid.shape)


def min_rmsd(xyz1, xyz2, c):
    ssd = 0.0
    for c_ in sorted(set(c)):
        xyz1_c = xyz1[c == c_]
        min_ssd_c = np.inf
        for xyz2_c in permute(xyz2[c == c_]):
            ssd_c = ((xyz2_c - xyz1_c)**2).sum()
            if ssd_c < min_ssd_c:
                min_ssd_c = ssd_c
        ssd += min_ssd_c
    return np.sqrt(ssd/len(c))


if __name__ == '__main__':

    out_name = '1AF'
    data_name = 'two_atoms'
    iter_ = 50000

    model_file = 'models/vr-le13_12_0.5_2_1lg_8_2_16_f.model'
    model_name = 'adam2_2_2_b_0.01_vr-le13_12_0.5_2_1lg_8_2_16_f_d11_12_2_1l_8_1_x'
    weights_file = '{}/{}.{}.0.all_gen_iter_{}.caffemodel'.format(model_name, model_name, data_name, iter_)

    data_root = '/net/pulsar/home/koes/mtr22/gan/data/' #dkoes/PDBbind/refined-set/'
    lig_file = data_root + 'O_2_0_0.sdf' #'1ai5/1ai5_min.sdf'
    rec_file = data_root + 'O_0_0_0.pdb' #'1ai5/1ai5_rec.pdb'
    data_file = g.get_temp_data_file([(rec_file, lig_file)])

    net_param = caffe_util.NetParameter.from_prototxt(model_file)
    net_param.set_molgrid_data_source(data_file, '')
    data_param = net_param.get_molgrid_data_param(caffe.TEST)
    data_param.random_rotation = True
    data_param.fix_center_to_origin = True
    resolution = data_param.resolution

    net = caffe_util.Net.from_param(net_param, weights_file, phase=caffe.TEST)
    net.forward()

    lig_mol = g.read_mols_from_sdf_file(lig_file)[0]
    lig_mol.removeh()
    center = g.get_mol_center(lig_mol)

    use_covalent_radius = True
    nr = 0 #len(g.atom_types.get_default_rec_channels(use_covalent_radius))
    channels = g.atom_types.get_default_lig_channels(use_covalent_radius)

    lig_types = lig_file.replace('.sdf', '.gninatypes')
    _, c = g.read_gninatypes_file(lig_types, channels)
    r = np.array([channel.atomic_radius for channel in channels])

    df = pd.DataFrame()
    grids = defaultdict(list)
    xyzs = defaultdict(list)

    blob_names = ['lig', 'lig_name']

    n_samples = 10
    for i in range(n_samples):

        #rec_grid = np.array(net.blobs['rec'].data[i])
        true_grid = np.array(net.blobs['lig'].data[i][nr:])
        gen_grid = np.array(net.blobs['lig_gen'].data[i][nr:])

        true_xyz, _ = fit_atoms_to_grid(true_grid, center, resolution, c, r)
        fit_xyz, fit_grid = fit_atoms_to_grid(gen_grid, center, resolution, c, r)

        df.loc[i, 'true_L2'] = np.linalg.norm(true_grid)
        df.loc[i, 'gen_L2'] = np.linalg.norm(gen_grid)
        df.loc[i, 'fit_L2'] = np.linalg.norm(fit_grid)

        df.loc[i, 'gen_true_L2'] = np.linalg.norm(true_grid - gen_grid)
        df.loc[i, 'fit_true_L2'] = np.linalg.norm(true_grid - fit_grid)
        df.loc[i, 'fit_gen_L2'] = np.linalg.norm(gen_grid - fit_grid)

        df.loc[i, 'fit_true_rmsd'] = min_rmsd(true_xyz, fit_xyz, c)

        #grids['rec'].append(rec_grid)
        grids['true'].append(true_grid)
        grids['gen'].append(gen_grid)
        grids['fit'].append(fit_grid)

        xyzs['true'].append(true_xyz)
        xyzs['fit'].append(fit_xyz)

    k = 0
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            
            true_grid_i = grids['true'][i]
            true_grid_j = grids['true'][j]
            df.loc[k, 'true_true_L2'] = np.linalg.norm(true_grid_j - true_grid_i)

            gen_grid_i = grids['gen'][i]
            gen_grid_j = grids['gen'][j]
            df.loc[k, 'gen_gen_L2'] = np.linalg.norm(gen_grid_j - gen_grid_i)

            fit_grid_i = grids['fit'][i]
            fit_grid_j = grids['fit'][j]
            df.loc[k, 'fit_fit_L2'] = np.linalg.norm(fit_grid_j - fit_grid_i)

            true_xyz_i = xyzs['true'][i]
            true_xyz_j = xyzs['true'][j]
            df.loc[k, 'true_true_rmsd'] = min_rmsd(true_xyz_i, true_xyz_j, c)

            fit_xyz_i = xyzs['fit'][i]
            fit_xyz_j = xyzs['fit'][j]
            df.loc[k, 'fit_fit_rmsd'] = min_rmsd(fit_xyz_i, fit_xyz_j, c)
            k += 1

    agg_df = pd.DataFrame(dict(mean=df.mean(), std=df.std()))
    print(agg_df)

    grid_names = []
    for grid_type in grids:
        for i, grid in enumerate(grids[grid_type]):
            grid_name = '{}_{}_{}'.format(out_name, grid_type, i)
            g.write_grids_to_dx_files(grid_name, grid, channels, center, resolution)
            grid_names.append(grid_name)

    extra_files = [rec_file, lig_file]
    for xyz_type in xyzs:
        fit_file = '{}_{}.sdf'.format(out_name, xyz_type, i)
        mols = [g.make_ob_mol(xyz, c, [], channels=channels) for xyz in xyzs[xyz_type]]
        g.write_ob_mols_to_sdf_file(fit_file, mols)
        extra_files.append(fit_file)

    g.write_pymol_script('{}.pymol'.format(out_name), grid_names, extra_files, len(extra_files)*[center])
