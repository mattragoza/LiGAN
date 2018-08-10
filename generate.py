from __future__ import print_function
import sys, os, re, argparse, ast, time
import numpy as np
from rdkit import Chem
from collections import Counter
import contextlib
import tempfile
from multiprocessing.pool import Pool, ThreadPool
from itertools import izip
from functools import partial
from scipy.stats import multivariate_normal
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
import openbabel as ob

import caffe_util
import channel_info


def get_atom_density(atom_pos, atom_radius, points, radius_multiple):
    '''
    Compute the density value of an atom at a set of points.
    '''
    dist2 = np.sum((points - atom_pos)**2, axis=1)
    dist = np.sqrt(dist2)
    h = 0.5*atom_radius
    ie2 = np.exp(-2)
    zero_cond = dist >= radius_multiple * atom_radius
    gauss_cond = dist <= atom_radius
    gauss_val = np.exp(-dist2 / (2*h**2))
    quad_val = dist2*ie2/(h**2) - 6*dist*ie2/h + 9*ie2
    return np.where(zero_cond, 0.0, np.where(gauss_cond, gauss_val, quad_val))


def get_atom_gradient(atom_pos, atom_radius, points, radius_multiple):
    '''
    Compute the derivative of an atom's density with respect
    to a set of points.
    '''
    diff = points - atom_pos
    dist2 = np.sum(diff**2, axis=1)
    dist = np.sqrt(dist2)
    h = 0.5*atom_radius
    ie2 = np.exp(-2)
    zero_cond = np.logical_or(dist >= radius_multiple * atom_radius, np.isclose(dist, 0))
    gauss_cond = dist <= atom_radius
    gauss_val = -dist / h**2 * np.exp(-dist2 / (2*h**2))
    quad_val = 2*dist*ie2/(h**2) - 6*ie2/h
    return -diff * np.where(zero_cond, 0.0, np.where(gauss_cond, gauss_val, quad_val) / dist)[:,np.newaxis]


def get_interatomic_energy(atom_pos1, atom_pos2, bond_length, bonds, width_factor=1.0):
    '''
    Compute the interatomic potential energy between an atom and a set of atoms.
    '''
    dist = np.sqrt(np.sum((atom_pos2 - atom_pos1)**2, axis=1))
    exp = np.exp(-width_factor*(dist - bond_length))
    return (1 - exp)**2 * bonds


def get_interatomic_forces(atom_pos1, atom_pos2, bond_length, bonds, width_factor=1.0):
    '''
    Compute the derivative of interatomic potential energy between an atom
    and a set of atoms with respect to the position of the first atom.
    '''
    diff = atom_pos2 - atom_pos1
    dist2 = np.sum(diff**2, axis=1)
    dist = np.sqrt(dist2)
    exp = np.exp(-width_factor*(dist - bond_length))
    d_energy = 2 * (1 - exp) * exp * width_factor * bonds
    return (-diff * (d_energy / dist)[:,np.newaxis])


def fit_atoms_by_GMM(points, density, xyz_init, atom_radius, radius_multiple, max_iter, 
                     noise_model='', noise_params_init={}, gof_crit='nll', verbose=0):
    '''
    Fit atom positions to a set of points with the given density values with
    a Gaussian mixture model (and optional noise model). Return the final atom
    positions and a goodness-of-fit criterion (negative log likelihood, Akaike
    information criterion, or L2 loss).
    '''
    assert gof_crit in {'nll', 'aic', 'L2'}, 'Invalid value for gof_crit argument'
    n_points = len(points)
    n_atoms = len(xyz_init)
    xyz = np.array(xyz_init)
    atom_radius = np.array(atom_radius)
    cov = (0.5*atom_radius)**2
    n_params = xyz.size

    assert noise_model in {'d', 'p', ''}, 'Invalid value for noise_model argument'
    if noise_model == 'd':
        noise_mean = noise_params_init['mean']
        noise_cov = noise_params_init['cov']
        n_params += 2
    elif noise_model == 'p':
        noise_prob = noise_params_init['prob']
        n_params += 1

    # initialize uniform prior over components
    n_comps = n_atoms + bool(noise_model)
    assert n_comps > 0, 'Need at least one component (atom or noise model) to fit GMM'
    P_comp = np.full(n_comps, 1.0/n_comps) # P(comp_j)
    n_params += n_comps - 1

    # maximize expected log likelihood
    ll = -np.inf
    i = 0
    while True:

        L_point = np.zeros((n_points, n_comps)) # P(point_i|comp_j)
        for j in range(n_atoms):
            L_point[:,j] = multivariate_normal.pdf(points, mean=xyz[j], cov=cov[j])
        if noise_model == 'd':
            L_point[:,-1] = multivariate_normal.pdf(density, mean=noise_mean, cov=noise_cov)
        elif noise_model == 'p':
            L_point[:,-1] = noise_prob

        P_joint = P_comp * L_point          # P(point_i, comp_j)
        P_point = np.sum(P_joint, axis=1)   # P(point_i)
        gamma = (P_joint.T / P_point).T     # P(comp_j|point_i) (E-step)

        # compute expected log likelihood
        ll_prev, ll = ll, np.sum(density * np.log(P_point))
        if ll - ll_prev < 1e-3 or i == max_iter:
            break

        # estimate parameters that maximize expected log likelihood (M-step)
        for j in range(n_atoms):
            xyz[j] = np.sum(density * gamma[:,j] * points.T, axis=1) \
                   / np.sum(density * gamma[:,j])
        if noise_model == 'd':
            noise_mean = np.sum(gamma[:,-1] * density) / np.sum(gamma[:,-1])
            noise_cov = np.sum(gamma[:,-1] * (density - noise_mean)**2) / np.sum(gamma[:,-1])
            if noise_cov == 0.0 or np.isnan(noise_cov): # reset noise
                noise_mean = noise_params_init['mean']
                noise_cov = noise_params_init['cov']
        elif noise_model == 'p':
            noise_prob = noise_prob
        if noise_model and n_atoms > 0:
            P_comp[-1] = np.sum(density * gamma[:,-1]) / np.sum(density)
            P_comp[:-1] = (1.0 - P_comp[-1])/n_atoms
        i += 1
        if verbose > 2:
            print('iteration = {}, nll = {} ({})'.format(i, -ll, -(ll - ll_prev)), file=sys.stderr)

    # compute the goodness-of-fit
    if gof_crit == 'L2':
        density_pred = np.zeros_like(density)
        for j in range(n_atoms):
            density_pred += get_atom_density(xyz[j], atom_radius[j], points, radius_multiple)
        gof = np.sum((density_pred - density)**2)/2
    elif gof_crit == 'aic':
        gof = 2*n_params - 2*ll
    else:
        gof = -ll

    return xyz, gof


def fit_atoms_by_GD(points, density, xyz_init, c, bonds, atom_radius, radius_multiple,
                    max_iter, lr=0.01, mo=0.9, lambda_E=0.0, verbose=0):
    '''
    Fit atom positions, provided by arrays xyz_init of initial positions, c of
    channel indices, and atom_radius of atom radius for each channel, to arrays
    of points with the given channel density values. Minimize the L2 loss (and
    optionally interatomic energy) between the true channel density and predicted
    channel density by gradient descent with momentum. Return the final atom
    positions and loss.
    '''
    n_atoms = len(xyz_init)
    xyz = np.array(xyz_init)
    d_xyz = np.zeros_like(xyz)
    d_xyz_prev = np.zeros_like(xyz)
    atom_radius = np.array(atom_radius)
    density_pred = np.zeros_like(density)
    d_density_pred = np.zeros_like(density)

    # minimize loss by gradient descent
    loss = np.inf
    i = 0
    while True:

        # L2 loss between predicted and true density
        density_pred[...] = 0.0
        for j in range(n_atoms):
            density_pred[:,c[j]] += \
                get_atom_density(xyz[j], atom_radius[c[j]], points, radius_multiple)
        d_density_pred[...] = density_pred - density
        L2 = np.sum(d_density_pred**2)/2

        # interatomic energy of predicted atom positions
        E = 0.0
        if lambda_E:
            for j in range(n_atoms): # assumes use_covalent_radius
                bond_length = atom_radius[c[j]] + atom_radius[c[j+1:]]
                E += np.sum(get_interatomic_energy(xyz[j], xyz[j+1:], bond_length, bonds[j,j+1:]))

        loss_prev = loss
        loss = L2 + lambda_E*E
        delta_loss = loss - loss_prev
        if verbose > 2:
            print('iteration = {}, L2 = {}, E = {}, loss = {} ({})' \
                  .format(i, L2, E, loss, delta_loss), file=sys.stderr)

        if not n_atoms or delta_loss > -1e-3 or i == max_iter:
            break

        # compute derivatives and descend loss gradient
        d_xyz_prev[...] = d_xyz
        d_xyz[...] = 0.0

        for j in range(n_atoms):
            gradient = get_atom_gradient(xyz[j], atom_radius[c[j]], points, radius_multiple)
            d_xyz[j] += np.sum(d_density_pred[:,c[j],np.newaxis]*gradient, axis=0)

        if lambda_E:
            for j in range(n_atoms-1):
                bond_length = atom_radius[c[j]] + atom_radius[c[j+1:]]
                forces = get_interatomic_forces(xyz[j], xyz[j+1:], bond_length, bonds[j,j+1:])
                d_xyz[j] += lambda_E*np.sum(forces, axis=0)
                d_xyz[j+1:,:] -= lambda_E*forces

        xyz[...] -= lr*(mo*d_xyz_prev + (1-mo)*d_xyz)
        i += 1

    return xyz, density_pred, loss


def wiener_deconv_grid(grid, center, resolution, atom_radius, radius_multiple, noise_ratio=0.0):
    '''
    Applies a convolution to the input grid that approximates the inverse
    of the operation that converts a set of atom positions to a grid of
    atom density.
    '''
    points, _ = grid_to_points_and_values(grid, center, resolution)
    h = get_atom_density(center+resolution/2, atom_radius, points, radius_multiple).reshape(grid.shape)
    h = np.roll(h, shift=(12,12,12), axis=(0,1,2)) # center at origin
    # we want a convolution g such that g * grid = a, where a is the atom positions
    # we assume that grid = h * a, so g is the inverse of h: g * (h * a) = a
    # take F() to be the Fourier transform, F-1() the inverse Fourier transform
    # convolution theorem: g * grid = F-1(F(g)F(grid))
    # Wiener deconvolution: F(g) = 1/F(h) |F(h)|^2 / (|F(h)|^2 + noise_ratio)
    F_h = np.fft.fftn(h) 
    F_grid = np.fft.fftn(grid)
    conj_F_h = np.conj(F_h)
    F_g = conj_F_h / (F_h*conj_F_h + noise_ratio)
    return np.real(np.fft.ifftn(F_grid * F_g))


def wiener_deconv_grids(grids, channels, center, resolution, radius_multiple, noise_ratio=0.0):
    deconv_grids = []
    for grid, (_, _, atom_radius) in zip(grids, channels):
        deconv_grid = wiener_deconv_grid(grid, center, resolution, atom_radius, radius_multiple, noise_ratio)
        deconv_grids.append(deconv_grid)
    return np.stack(deconv_grids, axis=0)


def get_grid_points(shape, center, resolution):
    '''
    Return an array of grid points with a certain
    shape.
    '''
    shape = np.array(shape)
    center = np.array(center)
    resolution = np.array(resolution)
    origin = center - resolution*(shape - 1)/2.0
    indices = np.array(list(np.ndindex(*shape)))
    return origin + resolution*indices


def grid_to_points_and_values(grid, center, resolution):
    '''
    Convert a grid with a center and resolution to lists
    of grid points and values at each point.
    '''
    points = get_grid_points(grid.shape, center, resolution)
    return points, grid.flatten()


def fit_atoms_to_grids(grids, channels, center, resolution, max_iter, lambda_E, radius_multiple,
                       deconv_fit=False, noise_ratio=0.0, greedy=False, bonded=False, verbose=0,
                       all_iters=False):
    '''
    Fit atoms to grids by iteratively placing atoms and then optimizing their
    positions by gradient descent on L2 loss between the true grid density and
    predicted density associated with the atoms until L2 loss stops improving.
    '''
    n_channels, grid_shape = grids.shape[0], grids.shape[1:]
    atom_radius = np.array(zip(*channels)[2])

    # convert grids to arrays of xyz points and channel density values
    points = get_grid_points(grid_shape, center, resolution)
    density = grids.reshape((n_channels, -1)).T

    # iteratively add atoms, fit, and assess goodness-of-fit
    xyz_init = np.ndarray((0, 3))
    c = np.ndarray(0, dtype=int)
    bonds = np.ndarray((0, 0))
    loss_best = np.inf
    if all_iters:
        all_xyz = []
        all_c = []
        all_bonds = []
    while True:

        # optimize atom positions by gradient descent
        xyz, density_pred, loss = \
            fit_atoms_by_GD(points, density, xyz_init, c, bonds, atom_radius, radius_multiple,
                            max_iter, lambda_E=lambda_E, verbose=verbose)
        if verbose > 1:
            added_str = 'added ' + channels[c[-1]][1] if len(xyz) > 0 else ''
            print('n_atoms = {}, loss = {:f}, {}'.format(len(xyz), loss, added_str), file=sys.stderr)

        # stop if fit gets worse (loss increases)
        if loss > loss_best:
            break
        else:
            xyz_best = xyz
            c_best = c
            bonds_best = bonds
            loss_best = loss
            if all_iters:
                all_xyz.append(xyz_best)
                all_c.append(c_best)
                all_bonds.append(bonds)
            if greedy:
                xyz_init[...] = xyz_best

        density_diff = density - density_pred
        if deconv_fit:
            grids_diff = density_diff.T.reshape((n_channels,) + grid_shape)
            grids_deconv = wiener_deconv_grids(grids_diff, channels, center, resolution, \
                                               radius_multiple, noise_ratio)
            density_deconv = grids_deconv.reshape((n_channels, -1)).T
            density_diff = density_deconv

        # try adding an atom to remaining density
        xyz_new, c_new, bonds_new = \
            get_next_atom(points, density, xyz_init, c, atom_radius, bonded)

        if xyz_new is not None:
            xyz_init = np.append(xyz_init, xyz_new[np.newaxis,:], axis=0)
            c = np.append(c, c_new)
            bonds = np.append(bonds, bonds_new[np.newaxis,:], axis=0)
            bonds_new = np.append(bonds_new, 0) # no self bond
            bonds = np.append(bonds, bonds_new[:,np.newaxis], axis=1)
        else:
            break

    if all_iters:
        return all_xyz, all_c, all_bonds, loss_best
    else:
        return xyz_best, c_best, bonds_best, loss_best


def get_next_atom(points, density, xyz_init, c, atom_radius, bonded):
    '''
    Get next atom tuple (xyz_new, c_new, bonds_new) of initial position,
    channel index, and bonds to other atoms. Select the atom as maximum
    density point within some distance range from the other atoms, given
    by positions xyz_init and channel indices c.
    '''
    xyz_new = None
    c_new = None
    bonds_new = None
    d_max = 0.0

    if bonded:
        # bond_length2[i,j] = length^2 of bond between channel[i] and channel[j]
        bond_length2 = (atom_radius[:,np.newaxis] + atom_radius[np.newaxis,:])**2

    for p, d in zip(points, density):

        # more_density[i] = p has more density in channel[i] than best point so far
        more_density = d > d_max
        if np.any(more_density):

            if len(xyz_init) == 0:
                xyz_new = p
                c_new = np.argmax(d)
                bonds_new = np.array([])
                d_max = d[c_new]

            else:
                # dist2[i] = distance^2 between p and xyz_init[i]
                dist2 = np.sum((p[np.newaxis,:] - xyz_init)**2, axis=1)

                # dist_min2[i,j] = min distance^2 between p and xyz_init[i] in channel[j]
                # dist_max2[i,j] = max distance^2 between p and xyz_init[i] in channel[j]
                if bonded:
                    dist_min2 = 0.75*bond_length2[c]
                    dist_max2 = 1.25*bond_length2[c]
                else:
                    dist_min2 = atom_radius[c,np.newaxis]
                    dist_max2 = np.full_like(dist_min2, np.inf)

                # far_enough[i,j] = p is far enough from xyz_init[i] in channel[j]
                # near_enough[i,j] = p is near enough to xyz_init[i] in channel[j]
                far_enough = dist2[:,np.newaxis] > dist_min2
                near_enough = dist2[:,np.newaxis] < dist_max2

                # in_range[i] = p is far enough from all xyz_init and near_enough to
                # some xyz_init to make a bond in channel[i]
                in_range = np.all(far_enough, axis=0) & np.any(near_enough, axis=0)
                if np.any(in_range & more_density):
                    xyz_new = p
                    c_new = np.argmax(in_range*more_density*d)
                    if bonded:
                        bonds_new = near_enough[:,c_new]
                    else:
                        bonds_new = np.zeros(len(xyz_init))
                    d_max = d[c_new]

    return xyz_new, c_new, bonds_new


def rec_and_lig_at_index_in_data_file(file, index):
    '''
    Read receptor and ligand names at a specific line number in a data file.
    '''
    with open(file, 'r') as f:
        line = f.readlines()[index]
    cols = line.rstrip().split()
    return cols[2], cols[3]


def best_loss_batch_index_from_net(net, loss_name, n_batches, best):
    '''
    Return the index of the batch that has the best loss out of
    n_batches forward passes of a net.
    '''
    loss = net.blobs[loss_name]
    best_index, best_loss = -1, None
    for i in range(n_batches):
        net.forward()
        l = float(np.max(loss.data))
        if i == 0 or best(l, best_loss) == l:
            best_loss = l
            best_index = i
            print('{} ({} / {})'.format(best_loss, i, n_batches), file=sys.stderr)
    return best_index


def n_lines_in_file(file):
    '''
    Count the number of lines in a file.
    '''
    with open(file, 'r') as f:
        return sum(1 for line in f)


def best_loss_rec_and_lig(model_file, weights_file, data_file, data_root, loss_name, best=max):
    '''
    Return the names of the receptor and ligand that have the best loss
    using a provided model, weights, and data file.
    '''
    n_batches = n_lines_in_file(data_file)
    with instantiate_model(model_file, data_file, data_file, data_root, 1) as model_file:
        net = caffe.Net(model_file, weights_file, caffe.TEST)
        index = best_loss_batch_index_from_net(net, loss_name, n_batches, best)
    return rec_and_lig_at_index_in_data_file(data_file, index)


def find_blobs_in_net(net, blob_pattern):
    '''
    Return a list of blobs in a net whose names match a regex pattern.
    '''
    blobs_found = []
    for blob_name, blob in net.blobs.items():
        if re.match('(?:' + blob_pattern + r')\Z', blob_name): # match full string
            blobs_found.append(blob)
    return blobs_found


def generate_grids_from_net(net, blob_pattern, n_grids=np.inf, lig_gen_mode='', diff_rec=False):
    '''
    Generate grids from a specific blob in net.
    '''
    assert lig_gen_mode in {'', 'unit', 'mean', 'zero'}
    blob = find_blobs_in_net(net, blob_pattern)[-1]
    batch_size = blob.shape[0]

    i = 0
    while i < n_grids:

        if (i % batch_size) == 0: # forward next batch up to latent vectors

            if diff_rec or lig_gen_mode:

                net.forward(end='latent_concat')

                if diff_rec: # roll rec latent vectors along batch axis by 1
                    net.blobs['rec_latent_fc'].data[...] = \
                        np.roll(net.blobs['rec_latent_fc'].data, shift=1, axis=0)

                # set lig_gen_mode parameters if necessary
                if lig_gen_mode == 'unit':
                    net.blobs['lig_latent_mean'].data[...] = 0.0
                    net.blobs['lig_latent_std'].data[...] = 1.0

                elif lig_gen_mode == 'mean':
                    net.blobs['lig_latent_std'].data[...] = 0.0

                elif lig_gen_mode == 'zero':
                    net.blobs['lig_latent_mean'].data[...] = 0.0
                    net.blobs['lig_latent_std'].data[...] = 0.0

                # forward from lig latent noise to output
                net.forward(start='lig_latent_noise')

            else:
                net.forward()

        yield blob.data[(i % batch_size)]
        i += 1


def combine_element_grids_and_channels(grids, channels):
    '''
    Return new lists of grids and channels by combining grids and channels
    that have the same element.
    '''
    elem_map = dict()
    elem_grids = []
    elem_channels = []
    for grid, (_, element, atom_radius) in zip(grids, channels):
        if element not in elem_map:
            elem_map[element] = len(elem_map)
            elem_grids.append(np.zeros_like(grid))
            elem_channels.append((element, element, atom_radius))
        elem_grids[elem_map[element]] += grid
    return np.array(elem_grids), elem_channels


def write_pymol_script(pymol_file, dx_files, rec_file, lig_file, fit_file=None):
    '''
    Write a pymol script with a map object for each of dx_files, a
    group of all map objects (if any), a rec_file, a lig_file, and
    an optional fit_file.
    '''
    with open(pymol_file, 'w') as out:
        map_objects = []
        for dx_file in dx_files:
            map_object = dx_file.replace('.dx', '_grid')
            out.write('load {}, {}\n'.format(dx_file, map_object))
            map_objects.append(map_object)
        if map_objects:
            map_group = pymol_file.replace('.pymol', '_grids')
            out.write('group {}, {}\n'.format(map_group, ' '.join(map_objects)))
        out.write('load {}\n'.format(rec_file))
        out.write('load {}\n'.format(lig_file))
        if fit_file:
            out.write('load {}\n'.format(fit_file))


def ob_mol_from_xyz_elems_bonds(xyz, elems, bonds):
    '''
    Return an OpenBabel molecule from an array of
    xyz atom positions and associated elements.
    '''
    table = ob.OBElementTable()
    mol = ob.OBMol()
    n_atoms = len(xyz)
    for (x, y, z), element in zip(xyz, elems):
        atom = mol.NewAtom()
        atom.SetAtomicNum(table.GetAtomicNum(element))
        atom.SetVector(x, y, z)
    n_bonds = 0
    for i in range(n_atoms):
        atom_i = mol.GetAtom(i)
        for j in range(i+1, n_atoms):
            atom_j = mol.GetAtom(j)
            if bonds[i,j]:
                bond = mol.NewBond()
                bond.Set(n_bonds, atom_i, atom_j, 1, 0)
                n_bonds += 1
    return mol


def write_ob_mols_to_sdf_file(sdf_file, mols):
    conv = ob.OBConversion()
    conv.SetOutFormat('sdf')
    for i, mol in enumerate(mols):
        conv.WriteFile(mol, sdf_file) if i == 0 else conv.Write(mol)
    conv.CloseOutFile()


def write_xyz_elems_bonds_to_sdf_file(sdf_file, xyz_elems_bonds):
    '''
    Write tuples of (xyz, elemes, bonds) atom positions and
    corresponding elements and bond matrix as chemical structures
    in an .sdf file.
    '''
    out = open(sdf_file, 'w')
    for xyz, elems, bonds in xyz_elems_bonds:
        out.write('\n mattragoza\n\n')
        n_atoms = xyz.shape[0]
        n_bonds = np.sum(bonds.astype(bool))
        out.write('{:3d}'.format(n_atoms))
        out.write('{:3d}'.format(n_bonds))
        out.write('  0  0  0  0  0  0  0  0')
        out.write('999 V2000\n')
        for (x, y, z), element in zip(xyz, elems):
            out.write('{:10.4f}'.format(x))
            out.write('{:10.4f}'.format(y))
            out.write('{:10.4f}'.format(z))
            out.write(' {:3}'.format(element))
            out.write(' 0  0  0  0  0  0  0  0  0  0  0  0\n')
        for i in range(n_atoms):
            for j in range(n_atoms):
                if bonds[i,j]:
                    out.write('{:3d}'.format(i+1))
                    out.write('{:3d}'.format(j+1))
                    out.write('  1  0  0  0\n')
        out.write('M  END\n')
        out.write('$$$$\n')
    out.close()


def write_grid_to_dx_file(dx_file, grid, center, resolution):
    '''
    Write a grid with a center and resolution to a .dx file.
    '''
    dim = grid.shape[0]
    origin = np.array(center) - resolution*(dim-1)/2.
    with open(dx_file, 'w') as f:
        f.write('object 1 class gridpositions counts %d %d %d\n' % (dim, dim, dim))
        f.write('origin %.5f %.5f %.5f\n' % tuple(origin))
        f.write('delta %.5f 0 0\n' % resolution)
        f.write('delta 0 %.5f 0\n' % resolution)
        f.write('delta 0 0 %.5f\n' % resolution)
        f.write('object 2 class gridconnections counts %d %d %d\n' % (dim, dim, dim))
        f.write('object 3 class array type double rank 0 items [ %d ] data follows\n' % (dim**3))
        total = 0
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    f.write('%.10f' % grid[i][j][k])
                    total += 1
                    if total % 3 == 0:
                        f.write('\n')
                    else:
                        f.write(' ')


def write_grids_to_dx_files(out_prefix, grids, channels, center, resolution):
    '''
    Write each of a list of grids a separate .dx file, using the channel names.
    '''
    dx_files = []
    for grid, (channel_name, _, _) in zip(grids, channels):
        dx_file = '{}_{}.dx'.format(out_prefix, channel_name)
        write_grid_to_dx_file(dx_file, grid, center, resolution)
        dx_files.append(dx_file)
    return dx_files


def get_mols_from_sdf_file(sdf_file):
    '''
    Read the molecules from an .sdf file.
    '''
    return Chem.rdmolfiles.SDMolSupplier(sdf_file)


def get_center_from_sdf_file(sdf_file, idx=0):
    '''
    Compute the center of a molecule in an .sdf file
    by taking the mean of the non-hydrogen atom positions.
    '''
    mol = get_mols_from_sdf_file(sdf_file)[idx]
    xyz = Chem.RemoveHs(mol).GetConformer().GetPositions()
    return xyz.mean(axis=0)


def get_n_atoms_from_sdf_file(sdf_file, idx=0):
    '''
    Count the number of atoms of each element in a molecule 
    from an .sdf file.
    '''
    mol = get_mols_from_sdf_file(sdf_file)[idx]
    return Counter(atom.GetSymbol() for atom in mol.GetAtoms())


def get_sdf_file_and_idx(gninatypes_file):
    '''
    Get the name of the .sdf file and conformer idx that a
    .gninatypes file was created from.
    '''
    m = re.match(r'.*_ligand_(\d+)\.gninatypes', gninatypes_file)
    if m:
        idx = int(m.group(1))
        from_str = r'_ligand_{}\.gninatypes$'.format(idx)
        to_str = '_docked.sdf'
    else:
        idx = 0
        m = re.match(r'.*_(.+)\.gninatypes$', gninatypes_file)
        from_str = r'_{}\.gninatypes'.format(m.group(1))
        to_str = '_{}.sdf'.format(m.group(1))
    sdf_file = re.sub(from_str, to_str, gninatypes_file)
    return sdf_file, idx
        

def write_examples_to_data_file(data_file, examples):
    '''
    Write (rec_file, lig_file) examples to data_file.
    '''
    with open(data_file, 'w') as f:
        for rec_file, lig_file in examples:
            f.write('0 0 {} {}\n'.format(rec_file, lig_file))
    return data_file


def get_temp_data_file(examples):
    '''
    Write (rec_file, lig_file) examples to a temporary
    data file and return the path to the file.
    '''
    _, data_file = tempfile.mkstemp()
    write_examples_to_data_file(data_file, examples)
    return data_file


def get_examples_from_data_file(data_file, data_root=''):
    '''
    Iterate through (rec_file, lig_file) examples from
    data_file, optionally prepended with data_root.
    '''
    with open(data_file, 'r') as f:
        for line in f:
            rec_file, lig_file = line.rstrip().split()[2:4]
            if data_root:
                rec_file = os.path.join(data_root, rec_file)
                lig_file = os.path.join(data_root, lig_file)
            yield rec_file, lig_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_file', required=True, help='Generative model prototxt file')
    parser.add_argument('-w', '--weights_file', default=None, help='Generative model weights file')
    parser.add_argument('-b', '--blob_name', required=True, help='Name of blob in model to generate/fit')
    parser.add_argument('-r', '--rec_file', default=[], action='append', help='Receptor file (relative to data_root)')
    parser.add_argument('-l', '--lig_file', default=[], action='append', help='Ligand file (relative to data_root)')
    parser.add_argument('--data_file', default='', help='Path to data file (generate for every line)')
    parser.add_argument('--data_root', default='', help='Path to root for receptor and ligand files')
    parser.add_argument('-o', '--out_prefix', required=True, help='Common prefix for output files')
    parser.add_argument('--output_dx', action='store_true', help='Output .dx files of atom density grids for each channel')
    parser.add_argument('--fit_atoms', action='store_true', help='Fit atoms to density grids and print the goodness-of-fit')
    parser.add_argument('--output_sdf', action='store_true', help='Output .sdf file of fit atom positions')
    parser.add_argument('--max_iter', type=int, default=np.inf, help='Maximum number of iterations for atom fitting')
    parser.add_argument('--lambda_E', type=float, default=0.0, help='Interatomic energy loss weight for gradient descent atom fitting')
    parser.add_argument('--fine_tune', action='store_true', help='Fine-tune final fit atom positions to summed grid channels')
    parser.add_argument('--fit_GMM', action='store_true', help='Fit atoms by a Gaussian mixture model instead of gradient descent')
    parser.add_argument('--noise_model', default='', help='Noise model for GMM atom fitting (d|p)')
    parser.add_argument('--combine_channels', action='store_true', help="Combine channels with same element for atom fitting")
    parser.add_argument('--lig_gen_mode', default='', help='Alternate ligand generation (|mean|unit|zero)')
    parser.add_argument('-r2', '--rec_file2', default='', help='Alternate receptor file (for receptor latent space)')
    parser.add_argument('-l2', '--lig_file2', default='', help='Alternate ligand file (for receptor latent space)')
    parser.add_argument('--deconv_grids', action='store_true', help="Apply Wiener deconvolution to atom density grids")
    parser.add_argument('--scale_grids', type=float, default=1.0, help='Factor by which to scale atom density grids')
    parser.add_argument('--deconv_fit', action='store_true', help="Apply Wiener deconvolution for atom fitting initialization")
    parser.add_argument('--noise_ratio', default=1.0, type=float, help="Noise-to-signal ratio for Wiener deconvolution")
    parser.add_argument('--greedy', action='store_true', help="Initialize atoms to optimized positions when atom fitting")
    parser.add_argument('--bonded', action='store_true', help="Add next atoms within bonding distance when atom fitting")
    parser.add_argument('--verbose', default=0, type=int, help="Verbose output level")
    parser.add_argument('--all_iters_sdf', action='store_true', help="Output a .sdf structure for each outer iteration of atom fitting")
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    # read the net parameter file and get values relevant for atom gridding
    net_param = caffe_util.NetParameter.from_prototxt(args.model_file)
    data_param = net_param.get_molgrid_data_param(caffe.TEST)
    data_param.shuffle = False
    data_param.balanced = False
    resolution = data_param.resolution
    radius_multiple = data_param.radius_multiple
    use_covalent_radius = data_param.use_covalent_radius

    if not args.data_file: # use the set of (rec_file, lig_file) examples
        assert len(args.rec_file) == len(args.lig_file)
        data_file = get_temp_data_file(zip(args.rec_file, args.lig_file))

    else: # use the examples in the provided data_file
        assert len(args.rec_file) == len(args.lig_file) == 0
        data_file = args.data_file

    # create the net in caffe
    net_param.set_molgrid_data_source(data_file, args.data_root, caffe.TEST)
    net = caffe_util.Net.from_param(net_param, args.weights_file, caffe.TEST)

    channels = None
    examples = get_examples_from_data_file(data_file, args.data_root)
    grids_generator = generate_grids_from_net(net, args.blob_name, lig_gen_mode=args.lig_gen_mode)

    if args.fit_atoms:
        out_file = '{}.fit_output'.format(args.out_prefix)
        out = open(out_file, 'w')

    for (rec_file, lig_file), grids in izip(examples, grids_generator):

        rec_file = rec_file.replace('.gninatypes', '.pdb')
        lig_file = lig_file.replace('.gninatypes', '.sdf')
        lig_name = os.path.splitext(os.path.basename(lig_file))[0]
        out_prefix = '{}_{}'.format(args.out_prefix, lig_name)

        try:
            center = get_center_from_sdf_file(lig_file)
        except:
            center = np.zeros(3) # TODO use openbabel, this is a hack 

        density_norm2 = np.sum(grids**2)
        density_sum = np.sum(grids)
        density_max = np.max(grids)
        assert grids.sum() > 0

        if not channels: # infer channel info from shape of first grids
            channels = channel_info.get_channels_for_grids(grids, use_covalent_radius)

        if args.combine_channels:
            grids, channels = combine_element_grids_and_channels(grids, channels)

        if args.deconv_grids:
            grids = wiener_deconv_grids(grids, channels, center, resolution, radius_multiple, \
                                        noise_ratio=args.noise_ratio)

        grids *= args.scale_grids

        if args.output_dx:
            dx_files = write_grids_to_dx_files(out_prefix, grids, channels, center, resolution)
        else:
            dx_files = []

        if args.fit_atoms: # fit atoms to density grids

            t_i = time.time()
            xyz, c, bonds, loss = \
                fit_atoms_to_grids(grids, channels,
                                   center=center,
                                   resolution=resolution,
                                   max_iter=args.max_iter,
                                   lambda_E=args.lambda_E,
                                   radius_multiple=radius_multiple,
                                   deconv_fit=args.deconv_fit,
                                   noise_ratio=args.noise_ratio,
                                   greedy=args.greedy,
                                   bonded=args.bonded,
                                   verbose=args.verbose,
                                   all_iters=args.all_iters_sdf)

            delta_t = time.time() - t_i
            out.write('{} {} {}\n'.format(lig_name, loss, delta_t))
            out.flush()

            if args.verbose > 0:
                print('{:20}shape = {}, density_norm2 = {:.5f}, density_sum = {:.5f}, density_max = {:.5f}, loss = {:.5f}' \
                      .format(lig_name, grids.shape, density_norm2, density_sum, density_max, loss), file=sys.stderr)
        else:
            if args.verbose > 0:
                print('{:20}shape = {}, density_norm2 = {:.5f}, density_sum = {:.5f}, density_max = {:.5f}' \
                      .format(lig_name, grids.shape, density_norm2, density_sum, density_max), file=sys.stderr)

        if args.fit_atoms and args.output_sdf:
            get_elements = lambda x: [channels[i][1] for i in x]
            if args.all_iters_sdf:
                xyz_elems_bonds = zip(xyz, map(get_elements, c), bonds)
            else:
                xyz_elems_bonds = [(xyz, get_elements(c), bonds)]

            fit_file = '{}_fit.sdf'.format(out_prefix)
            #mols = [ob_mol_from_xyz_elems_bonds(*args) for args in xyz_elems_bonds]
            #write_ob_mols_to_sdf_file(fit_file, mols)
            write_xyz_elems_bonds_to_sdf_file(fit_file, xyz_elems_bonds)
        else:
            fit_file = None

        pymol_file = '{}.pymol'.format(out_prefix)
        write_pymol_script(pymol_file, dx_files, rec_file, lig_file, fit_file)


if __name__ == '__main__':
    main(sys.argv[1:])
