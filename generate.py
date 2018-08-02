from __future__ import print_function
import sys, os, re, argparse, ast, time
import numpy as np
from rdkit import Chem
from collections import Counter
import contextlib
import tempfile
from multiprocessing.pool import Pool, ThreadPool
from functools import partial
from scipy.stats import multivariate_normal
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

import caffe_util
import cgenerate

BOND_LENGTH_K = 0.8


# channel name, element, atom radius
rec_channels = [("rec_AliphaticCarbonXSHydrophobe", "C", 1.90),
                ("rec_AliphaticCarbonXSNonHydrophobe", "C", 1.90),
                ("rec_AromaticCarbonXSHydrophobe", "C", 1.90),
                ("rec_AromaticCarbonXSNonHydrophobe", "C", 1.90),
                ("rec_Calcium", "Ca", 1.20),
                ("rec_Iron", "Fe", 1.20),
                ("rec_Magnesium", "Mg", 1.20),
                ("rec_Nitrogen", "N", 1.80),
                ("rec_NitrogenXSAcceptor", "N", 1.80),
                ("rec_NitrogenXSDonor", "N", 1.80),
                ("rec_NitrogenXSDonorAcceptor", "N", 1.80),
                ("rec_OxygenXSAcceptor", "O", 1.70),
                ("rec_OxygenXSDonorAcceptor", "O", 1.70),
                ("rec_Phosphorus", "P", 2.10),
                ("rec_Sulfur", "S", 2.00),
                ("rec_Zinc", "Zn", 1.20)]

lig_channels = [("lig_AliphaticCarbonXSHydrophobe", "C", 1.90),
                ("lig_AliphaticCarbonXSNonHydrophobe", "C", 1.90),
                ("lig_AromaticCarbonXSHydrophobe", "C", 1.90),
                ("lig_AromaticCarbonXSNonHydrophobe", "C", 1.90),
                ("lig_Bromine", "Br", 2.00),
                ("lig_Chlorine", "Cl", 1.80),
                ("lig_Fluorine", "F", 1.50),
                ("lig_Nitrogen", "N", 1.80),
                ("lig_NitrogenXSAcceptor", "N", 1.80),
                ("lig_NitrogenXSDonor", "N", 1.80),
                ("lig_NitrogenXSDonorAcceptor", "N", 1.80),
                ("lig_Oxygen", "O", 1.70),
                ("lig_OxygenXSAcceptor", "O", 1.70),
                ("lig_OxygenXSDonorAcceptor", "O", 1.70),
                ("lig_Phosphorus", "P", 2.10),
                ("lig_Sulfur", "S", 2.00),
                ("lig_SulfurAcceptor", "S", 2.00),
                ("lig_Iodine", "I", 2.20),
                ("lig_Boron", "B", 1.92)]


def get_color_for_channel(channel_name):
    '''
    Return the color of a channel by getting the element from its name.
    '''
    if 'Carbon' in channel_name:
        return 'grey'
    elif 'Oxygen' in channel_name:
        return 'red'
    elif 'Nitrogen' in channel_name:
        return 'blue'
    elif 'Phosphorus' in channel_name:
        return 'orange'
    elif 'Sulfur' in channel_name:
        return 'yellow'
    else:
        return 'green'


class GaussianMixtureGridLayer(caffe.Layer):
    '''
    A caffe layer that fits atoms to the input blob and then computes
    an atom density grid from the fit atoms as the output blob.
    '''
    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception('Must have one input.')
        params = ast.literal_eval(self.param_str)
        self.f = partial(fit_grids_to_grids, max_iter=params.get('max_iter', 20),
                         resolution=params.get('resolution', 0.5))

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].shape)
        self.pool = ThreadPool(bottom[0].shape[0])

    def forward(self, bottom, top):
        top[0].data[...] = np.array(self.pool.map(self.f, bottom[0].data))

    def backward(self, top, propagate_down, bottom):
        pass


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


def get_interatomic_energy(atom_pos1, atom_pos2, bond_length, width_factor=1.0):
    '''
    Compute the interatomic potential energy between an atom and a set of atoms.
    '''
    dist = np.sqrt(np.sum((atom_pos2 - atom_pos1)**2, axis=1))
    exp = np.exp(-width_factor*(dist - bond_length))
    return (1 - exp)**2 - 1


def get_interatomic_forces(atom_pos1, atom_pos2, bond_length, width_factor=1.0):
    '''
    Compute the derivative of interatomic potential energy between an atom
    and a set of atoms with respect to the position of the first atom.
    '''
    diff = atom_pos2 - atom_pos1
    dist2 = np.sum(diff**2, axis=1)
    dist = np.sqrt(dist2)
    exp = np.exp(-width_factor*(dist - bond_length))
    d_energy = 2 * (1 - exp) * exp * width_factor
    return -diff * (d_energy / dist)[:,np.newaxis]


def fit_atoms_by_GMM(points, density, xyz_init, atom_radius, radius_multiple, max_iter, 
                     noise_model='', noise_params_init={}, gof_crit='nll', verbose=False):
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
        if verbose:
            print('ITERATION {} | nll = {} ({})'.format(i, -ll, -(ll - ll_prev)), file=sys.stderr)

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


def fit_atoms_by_GD(points, density, xyz_init, atom_radius, radius_multiple,
                    max_iter, lr=0.01, mo=0.9, lambda_E=0.0, verbose=False):
    '''
    Fit atom positions to a set of points with the given density values by
    minimizing the L2 loss (and interatomic energy) by gradient descent with
    momentum. Return the final atom positions and loss.
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

        # L2 loss is the squared L2 norm of the difference between predicted and true density
        density_pred[...] = 0.0
        for j in range(n_atoms):
            density_pred += get_atom_density(xyz[j], atom_radius[j], points, radius_multiple)
        d_density_pred[...] = density_pred - density
        L2 = np.sum(d_density_pred**2)/2

        # interatomic energy of predicted atom positions
        E = 0.0
        if lambda_E:
            for j in range(n_atoms):
                bond_length = BOND_LENGTH_K * (atom_radius[j] + atom_radius[j+1:])/2.0
                E += 2*np.sum(get_interatomic_energy(xyz[j], xyz[j+1:,:], bond_length))

        loss_prev, loss = loss, L2 + lambda_E*E
        delta_loss = loss - loss_prev
        if verbose:
            if lambda_E:
                print('ITERATION {} | L2 = {}, E = {}, loss = {} ({})'.format(i, L2, E, loss, delta_loss), file=sys.stderr)
            else:
                print('ITERATION {} | L2 = {} ({})'.format(i, loss, delta_loss), file=sys.stderr)
        if delta_loss > -1e-3 or i == max_iter:
            break

        # compute derivatives and descend loss gradient
        d_xyz_prev[...] = d_xyz
        d_xyz[...] = 0.0

        for j in range(n_atoms):
            d_xyz[j] += np.sum(d_density_pred[:,np.newaxis] * \
                get_atom_gradient(xyz[j], atom_radius[j], points, radius_multiple), axis=0)

        if lambda_E:
            for j in range(n_atoms-1):
                bond_length = BOND_LENGTH_K * (atom_radius[j] + atom_radius[j+1:])/2.0
                forces = get_interatomic_forces(xyz[j], xyz[j+1:,:], bond_length)
                d_xyz[j] += lambda_E * np.sum(forces, axis=0)
                d_xyz[j+1:,:] -= lambda_E * forces

        xyz[...] -= lr*(mo*d_xyz_prev + (1-mo)*d_xyz)
        i += 1

    return xyz, loss


def wiener_deconvolution(grid, center, resolution, atom_radius, radius_multiple, noise_ratio=0.0):
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


def get_max_density_points(points, density, distance):
    '''
    Generate maximum density points that are at least some distance
    apart from each other from a list of points and densities.
    '''
    assert len(points) > 0, 'no points provided'
    distance_check = lambda a, b: np.sum((a - b)**2) > distance**2
    max_points = []
    for p, d in sorted(zip(points, density), key=lambda pd: -pd[1]):
        if all(distance_check(p, max_p) for max_p in max_points):
            max_points.append(p)
            yield max_points[-1]


def grid_to_points_and_values(grid, center, resolution):
    '''
    Convert a grid with a center and resolution to lists
    of grid points and values at each point.
    '''
    dims = np.array(grid.shape)
    origin = np.array(center) - resolution*(dims-1)/2.
    indices = np.array(list(np.ndindex(*dims)))
    return origin + resolution*indices, grid.flatten()


def fit_atoms_to_grid(grid_args, center, resolution, max_iter, lambda_E, fit_GMM, noise_model, gof_criterion,
                      radius_multiple, density_threshold=0.0, deconv_fit=False, noise_ratio=0.0, verbose=False):
    '''
    Fit atom positions to a grid either by gradient descent on L2 loss with an
    optional interatomic energy term or using a Gaussian mixture model with an
    optional noise model.
    '''
    grid, (channel_name, element, atom_radius), n_atoms = grid_args

    # nothing to fit if the whole grid is sub threshold
    if np.max(grid) <= density_threshold:
        return np.ndarray((0, 3)), 0.0

    if verbose:
        print('\nfitting {}'.format(channel_name), file=sys.stderr)

    # convert grid to arrays of xyz points and density values
    points, density = grid_to_points_and_values(grid, center, resolution)

    if fit_GMM: # initialize noise model params

        noise_params_init = dict(mean=np.mean(density), cov=np.cov(density), prob=1.0/len(points))

        if noise_model != 'd': # TODO this breaks d noise model
            # speed up GMM by only fitting points above threshold
            points = points[density > density_threshold,:]
            density = density[density > density_threshold]

    elif noise_model:
        raise NotImplementedError('noise model only allowed for GMM atom fitting')

    # generator for inital atom positions
    if deconv_fit:
        deconv_grid = wiener_deconvolution(grid, center, resolution, atom_radius, radius_multiple, noise_ratio=noise_ratio)
        deconv_density = deconv_grid.flatten()
        max_density_points = get_max_density_points(points, deconv_density, BOND_LENGTH_K*atom_radius)
    else:
        max_density_points = get_max_density_points(points, density, BOND_LENGTH_K*atom_radius)

    if n_atoms is None: # iteratively add atoms, fit, and assess goodness-of-fit

        xyz_init = np.ndarray((0, 3))
        if fit_GMM and not noise_model: # can't fit GMM with 0 atoms and no noise model
            xyz_init = np.append(xyz_init, next(max_density_points)[np.newaxis,:], axis=0)
        n_atoms = len(xyz_init)

        xyz_best, gof_best = [], np.inf
        while True:
            if fit_GMM:
                xyz, gof = fit_atoms_by_GMM(points, density, xyz_init, [atom_radius]*n_atoms, radius_multiple, max_iter,
                                            noise_model, noise_params_init, gof_criterion, verbose=verbose)
            else:
                xyz, gof = fit_atoms_by_GD(points, density, xyz_init, [atom_radius]*n_atoms, radius_multiple, max_iter,
                                           lambda_E=lambda_E, verbose=verbose)

            if verbose:
                print('n_atoms = {}\tgof = {:f}'.format(n_atoms, gof), file=sys.stderr)

            # stop when fit gets worse (gof increases) or there are no more initial atom positions
            if gof > gof_best:
                break
            xyz_best, gof_best = xyz, gof
            try:
                xyz_init = np.append(xyz_init, next(max_density_points)[np.newaxis,:], axis=0)
                n_atoms += 1
            except StopIteration:
                break

    else: # fit an exact number of atoms

        xyz_init = np.ndarray((0, 3))
        while len(xyz_init) < n_atoms:
            xyz_init = np.append(xyz_init, next(max_density_points)[np.newaxis,:], axis=0)

        if fit_GMM:
            xyz, gof = fit_atoms_by_GMM(points, density, xyz_init, [atom_radius]*n_atoms, radius_multiple, max_iter,
                                        noise_model, noise_params_init, gof_criterion, verbose=verbose)
        else:
            xyz, gof = fit_atoms_by_GD(points, density, xyz_init, [atom_radius]*n_atoms, radius_multiple, max_iter,
                                       lambda_E=lambda_E, verbose=verbose)

        if verbose:
            print('n_atoms = {}\tgof = {:f}'.format(n_atoms, gof), file=sys.stderr)

        xyz_best, gof_best = xyz, gof

    return xyz_best, gof_best


def fit_atoms_to_grids(grids, channels, n_atoms, parallel=True, *args, **kwargs):
    '''
    Fit atom positions to lists of grids with corresponding channel info and
    optional numbers of atoms, in parallel by default. Return a list of lists
    of fit atoms positions (one per channel) and the overall goodness-of-fit.
    '''
    grid_args = zip(grids, channels, n_atoms)
    map_func = Pool(processes=len(grid_args)).map if parallel else map
    xyzs, gofs = zip(*map_func(partial(fit_atoms_to_grid, *args, **kwargs), grid_args))
    return xyzs, np.sum(gofs)


def fit_grid_to_grid(grid_args, center, resolution, *args, **kwargs):
    '''
    Fit atom positions to a grid and then recompute a grid from the
    atom positions.
    '''
    xyz = fit_atoms_to_grid(grid_args, center, resolution, *args, **kwargs)
    atom_radius = [grid_args[1][2] for _ in xyz]
    return add_atoms_to_grid(np.zeros_like(grid), xyz, center, resolution, atom_radius)


def fit_grids_to_grids(grids, channels, n_atoms, parallel=True, *args, **kwargs):
    '''
    Fit atom positions to lists of grids with corresponding channel info and
    optional numbers of atoms and then recompute the grids, in parallel by default.
    '''
    grid_args = zip(grids, channels, n_atoms)
    if parallel:
        fmap = Pool(processes=len(grid_args)).map
    else:
        fmap = map
    f = partial(fit_grid_to_grid, *args, **kwargs)
    return np.array(fmap(f, grid_args))


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


def generate_grids_from_net(net, blob_pattern, index=0, lig_gen_mode=None, diff_rec=False):
    '''
    Generate grids from a specific blob in a net at a specific data index.
    '''
    blob = find_blobs_in_net(net, blob_pattern)[-1]
    batch_size = blob.shape[0]
    index += batch_size
    assert lig_gen_mode in {None, 'unit', 'mean', 'zero'}
    while index >= batch_size:
        net.forward(end='latent_concat') # get rec latent and "init" var lig layers
        if diff_rec:
            net.blobs['rec'].data[index%batch_size,...] = \
                net.blobs['rec'].data[(index+1)%batch_size,...]
            net.blobs['rec_latent_fc'].data[index%batch_size,...] = \
                net.blobs['rec_latent_fc'].data[(index+1)%batch_size,...]
        if lig_gen_mode == 'unit':
            net.blobs['lig_latent_mean'].data[...] = 0.0
            net.blobs['lig_latent_std'].data[...] = 1.0
        elif lig_gen_mode == 'mean':
            net.blobs['lig_latent_std'].data[...] = 0.0
        elif lig_gen_mode == 'zero':
            net.blobs['lig_latent_mean'].data[...] = 0.0
            net.blobs['lig_latent_std'].data[...] = 0.0
        net.forward(start='lig_latent_noise') 
        index -= batch_size
    return blob.data[index]


def combine_element_grids_and_channels(grids, channels):
    '''
    Return new lists of grids and channels by combining grids and channels
    that have the same element.
    '''
    elem_map = dict()
    elem_grids = []
    elem_channels = []
    for grid, (channel_name, element, atom_radius) in zip(grids, channels):
        if element not in elem_map:
            elem_map[element] = len(elem_map)
            elem_grids.append(np.zeros_like(grid))
            elem_channels.append((element, element, atom_radius))
        elem_grids[elem_map[element]] += grid
    return np.array(elem_grids), elem_channels


@contextlib.contextmanager
def temp_data_file(examples):
    '''
    Temporarily create a data file with a line for each
    receptor ligand pair in examples.
    '''
    _, data_file = tempfile.mkstemp()
    with open(data_file, 'w') as f:
        for rec_file, lig_file in examples:
            f.write('0 0 {} {}\n'.format(rec_file, lig_file))
    yield data_file
    os.remove(data_file)


def write_pymol_script(pymol_file, dx_files, *extra_files):
    '''
    Write a pymol script with a map object for each of dx_files,
    a group of all map objects, and load some extra_files
    '''
    with open(pymol_file, 'w') as out:
        map_objects = []
        for dx_file in dx_files:
            map_object = dx_file.replace('.dx', '_map')
            out.write('load {}, {}\n'.format(dx_file, map_object))
            map_objects.append(map_object)
        map_group = pymol_file.replace('.pymol', '_maps')
        out.write('group {}, {}\n'.format(map_group, ' '.join(map_objects)))
        for extra_file in extra_files:
            out.write('load {}\n'.format(extra_file))


def write_atoms_to_sdf_file(sdf_file, xyzs, channels):
    '''
    Write a list of lists of atoms corresponding, each corresponding to one
    of the provided channels, as a chemical structure .sdf file.
    '''
    out = open(sdf_file, 'w')
    out.write('\n\n\n')
    n_atoms = sum(len(xyz) for xyz in xyzs)
    out.write('{:3d}'.format(n_atoms))
    out.write('  0  0  0  0  0  0  0  0  0')
    out.write('999 V2000\n')
    for xyz, (_, element, _) in zip(xyzs, channels):
        for x,y,z in xyz:
            out.write('{:10.4f}'.format(x))
            out.write('{:10.4f}'.format(y))
            out.write('{:10.4f}'.format(z))
            out.write(' {:3}'.format(element))
            out.write(' 0  0  0  0  0  0  0  0  0  0  0  0\n')
    out.write('M  END\n')
    out.write('$$$$')
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


def get_channel_info_for_grids(grids):
    '''
    Infer the channel info for a list of grids by the number of grids.
    '''
    n_channels = grids.shape[0]
    if n_channels == len(rec_channels):
        channels = rec_channels
    elif n_channels == len(lig_channels):
        channels = lig_channels
    elif n_channels == len(rec_channels) + len(lig_channels):
        channels = rec_channels + lig_channels
    else:
        raise ValueError('could not infer channel info for grids with {} channels'.format(n_channels))
    return channels


def get_center_from_sdf_file(sdf_file):
    '''
    Compute the center of a chemical structure .sdf file as the mean of the
    non-hydrogen atom positions, using the first conformation.
    '''
    mol = Chem.MolFromMolFile(sdf_file)
    xyz = Chem.RemoveHs(mol).GetConformer().GetPositions()
    return xyz.mean(axis=0)


def get_n_atoms_from_sdf_file(sdf_file):
    '''
    Count the number of atoms of each element in a chemical structure .sdf file.
    '''
    mol = Chem.MolFromMolFile(sdf_file)
    return Counter(a.GetSymbol() for a in mol.GetAtoms())


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_file', required=True, help='Generative model prototxt file')
    parser.add_argument('-w', '--weights_file', default=None, help='Generative model weights file')
    parser.add_argument('-b', '--blob_name', required=True, help='Name of blob in model to generate/fit')
    parser.add_argument('-r', '--rec_file', default='', help='Receptor file (relative to data_root)')
    parser.add_argument('-l', '--lig_file', default='', help='Ligand file (relative to data_root)')
    #parser.add_argument('-f', '--data_file', default='', help='Path to data file (generate for every line)')
    parser.add_argument('-d', '--data_root', default='', help='Path to root for receptor and ligand files')
    parser.add_argument('-o', '--out_prefix', default=None, help='Common prefix for output files')
    parser.add_argument('--output_dx', action='store_true', help='Output .dx files of atom density grids for each channel')
    parser.add_argument('--fit_atoms', action='store_true', help='Fit atoms to density grids and print the goodness-of-fit')
    parser.add_argument('--output_sdf', action='store_true', help='Output .sdf file of fit atom positions')
    parser.add_argument('--max_iter', type=int, default=np.inf, help='Maximum number of iterations for atom fitting')
    parser.add_argument('--lambda_E', type=float, default=0.0, help='Interatomic energy loss weight for gradient descent atom fitting')
    parser.add_argument('--fine_tune', action='store_true', help='Fine-tune final fit atom positions to summed grid channels')
    parser.add_argument('--fit_GMM', action='store_true', help='Fit atoms by a Gaussian mixture model instead of gradient descent')
    parser.add_argument('--noise_model', default='', help='Noise model for GMM atom fitting (d|p)')
    parser.add_argument('--gof_criterion', default='nll', help='Goodness-of-fit criterion for GMM atom fitting (nll|aic|L2)')
    parser.add_argument('--combine_channels', action='store_true', help="Combine channels with same element for atom fitting")
    parser.add_argument('--read_n_atoms', action='store_true', help="Get exact number of atoms to fit from ligand file")
    parser.add_argument('--channel_info', default=None, help='How to interpret grid channels (None|data|rec|lig)')
    parser.add_argument('--lig_gen_mode', default=None, help='Alternate ligand generation (None|mean|unit)')
    parser.add_argument('-r2', '--rec_file2', default='', help='Alternate receptor file (for receptor latent space)')
    parser.add_argument('-l2', '--lig_file2', default='', help='Alternate ligand file (for receptor latent space)')
    parser.add_argument('--deconv_grids', action='store_true', help="Apply Wiener deconvolution to atom density grids")
    parser.add_argument('--scale_grids', type=float, default=1.0, help='Factor by which to scale atom density grids')
    parser.add_argument('--deconv_fit', action='store_true', help="Apply Wiener deconvolution for atom fitting initialization")
    parser.add_argument('--noise_ratio', default=1.0, type=float, help="Noise-to-signal ratio for Wiener deconvolution")
    parser.add_argument('--parallel', action='store_true', help="Fit atoms to each grid channel in parallel")
    parser.add_argument('--verbose', action='store_true', help="Verbose output for atom fitting")
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    rec_file = os.path.join(args.data_root, args.rec_file)
    lig_file = os.path.join(args.data_root, args.lig_file)
    center = get_center_from_sdf_file(lig_file)
    data_examples = [[rec_file, lig_file]]

    if args.rec_file2 and args.lig_file2:
        rec_file2 = os.path.join(args.data_root, args.rec_file2)
        lig_file2 = os.path.join(args.data_root, args.lig_file2)
        center2 = get_center_from_sdf_file(lig_file2)
        data_examples.append([rec_file2, lig_file2])

    net_param = caffe_util.NetParameter.from_prototxt(args.model_file)
    data_param = net_param.get_molgrid_data_param(caffe.TEST)
    resolution = data_param.resolution
    radius_multiple = data_param.radius_multiple
    with temp_data_file(data_examples) as data_file:
        net_param.set_molgrid_data_source(data_file, '', caffe.TEST)
        net = caffe_util.Net.from_param(net_param, args.weights_file, caffe.TEST)
    grids = generate_grids_from_net(net, args.blob_name, 0, args.lig_gen_mode, len(data_examples) > 1)

    if args.verbose:
        print('shape = {}\ndensity_norm = {}'.format(grids.shape, np.sum(grids**2)**0.5), file=sys.stderr)
    assert np.sum(grids) > 0

    if args.channel_info is None:
        channels = get_channel_info_for_grids(grids)
    elif args.channel_info == 'data':
        channels = rec_channels + lig_channels
    elif args.channel_info == 'rec':
        channels = rec_channels
    elif args.channel_info == 'lig':
        channels = lig_channels
    else:
        raise ValueError('--channel_info must be data, rec, or lig')

    if args.deconv_grids:
        grids = np.stack([wiener_deconvolution(grid, center, resolution, atom_radius, radius_multiple, noise_ratio=args.noise_ratio) \
                    for grid, (_, _, atom_radius) in zip(grids, channels)], axis=0)
    grids *= args.scale_grids

    if args.combine_channels:
        grids, channels = combine_element_grids_and_channels(grids, channels)

    dx_files = []
    if args.output_dx:
        dx_files += write_grids_to_dx_files(args.out_prefix, grids, channels, center, resolution)

    extra_files = [f for example in data_examples for f in example]
    if args.fit_atoms:

        if args.read_n_atoms:
            n_atoms = get_n_atoms_from_sdf_file(lig_file)
        else:
            n_atoms = [None for g in grids]

        # fit atoms to each grid channel separately
        xyzs, loss = fit_atoms_to_grids(grids, channels, n_atoms,
                                        center=center,
                                        resolution=resolution,
                                        max_iter=args.max_iter,
                                        lambda_E=args.lambda_E,
                                        fit_GMM=args.fit_GMM,
                                        noise_model=args.noise_model,
                                        gof_criterion=args.gof_criterion,
                                        radius_multiple=radius_multiple,
                                        deconv_fit=args.deconv_fit,
                                        noise_ratio=args.noise_ratio,
                                        parallel=args.parallel,
                                        verbose=args.verbose)

        # fine-tune atoms by fitting to summed grid channels
        if args.fine_tune:
            chan_map = [i for i, xyz in enumerate(xyzs) for _ in xyz]
            points, density = grid_to_points_and_values(np.sum(grids, axis=0), center, resolution)
            all_xyz, _ = fit_atoms_by_GD(points, density,
                                         xyz_init=np.concatenate(xyzs, axis=0),
                                         atom_radius=[channels[i][2] for i in chan_map],
                                         radius_multiple=radius_multiple,
                                         max_iter=args.max_iter,
                                         lambda_E=args.lambda_E)
            xyzs = [[] for _ in channels]
            for i, (x,y,z) in zip(chan_map, all_xyz):
                xyzs[i].append((x,y,z))

        loss = 0.0
        for xyz, grid, (_, _, atom_radius) in zip(xyzs, grids, channels):
            points, density = grid_to_points_and_values(grid, center, resolution)
            density_pred = np.zeros_like(density)
            for i in range(len(xyz)):
                density_pred += get_atom_density(xyz[i], atom_radius, points, radius_multiple)
            loss += np.sum((density_pred - density)**2)/2.0
        print(loss)

        if args.output_sdf:
            pred_file = '{}_fit.sdf'.format(args.out_prefix)
            write_atoms_to_sdf_file(pred_file, xyzs, channels)
            extra_files.append(pred_file)

    pymol_file = '{}.pymol'.format(args.out_prefix)
    write_pymol_script(pymol_file, dx_files, *extra_files)


if __name__ == '__main__':
    main(sys.argv[1:])
