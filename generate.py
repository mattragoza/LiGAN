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


def get_atom_density(atom_pos, atom_radius, grid_point):
    diff = grid_point - atom_pos
    dist2 = np.sum(diff**2)
    dist = np.sqrt(dist2)
    if dist >= 1.5 * atom_radius:
        return 0.0
    elif dist <= atom_radius:
        h = 0.5 * atom_radius
        ex = -dist2 / (2*h*h)
        return np.exp(ex)
    else:
        h = 0.5 * atom_radius
        inv_e2 = np.exp(-2)
        q = dist2*inv_e2/(h**2) - 6*dist*inv_e2/h + 9*inv_e2
        return q


def get_atom_gradient(atom_pos, atom_radius, point):
    diff = point - atom_pos
    dist2 = np.sum(diff**2)
    dist = np.sqrt(dist2)
    if dist >= 1.5 * atom_radius or np.isclose(dist, 0):
        return 0.0
    elif dist <= atom_radius:
        h = 0.5 * atom_radius
        ex = -dist2 / (2*h*h)
        coef = -dist / (h*h)
        d_density = coef * np.exp(ex)
    else:
        h = 0.5 * atom_radius
        inv_e2 = np.exp(-2)
        d_density = 2*dist*inv_e2/(h**2) - 6*inv_e2/h

    return -(diff / dist) * d_density


def add_atoms_to_grid(grid, atoms, center, resolution, atom_radius):
    dims = np.array(grid.shape)
    origin = np.array(center) - resolution*(dims-1)/2.
    for atom_pos in atoms:
        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    grid_point = origin + resolution*np.array([i,j,k])
                    density = get_atom_density(atom_pos, atom_radius, grid_point)
                    grid[i][j][k] += density
    return grid


def fit_atoms_gmm(points, density, atom_mean_init, atom_radius,
                  noise_type, noise_params_init, max_iter):
    n_points = len(points)
    n_atoms = len(atom_mean_init)

    # initialize component parameters
    atom_mean = np.array(atom_mean_init)
    atom_cov = (0.5*atom_radius)**2
    n_params = atom_mean.size
    if noise_type == 'd':
        noise_mean = noise_params_init['mean']
        noise_cov = noise_params_init['cov']
        n_params += 2
    elif noise_type == 'p':
        noise_prob = noise_params_init['prob']
        n_params += 1
    elif len(noise_type) > 0:
        raise TypeError("noise_type must be 'd' or 'p', or '', got {}".format(repr(noise_type)))
    n_comps = n_atoms + bool(noise_type)
    assert n_comps > 0

    # initialize prior over components
    P_comp = np.full(n_comps, 1.0/n_comps) # P(comp_j)
    n_params += n_comps - 1

    # maximize expected log likelihood
    ll = -np.inf
    for i in range(max_iter+1):

        L_point = np.zeros((n_points, n_comps)) # P(point_i|comp_j)
        for j in range(n_atoms):
            L_point[:,j] = multivariate_normal.pdf(points, mean=atom_mean[j], cov=atom_cov)
        if noise_type == 'd':
            L_point[:,-1] = multivariate_normal.pdf(density, mean=noise_mean, cov=noise_cov)
        elif noise_type == 'p':
            L_point[:,-1] = noise_prob

        P_joint = P_comp * L_point          # P(point_i, comp_j)
        P_point = np.sum(P_joint, axis=1)   # P(point_i)
        gamma = (P_joint.T / P_point).T     # P(comp_j|point_i) (E-step)

        # compute expected log likelihood
        ll_prev, ll = ll, np.sum(density * np.log(P_point))
        if ll - ll_prev < 1e-8 or i == max_iter:
            break

        # estimate parameters that maximize expected log likelihood (M-step)
        for j in range(n_atoms):
            atom_mean[j] = np.sum(density * gamma[:,j] * points.T, axis=1) \
                         / np.sum(density * gamma[:,j])
        if noise_type == 'd':
            noise_mean = np.sum(gamma[:,-1] * density) / np.sum(gamma[:,-1])
            noise_cov = np.sum(gamma[:,-1] * (density - noise_mean)**2) / np.sum(gamma[:,-1])
            if noise_cov == 0.0 or np.isnan(noise_cov): # reset noise
                noise_mean = noise_params_init['mean']
                noise_cov = noise_params_init['cov']
        elif noise_type == 'p':
            noise_prob = noise_prob
        if noise_type and n_atoms > 0:
            P_comp[-1] = np.sum(density * gamma[:,-1]) / np.sum(density)
            P_comp[:-1] = (1.0 - P_comp[-1])/n_atoms

    return atom_mean, 2*ll - 2*n_params


def fit_atoms_l2(grid, center, resolution, atom_mean_init, atom_radius, max_iter, lr=0.01):

    # initialize component parameters
    n_atoms = len(atom_mean_init)
    atom_mean = np.array(atom_mean_init)
    d_atom_mean = np.zeros_like(atom_mean)

    # minimize L2 loss by gradient descent
    l2 = np.inf
    for i in range(max_iter+1):

        grid_pred = add_atoms_to_grid(np.zeros_like(grid), atom_mean, center, resolution, atom_radius)
        d_grid_pred = grid_pred - grid
        l2_prev, l2 = l2, np.sum(d_grid_pred**2)/2

        if i == max_iter:
            break

        # dl2/datom = sum_grid dl2/dgrid * dgrid/datom
        points, d_density = grid_to_points_and_values(d_grid_pred, center, resolution)
        d_atom_mean[...] = 0.0
        for j in range(n_atoms):
            for p, d in zip(points, d_density):
                d_atom_mean[j] += d * get_atom_gradient(atom_mean[j], atom_radius, p)
        atom_mean[...] -= lr*d_atom_mean

    return atom_mean, l2


def get_max_density_points(points, density, radius):
    assert len(points) > 0, 'no points provided'
    distance_check = lambda a, b: np.sum((a - b)**2) > radius**2
    max_points = []
    for p, d in sorted(zip(points, density), key=lambda x: -x[1]):
        if all(distance_check(p, max_p) for max_p in max_points):
            max_points.append(p)
            yield max_points[-1]


def grid_to_points_and_values(grid, center, resolution):
    dims = np.array(grid.shape)
    origin = np.array(center) - resolution*(dims-1)/2.
    indices = np.array(list(np.ndindex(*dims)))
    return origin + resolution*indices, grid.flatten()


def fit_atoms_to_grid(grid_args, center, resolution, max_iter, noise_type, by_L2, 
                      cython, density_threshold=0.0, verbose=True):
    grid, (channel_name, element, atom_radius), n_atoms = grid_args
    # nothing to fit if the whole grid is sub threshold
    if np.max(grid) <= density_threshold:
        return []
    if verbose:
        print('\nfitting {}'.format(channel_name))
    # convert grid to arrays of xyz points and density values
    points, density = grid_to_points_and_values(grid, center, resolution)
    if not by_L2: # initialize noise model params
        noise_params_init = dict(mean=np.mean(density),
                                 cov=np.cov(density),
                                 prob=1.0/len(points))
        if noise_type != 'd': # TODO this breaks d noise model
            # speed up GMM by only fitting points above threshold
            points = points[density > density_threshold,:]
            density = density[density > density_threshold]
    elif noise_type:
        raise NotImplementedError('TODO implement by_L2 noise model')
    density_sum = np.sum(density)
    # generator for inital atom positions
    max_density_points = get_max_density_points(points, density, atom_radius)
    xyz_init = np.ndarray((0, 3))
    if cython:
        if by_L2:
            raise NotImplementedError('TODO implement by_L2 cython')
        else:
            fit_atoms = cgenerate.fit_atoms_gmm
    else:
        if by_L2:
            fit_atoms = fit_atoms_l2
        else:
            fit_atoms = fit_atoms_gmm
    xyz_best, loss_best = [], np.inf
    if n_atoms is None: # iteratively add atoms and assess fit
        if not by_L2 and not noise_type:
            # can't fit GMM with 0 atoms and no noise model
            xyz_init = np.append(xyz_init, next(max_density_points)[np.newaxis,:], axis=0)
        while True:
            if by_L2:
                xyz, l2 = fit_atoms(grid, center, resolution, xyz_init, atom_radius, max_iter)
                loss = l2
            else:
                xyz, ll = fit_atoms(points, density, xyz_init, atom_radius,
                                    noise_type, noise_params_init, max_iter)
                loss = -ll
            if verbose:
                print('{:36}density_sum = {:.5f}\tn_atoms = {}\tloss = {:f}' \
                      .format(channel_name, density_sum, len(xyz), loss))
            if loss < loss_best:
                xyz_best, loss_best = xyz, loss
                try:
                    xyz_init = np.append(xyz_init, next(max_density_points)[np.newaxis,:], axis=0)
                except StopIteration:
                    break
            else:
                break
    else: # fit exactly n_atoms
        while len(xyz_init) < n_atoms:
            xyz_init = np.append(xyz_init, next(max_density_points)[np.newaxis,:], axis=0)
        if by_L2:
            xyz, l2 = fit_atoms(grid, center, resolution, xyz_init, atom_radius, max_iter)
            loss = l2
        else:
            xyz, ll = fit_atoms(points, density, xyz_init, atom_radius,
                                noise_type, noise_params_init, max_iter)
            loss = -ll
        if verbose:
            print('{:36}density_sum = {:.5f}\tn_atoms = {}\tloss = {:f}' \
                  .format(channel_name, density_sum, len(xyz), loss))
        xyz_best, loss_best = xyz, loss
    return xyz_best


def fit_atoms_to_grids(grids, channels, n_atoms, parallel=True, *args, **kwargs):
    grid_args = zip(grids, channels, n_atoms)
    if parallel:
        fmap = Pool(processes=len(grid_args)).map
    else:
        fmap = map
    f = partial(fit_atoms_to_grid, *args, **kwargs)
    return fmap(f, grid_args)


def fit_grid_to_grid(grid_args, center, resolution, *args, **kwargs):
    atom_radius = grid_args[1][2]
    xyz = fit_atoms_to_grid(grid_args, center, resolution, *args, **kwargs)
    return add_atoms_to_grid(np.zeros_like(grid), xyz, center, resolution, atom_radius)


def fit_grids_to_grids(grids, channels, n_atoms, parallel=True, *args, **kwargs):
    grid_args = zip(grids, channels, n_atoms)
    if parallel:
        fmap = Pool(processes=len(grid_args)).map
    else:
        fmap = map
    f = partial(fit_grid_to_grid, *args, **kwargs)
    return np.array(fmap(f, grid_args))


def rec_and_lig_at_index_in_data_file(file, index):
    with open(file, 'r') as f:
        line = f.readlines()[index]
    cols = line.rstrip().split()
    return cols[2], cols[3]


def best_loss_batch_index_from_net(net, loss_name, n_batches, best):
    loss = net.blobs[loss_name]
    best_index, best_loss = -1, None
    for i in range(n_batches):
        net.forward()
        l = float(np.max(loss.data))
        if i == 0 or best(l, best_loss) == l:
            best_loss = l
            best_index = i
            print('{} ({} / {})'.format(best_loss, i, n_batches))
    return best_index


def n_lines_in_file(file):
    with open(file, 'r') as f:
        return sum(1 for line in f)


def best_loss_rec_and_lig(model_file, weights_file, data_file, data_root, loss_name, best=max):
    n_batches = n_lines_in_file(data_file)
    with instantiate_model(model_file, data_file, data_file, data_root, 1) as model_file:
        net = caffe.Net(model_file, weights_file, caffe.TEST)
        index = best_loss_batch_index_from_net(net, loss_name, n_batches, best)
    return rec_and_lig_at_index_in_data_file(data_file, index)


def find_blobs_in_net(net, blob_pattern):
    blobs_found = []
    for blob_name, blob in net.blobs.items():
        if re.match('(?:' + blob_pattern + r')\Z', blob_name): # match full string
            blobs_found.append(blob)
    return blobs_found


def generate_grids_from_net(net, blob_pattern, index=0, rec_mode=None, lig_mode=None):
    blob = find_blobs_in_net(net, blob_pattern)[-1]
    index = batch_size = blob.shape[0]
    assert rec_mode in {None, 'zero', 'diff'}
    assert lig_mode in {None, 'unit', 'mean'}
    while index >= batch_size:
        net.forward(end='latent_concat')
        if rec_mode == 'zero':
            net.blobs['rec_latent_fc'].data[...] = 0.0
        if rec_mode == 'diff':
            net.forward(end='rec_latent_fc')
        if lig_mode == 'unit':
            net.blobs['lig_latent_mean'].data[...] = 0.0
            net.blobs['lig_latent_std'].data[...] = 1.0
        elif lig_mode == 'mean':
            net.forward(end='latent_concat')
            net.blobs['lig_latent_std'].data[...] = 0.0
        net.forward(start='lig_latent_noise') 
        index -= batch_size
    return blob.data[index]


def combine_element_grids_and_channels(grids, channels):
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
def temp_data_file(rec_file, lig_file):
    _, data_file = tempfile.mkstemp()
    with open(data_file, 'w') as f:
        f.write('0 0 {} {}'.format(rec_file, lig_file))
    yield data_file
    os.remove(data_file)


def write_pymol_script(pymol_file, dx_files, *extra_files):
    out = open(pymol_file, 'w')
    out.write('run set_atom_level.py\n')
    out.write('run isoslider.py\n')
    map_objects = []
    for dx_file in dx_files:
        map_object = dx_file.replace('.dx', '_map')
        out.write('load {}, {}\n'.format(dx_file, map_object))
        map_objects.append(map_object)
    map_group = pymol_file.replace('.pymol', '_maps')
    out.write('group {}, {}\n'.format(map_group, ' '.join(map_objects)))
    for extra_file in extra_files:
        out.write('load {}\n'.format(extra_file))
    out.close()


def write_atoms_to_sdf_file(sdf_file, xyzs, channels):
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
    dx_files = []
    for grid, (channel_name, _, _) in zip(grids, channels):
        dx_file = '{}_{}.dx'.format(out_prefix, channel_name)
        write_grid_to_dx_file(dx_file, grid, center, resolution)
        dx_files.append(dx_file)
    return dx_files


def get_channel_info_for_grids(grids):
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
    mol = Chem.MolFromMolFile(sdf_file)
    xyz = Chem.RemoveHs(mol).GetConformer().GetPositions()
    return xyz.mean(axis=0)


def get_n_atoms_from_sdf_file(sdf_file):
    mol = Chem.MolFromMolFile(sdf_file)
    return Counter(a.GetSymbol() for a in mol.GetAtoms())


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_file', required=True)
    parser.add_argument('-w', '--weights_file', required=True)
    parser.add_argument('-b', '--blob_name', required=True)
    parser.add_argument('-r', '--rec_file', default='')
    parser.add_argument('-l', '--lig_file', default='')
    parser.add_argument('-d', '--data_root', default='')
    parser.add_argument('-o', '--out_prefix', default=None)
    parser.add_argument('-c', '--cython', action='store_true')
    parser.add_argument('-n', '--noise_type', default='')
    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--combine_channels', action='store_true')
    parser.add_argument('--read_n_atoms', action='store_true')
    parser.add_argument('--output_dx', action='store_true')
    parser.add_argument('--output_sdf', action='store_true')
    parser.add_argument('--channel_info', default=None)
    parser.add_argument('--by_L2', action='store_true')
    parser.add_argument('--rec_mode')
    parser.add_argument('--lig_mode')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    rec_file = os.path.join(args.data_root, args.rec_file)
    lig_file = os.path.join(args.data_root, args.lig_file)
    center = get_center_from_sdf_file(lig_file)

    net_param = caffe_util.NetParameter.from_prototxt(args.model_file)
    resolution = net_param.get_molgrid_data_resolution(caffe.TEST)
    with temp_data_file(args.rec_file, args.lig_file) as data_file:
        net_param.set_molgrid_data_source(data_file, args.data_root, caffe.TEST)
        net = caffe_util.Net.from_param(net_param, args.weights_file, caffe.TEST)
    grids = generate_grids_from_net(net, args.blob_name, 0, args.rec_mode, args.lig_mode)

    print('shape = {}\ndensity sum = {}'.format(grids.shape, np.sum(grids)))

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

    dx_files = []
    if args.output_dx:
        dx_files += write_grids_to_dx_files(args.out_prefix, grids, channels, center, resolution)

    extra_files = [rec_file, lig_file]
    if args.output_sdf:

        if args.combine_channels:
            grids, channels = combine_element_grids_and_channels(grids, channels)

        if args.read_n_atoms:
            n_atoms = get_n_atoms_from_sdf_file(lig_file)
        else:
            n_atoms = [None for _ in grids]

        xyzs = fit_atoms_to_grids(grids, channels, n_atoms, center=center, resolution=resolution,
                                  max_iter=args.max_iter, noise_type=args.noise_type,
                                  by_L2=args.by_L2, cython=args.cython)

        pred_file = '{}_fit.sdf'.format(args.out_prefix)
        write_atoms_to_sdf_file(pred_file, xyzs, channels)
        extra_files.append(pred_file)

    pymol_file = '{}.pymol'.format(args.out_prefix)
    write_pymol_script(pymol_file, dx_files, *extra_files)


if __name__ == '__main__':
    main(sys.argv[1:])
