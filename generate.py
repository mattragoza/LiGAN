from __future__ import print_function
import sys, os, re, argparse, time, glob, struct, gzip, pickle
import datetime as dt
import numpy as np
import pandas as pd
import scipy as sp
from collections import defaultdict, Counter
import threading
import contextlib
import tempfile
try:
    from itertools import izip
except ImportError:
    izip = zip
from functools import partial
from scipy.stats import multivariate_normal
import caffe
import torch
import torch.multiprocessing as mp
from GPUtil import getGPUs
from openbabel import openbabel as ob
import rdkit
from rdkit import Chem, Geometry, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED, Crippen
from rdkit.Chem.Fingerprints import FingerprintMols
from SA_Score import sascorer
from NP_Score import npscorer

import molgrid
import caffe_util
import atom_types
from results import get_terminal_size


class MolGrid(object):
    '''
    An atomic density grid.
    '''
    def __init__(self, values, channels, center, resolution, **info):

        if len(values.shape) != 4:
            raise ValueError('MolGrid values must have 4 dims')

        if values.shape[0] != len(channels):
            raise ValueError('MolGrid values have wrong number of channels')

        if not (values.shape[1] == values.shape[2] == values.shape[3]):
            raise ValueError('last 3 dims of MolGrid values must be equal')

        self.values = values
        self.channels = channels
        self.center = center
        self.resolution = resolution
        self.size = self.values.shape[1]
        self.dimension = self.compute_dimension(self.size, resolution)

        self.info = info

    @classmethod
    def compute_dimension(cls, size, resolution):
        return (size-1)*resolution

    @classmethod
    def compute_size(cls, dimension, resolution):
        return int(np.ceil(dimension/resolution+1))

    def to_dx(self, dx_prefix, center=None):
        write_grids_to_dx_files(dx_prefix, self.values,
                                channels=self.channels,
                                center=self.center if center is None else center,
                                resolution=self.resolution)


class MolStruct(object):
    '''
    An atomic structure.
    '''
    def __init__(self, xyz, c, channels, bonds=None, **info):

        if len(xyz.shape) != 2:
            raise ValueError('MolStruct xyz must have 2 dims')

        if len(c.shape) != 1:
            raise ValueError('MolStruct c must have 1 dimension')

        if xyz.shape[0] != c.shape[0]:
            raise ValueError('first dim of MolStruct xyz and c must be equal')

        if xyz.shape[1] != 3:
            raise ValueError('second dim of MolStruct xyz must be 3')

        if any(c < 0) or any(c >= len(channels)):
            raise ValueError('invalid value in MolStruct c')

        self.n_atoms = xyz.shape[0]
        self.xyz = xyz

        self.channels = channels
        self.c = c

        if bonds is not None:
            if bonds.shape != (self.n_atoms, self.n_atoms):
                raise ValueError('MolStruct bonds must have shape (n_atoms, n_atoms)')
            self.bonds = bonds
        else:
            self.bonds = np.zeros((self.n_atoms, self.n_atoms))

        self.center = self.xyz.mean(0)

        if self.n_atoms > 0:
            self.radius = max(np.linalg.norm(self.xyz - self.center, axis=1))
        else:
            self.radius = np.nan

        self.info = info

    @classmethod
    def from_coord_set(self, coord_set, channels, **info):
        
        if not coord_set.has_indexed_types():
            raise ValueError('can only make MolStruct from CoordinateSet with indexed types')
        
        xyz = np.array(coord_set.coords, dtype=np.float32)
        c = np.array(coord_set.type_index, dtype=np.int16)
        
        return MolStruct(xyz, c, channels, **info)

    def to_ob_mol(self):
        mol = make_ob_mol(self.xyz.astype(float), self.c, self.bonds, self.channels)
        return mol

    def to_rd_mol(self):
        raise NotImplementedError('TODO')

    def to_sdf(self, sdf_file):
        write_rd_mols_to_sdf_file(sdf_file, [self.to_rd_mol()])

    def add_bonds(self, tol=0.0):

        nax = np.newaxis
        channel_radii = np.array([c.atomic_radius for c in self.channels])

        atom_dist2 = ((self.xyz[nax,:,:] - self.xyz[:,nax,:])**2).sum(axis=2)
        max_bond_dist2 = channel_radii[self.c][nax,:] + channel_radii[self.c][:,nax]
        self.bonds = (atom_dist2 < max_bond_dist2 + tol)


class AtomFitter(object):
    
    def __init__(self, beam_size, beam_stride, atom_init, interm_iters, final_iters,
                 learning_rate, beta1, beta2, weight_decay, constrain_types,
                 r_factor, output_visited, output_kernel, device, verbose):

        assert atom_init in {'none', 'conv', 'deconv'}

        self.beam_size = beam_size
        self.beam_stride = beam_stride
        self.atom_init = atom_init
        self.interm_iters = interm_iters
        self.final_iters = final_iters

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay

        self.constrain_types = constrain_types
        self.r_factor = r_factor

        self.device = device
        self.verbose = verbose

        self.grid_maker = molgrid.GridMaker()
        self.c2grid = molgrid.Coords2Grid(self.grid_maker)

        self.output_visited = output_visited
        self.output_kernel = output_kernel

    def init_kernel(self, channels, resolution, deconv=False):
        '''
        Initialize the convolution kernel that is used
        to propose next atom initializations on grids.
        '''
        n_channels = len(channels)

        # kernel is created by computing a molgrid from a
        # struct with one atom of each type at the origin
        xyz = torch.zeros((n_channels, 3), device=self.device)
        c = torch.eye(n_channels, device=self.device) # one-hot vector types
        r = torch.tensor([ch.atomic_radius for ch in channels], device=self.device)

        self.grid_maker.set_radii_type_indexed(True)
        self.grid_maker.set_dimension(2*1.5*max(r).item()) # kernel must fit max radius atom
        self.grid_maker.set_resolution(resolution)

        self.c2grid.center = (0.,0.,0.)
        kernel = self.c2grid(xyz, c, r)

        if deconv:
            dtype = kernel.dtype
            kernel = 100*torch.tensor(weiner_invert_kernel(kernel.cpu(), noise_ratio=1),
                                      dtype=dtype, device=self.device)

        if self.output_kernel:
            dx_prefix = 'deconv_kernel' if deconv else 'conv_kernel'
            if self.verbose:
                print('writing out {} (norm={})'.format(dx_prefix, np.linalg.norm(kernel.cpu())))
            write_grids_to_dx_files(dx_prefix, kernel.cpu(),
                                channels=channels,
                                center=np.zeros(3),
                                resolution=resolution)
            self.output_kernel = False

        return kernel

    def init_atoms(self, grids, center, resolution, kernel=None, types=None):
        '''
        Apply atom initialization kernel if needed and
        then return best next atom initializations on a
        grids or batch of grids (tensors).
        '''
        n_grids = grids.shape[0]
        n_channels = grids.shape[1]
        grid_dim = grids.shape[2]

        if self.constrain_types:
            grids = grids.clone()
            for i in range(n_grids):
                grids[i,types[i]<=0] = -1

        if kernel is not None: # apply atom init function to grids
            grids = torch.nn.functional.conv3d(grids, kernel.unsqueeze(1),
                                               padding=kernel.shape[-1]//2,
                                               groups=n_channels)

        # get indices of next atom positions and channels
        k = self.beam_size*self.beam_stride
        idx_flat = grids.reshape(n_grids, -1).topk(k).indices[:,::self.beam_stride]
        idx_grid = np.unravel_index(idx_flat.cpu(), grids.shape[1:])
        idx_xyz = torch.tensor(idx_grid[1:], device=self.device).permute(1, 2, 0)
        idx_c = idx_grid[0]

        # transform to xyz coordiates and type vectors
        xyz = center + resolution*(idx_xyz - (grid_dim-1)/2.)
        c = make_one_hot(idx_c, n_channels, dtype=torch.float32, device=self.device)

        return xyz, c

    def fit(self, grid, types, use_r_factor=False):
        '''
        Fit atomic structure to mol grid.
        '''
        t_start = time.time()

        types = torch.tensor(types, device=self.device)
        r_factor = self.r_factor if use_r_factor else 1.0

        if self.atom_init: # initialize convolution kernel
            deconv = (self.atom_init == 'deconv')
            kernel = self.init_kernel(grid.channels, grid.resolution, deconv)
        else:
            kernel = None

        # get true grid on appropriate device
        grid_true = MolGrid(values=torch.as_tensor(grid.values, device=self.device),
                            channels=grid.channels,
                            center=torch.as_tensor(grid.center, device=self.device),
                            resolution=grid.resolution)

        # empty initial struct
        n_channels = len(grid.channels)
        xyz = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        c = torch.zeros((0, n_channels), dtype=torch.float32, device=self.device)
        fit_loss = ((grid_true.values)**2).sum()/2.
        type_diff = types

        # to constrain types, order structs first by type-correctness, then by L2 loss
        if self.constrain_types:
            heuristic = (type_diff.abs().sum().item(), fit_loss.item())

        else: # otherwise, order structs only by L2 loss
            heuristic = fit_loss.item()

        # get next atom init locations and channels
        xyz_next, c_next = self.init_atoms(grid_true.values.unsqueeze(0), grid_true.center,
                                           grid.resolution, kernel, type_diff.unsqueeze(0))

        # batch of best structures so far
        best_structs = [(heuristic, 0, xyz, c, xyz_next[0], c_next[0])]
        found_new_best_struct = True
        visited = set()
        struct_count = 1
        visited_structs = []

        # search until we can't find a better structure
        while found_new_best_struct:

            new_best_structs = []
            new_best_grid_diffs = []
            new_best_type_diffs = []
            found_new_best_struct = False

            # try to expand each current best structure
            for heuristic, struct_id, xyz, c, xyz_next, c_next in best_structs:

                if struct_id in visited:
                    continue

                # evaluate each possible next atom
                for xyz_next_, c_next_ in zip(xyz_next, c_next):

                    # add next atom to structure
                    xyz_new = torch.cat([xyz, xyz_next_.unsqueeze(0)])
                    c_new   = torch.cat([c, c_next_.unsqueeze(0)])

                    # compute diff and loss after gradient descent
                    xyz_new, grid_pred, grid_diff, fit_loss = \
                        self.fit_gd(grid_true, xyz_new, c_new, self.interm_iters, r_factor)
                    type_diff = types - c_new.sum(dim=0)

                    if self.constrain_types:
                        heuristic_new = (type_diff.abs().sum().item(), fit_loss.item())
                    else:
                        heuristic_new = fit_loss.item()

                    # check if new structure is one of the best yet
                    if any(heuristic_new < s[0] for s in best_structs):

                        new_best_structs.append((heuristic_new, struct_count, xyz_new, c_new))
                        new_best_grid_diffs.append(grid_diff)
                        new_best_type_diffs.append(type_diff)
                        found_new_best_struct = True
                        struct_count += 1

                visited.add(struct_id)
                if self.output_visited:
                    visited_structs.append((heuristic, struct_id, time.time()-t_start, xyz, c))

            if found_new_best_struct:

                new_best_grid_diffs = torch.stack(new_best_grid_diffs)
                new_best_type_diffs = torch.stack(new_best_type_diffs)

                # get next-atom init coords and types for new set of best structures
                xyz_next, c_next = self.init_atoms(new_best_grid_diffs, grid_true.center,
                                                   grid_true.resolution, kernel, new_best_type_diffs)

                # determine new limited set of best structures
                for i, (heuristic, struct_id, xyz, c) in enumerate(new_best_structs):
                    best_structs.append((heuristic, struct_id, xyz, c, xyz_next[i], c_next[i]))

                best_structs = sorted(best_structs)[:self.beam_size]
                best_heuristic = best_structs[0][0]
                best_id = best_structs[0][1]

                if self.verbose:
                    try:
                        gpu_usage = getGPUs()[0].memoryUtil
                    except:
                        gpu_usage = np.nan
                    print('best struct # {} (heuristic={}, GPU={})'
                          .format(best_id, best_heuristic, gpu_usage))

        best_heuristic, best_id, xyz_best, c_best, _, _ = best_structs[0]

        xyz_best, grid_pred, grid_diff, fit_loss = \
            self.fit_gd(grid_true, xyz_best, c_best, self.final_iters, r_factor)
        type_diff = (types - c_best.sum(dim=0)).abs().sum().item()

        grid_pred = MolGrid(grid_pred.cpu().detach().numpy(),
                            grid.channels, grid.center, grid.resolution)

        if self.output_visited: # return all visited structures

            struct_best = []
            for heuristic, struct_id, fit_time, xyz, c in visited_structs:

                if self.constrain_types:
                    type_diff, fit_loss = heuristic
                else:
                    fit_loss = heuristic
                    type_diff = (types - c.sum(dim=0)).abs().sum().item()

                c = torch.argmax(c, dim=1) if len(c) > 0 else torch.zeros((0,))
                struct = MolStruct(xyz.cpu().detach().numpy(), c.cpu().detach().numpy(),
                                   grid.channels, loss=fit_loss, type_diff=type_diff, time=fit_time)
                struct_best.append(struct)

        else: # return only the best structure
            c_best = torch.argmax(c_best, dim=1) if len(c_best) > 0 else torch.zeros((0,))
            struct_best = MolStruct(xyz_best.cpu().detach().numpy(), c_best.cpu().detach().numpy(),
                                    grid.channels, loss=fit_loss, type_diff=type_diff, time=time.time()-t_start)

        return grid_pred, struct_best

    def fit_gd(self, grid, xyz, c, n_iters, r_factor=1.0):

        r = torch.tensor([ch.atomic_radius for ch in grid.channels],
                         device=self.device) * r_factor

        xyz = xyz.clone().detach().to(self.device)
        c = c.clone().detach().to(self.device)
        xyz.requires_grad = True

        solver = torch.optim.Adam((xyz,), lr=self.learning_rate,
                                  betas=(self.beta1, self.beta2),
                                  weight_decay=self.weight_decay)

        self.grid_maker.set_radii_type_indexed(True)
        self.grid_maker.set_dimension(grid.dimension)
        self.grid_maker.set_resolution(grid.resolution)
        self.c2grid.center = tuple(grid.center.cpu().numpy().astype(float))

        for i in range(n_iters+1):
            solver.zero_grad()

            grid_pred = self.c2grid(xyz, c, r)
            grid_diff = grid.values - grid_pred
            loss = (grid_diff**2).sum()/2.

            if i == n_iters: # or converged
                break

            loss.backward()
            solver.step()

        return xyz.detach(), grid_pred.detach(), grid_diff.detach(), loss.detach()


class OutputWriter(object):
    '''
    A data structure for receiving and organizing MolGrids and
    MolStructs from a generative model or atom fitting algorithm,
    computing metrics, and writing files to disk as necessary.
    '''
    def __init__(self, out_prefix, output_dx, output_sdf, output_channels,
                 n_samples, blob_names, fit_atoms, verbose):

        self.out_prefix = out_prefix
        self.output_dx = output_dx
        self.output_sdf = output_sdf
        self.output_channels = output_channels
        self.fit_atoms = fit_atoms

        # organize grids and structs by lig_name, grid_name, sample_idx
        self.grids = defaultdict(lambda: defaultdict(dict))
        self.structs = defaultdict(lambda: defaultdict(dict))

        # compute the number of grids to expect
        self.n_grids = 0
        for b in blob_names:
            self.n_grids += 1
            if fit_atoms and b.startswith('lig'):
                self.n_grids += 1
        self.n_samples = n_samples

        # accumulate metrics in dataframe
        self.metric_file = '{}.gen_metrics'.format(out_prefix)
        columns = ['lig_name', 'sample_idx']
        self.metrics = pd.DataFrame(columns=columns).set_index(columns)

        # write a pymol script when finished
        self.pymol_file = '{}.pymol'.format(out_prefix)
        self.dx_prefixes = []
        self.struct_files = []
        self.centers = []

        self.verbose = verbose
        self.nps_model = npscorer.readNPModel()

    def write(self, lig_name, grid_name, sample_idx, grid, struct):
        '''
        Add grid and struct to the data structure and write output
        for lig_name, if all expected grids and structs are present.
        '''
        if self.verbose:
            print('out_writer got {} {} {}'.format(lig_name, grid_name, sample_idx))

        self.grids[lig_name][grid_name][sample_idx] = grid
        self.structs[lig_name][grid_name][sample_idx] = struct

        has_all_grids = (len(self.grids[lig_name]) == self.n_grids)
        has_all_samples = all(len(g) == self.n_samples for g in self.grids[lig_name].values())

        if not (has_all_grids and has_all_samples):
            return False

        if self.verbose:
            print('out_writer has all grids for ' + lig_name)

        lig_grids = self.grids[lig_name]
        lig_structs = self.structs[lig_name]
        lig_prefix = '{}_{}'.format(self.out_prefix, lig_name)

        if self.output_dx: # load ligand grids in a single pymol group
            self.dx_prefixes.append(lig_prefix)

        for grid_name in lig_grids:
            grid_prefix = '{}_{}_{}'.format(self.out_prefix, lig_name, grid_name)

            if self.output_dx:

                for sample_idx, grid in lig_grids[grid_name].items():
                    sample_prefix = grid_prefix + '_' + str(sample_idx)
                    if self.verbose:
                        print('out_writer writing ' + sample_prefix + ' .dx files')
                    grid.to_dx(sample_prefix, center=np.zeros(3))

            if self.output_sdf:

                structs = []
                for sample_idx, struct in lig_structs[grid_name].items():

                    if isinstance(struct, list): # all visited structs
                        sample_prefix = grid_prefix + '_' + str(sample_idx)
                        struct_file = sample_prefix + '.sdf'
                        if self.verbose:
                            print('out_writer writing ' + struct_file)
                        write_structs_to_sdf_file(struct_file, struct)
                        self.struct_files.append(struct_file)

                        # best struct
                        struct = sorted(struct, key=lambda s: s.info['loss'])[0]
                        lig_structs[grid_name][i] = struct
                        self.centers.append(struct.center)

                    if struct is not None:
                        structs.append(struct)

                        if self.output_channels:
                            sample_prefix = grid_prefix + '_' + str(sample_idx)
                            channels_file = '{}.channels'.format(sample_prefix)
                            if self.verbose:
                                print('out_writer writing ' + channels_file)
                            write_channels_to_file(channels_file, struct.c, struct.channels)

                # write final stucts to single sdf file so that they
                # are loaded into diff states of single pymol object
                if structs:
                    struct_file = grid_prefix + '.sdf'
                    if self.verbose:
                        print('out_writer writing ' + struct_file)
                    write_structs_to_sdf_file(struct_file, structs)
                    self.struct_files.append(struct_file)
                    self.centers.append(structs[0].center)

        if self.verbose:
            print('out_writer computing metrics for ' + lig_name)
        self.compute_metrics(lig_name, lig_grids, lig_structs)

        if self.verbose:
            print('out_writer writing ' + self.metric_file)
        self.metrics.to_csv(self.metric_file, sep=' ')

        if self.verbose:
            print('out_writer writing ' + self.pymol_file)
        write_pymol_script(self.pymol_file, self.dx_prefixes, self.struct_files, self.centers)

        if self.verbose:
            print('out_writer freeing ' + lig_name)
        del self.grids[lig_name] # free memory
        del self.structs[lig_name]

    def compute_metrics(self, lig_name, grids, structs):
        '''
        Compute metrics on generated density grids, fit atoms, and valid
        molecules for a given ligand and insert in metrics data frame.
        '''
        # use mean grids to evaluate variability
        lig_grid_mean     = sum(g.values for g in grids['lig'].values())/self.n_samples
        lig_gen_grid_mean = sum(g.values for g in grids['lig_gen'].values())/self.n_samples

        for i in range(self.n_samples):
            idx = (lig_name, i)

            lig_grid     = grids['lig'][i]
            lig_gen_grid = grids['lig_gen'][i]

            self.compute_grid_metrics(idx, 'lig', lig_grid, lig_gen_grid, lig_grid_mean, lig_gen_grid_mean)

            if not self.fit_atoms: # start of atom fitting metrics
                continue

            lig_fit_grid     = grids['lig_fit'][i]
            lig_gen_fit_grid = grids['lig_gen_fit'][i]

            lig_struct         = structs['lig'][i]
            lig_fit_struct     = structs['lig_fit'][i]
            lig_gen_fit_struct = structs['lig_gen_fit'][i]

            self.compute_fit_metrics(idx, 'lig',     lig_grid,     lig_fit_grid,     lig_struct, lig_fit_struct)
            self.compute_fit_metrics(idx, 'lig_gen', lig_gen_grid, lig_gen_fit_grid, lig_struct, lig_gen_fit_struct)

            lig_mol = lig_struct.info['src_mol']

            self.compute_mol_validity(idx, 'lig',     lig_mol, lig_fit_struct)
            self.compute_mol_validity(idx, 'lig_gen', lig_mol, lig_gen_fit_struct)

        if self.verbose:
            print(self.metrics.loc[lig_name])

    def compute_grid_metrics(self, idx, prefix, true_grid, gen_grid, true_grid_mean, gen_grid_mean):

        m = self.metrics

        # density magnitude
        m.loc[idx, prefix+'_norm']     = np.linalg.norm(true_grid.values)
        m.loc[idx, prefix+'_gen_norm'] = np.linalg.norm(gen_grid.values)

        # latent sample magnitude
        m.loc[idx, prefix+'_latent_norm'] = gen_grid.info['latent_norm']

        # generated density L2 loss
        m.loc[idx, prefix+'_gen_L2_loss'] = \
            (((true_grid.values - gen_grid.values)**2).sum()/2).item()

        # density variance (divide by n_samples(+1) for sample (population) variance)
        m.loc[idx, prefix+'_variance']     = ((true_grid.values - true_grid_mean)**2).sum().item()
        m.loc[idx, prefix+'_gen_variance'] = ((gen_grid.values  - gen_grid_mean)**2).sum().item()

    def compute_fit_metrics(self, idx, prefix, true_grid, fit_grid, true_struct, fit_struct):

        m = self.metrics

        # fit density L2 loss
        m.loc[idx, prefix+'_fit_L2_loss'] = \
            (((true_grid.values - fit_grid.values)**2).sum()/2).item()

        # number of atoms
        if not prefix.endswith('_gen'):
            m.loc[idx, prefix+'_n_atoms'] = true_struct.n_atoms
        m.loc[idx, prefix+'_fit_n_atoms'] = fit_struct.n_atoms

        # fit structure radius
        if not prefix.endswith('_gen'):
            m.loc[idx, prefix+'_radius'] = true_struct.radius
        m.loc[idx, prefix+'_fit_radius'] = fit_struct.radius

        n_types = len(true_struct.channels)
        true_type_count = count_types(true_struct.c, n_types)
        fit_type_count  = count_types(fit_struct.c, n_types)

        # fit type difference
        m.loc[idx, prefix+'_fit_type_diff'] = np.linalg.norm(true_type_count - fit_type_count, ord=1)

        # fit minimum RMSD
        try:
            rmsd = get_min_rmsd(true_struct.xyz, true_struct.c, fit_struct.xyz, fit_struct.c)
        except (ValueError, ZeroDivisionError):
            rmsd = np.nan
        m.loc[idx, prefix+'_fit_RMSD'] = rmsd

        # fit time
        m.loc[idx, prefix+'_fit_time'] = fit_struct.info['time']

    def compute_mol_validity(self, idx, prefix, true_mol, fit_struct):

        m = self.metrics

        # get aromatic carbon, nitrogen donor channels
        aroma_c_channels = set()
        n_donor_channels = set()
        for i, channel in enumerate(fit_struct.channels):
            if 'AromaticCarbon' in channel.name:
                aroma_c_channels.add(i)
            if 'Nitrogen' in channel.name and 'Donor' in channel.name:
                n_donor_channels.add(i)

        # get aromatic carbon atoms
        aroma_c_atoms = set()
        for i, c in enumerate(fit_struct.c):
            if c in aroma_c_channels:
                aroma_c_atoms.add(i)

        # perceive bonds in openbabel
        mol = fit_struct.to_ob_mol()
        mol.ConnectTheDots()
        mol.PerceiveBondOrders()
        # WARNING writing from OpenBabel to sdf, then reading with RDKit causes radicals to appear

        # do the rest in rdkit
        mol = ob_mol_to_rd_mol(mol)

        # sanity check- validity of true molecule
        if not prefix.endswith('_gen'):
            n_frags, error, valid = get_rd_mol_validity(true_mol)
            m.loc[idx, prefix+'_n_frags'] = n_frags
            m.loc[idx, prefix+'_error']   = error
            m.loc[idx, prefix+'_valid']   = valid

        # check validity prior to bond adding
        n_frags, error, valid = get_rd_mol_validity(mol)
        m.loc[idx, prefix+'_fit_n_frags'] = n_frags
        m.loc[idx, prefix+'_fit_error']   = error
        m.loc[idx, prefix+'_fit_valid']   = valid

        # make aromatic rings using channel info
        rings = Chem.GetSymmSSSR(mol)
        for ring_atoms in rings:
            ring_atoms = set(ring_atoms)
            if len(ring_atoms & aroma_c_atoms) == 0:
                continue
            if (len(ring_atoms) - 2)%4 != 0:
                continue
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                if i in ring_atoms and j in ring_atoms:
                    bond.SetBondType(Chem.BondType.AROMATIC)

        #Chem.Kekulize(mol) # do we need to do this?

        # TODO add hydrogens using channel info

        # try to connect fragments by adding min distance bonds
        frags = Chem.GetMolFrags(mol)
        n_frags = len(frags)
        if n_frags > 1:

            nax = np.newaxis
            xyz = mol.GetConformer(0).GetPositions()
            dist2 = ((xyz[nax,:,:] - xyz[:,nax,:])**2).sum(axis=2)

            pt = Chem.GetPeriodicTable()
            while n_frags > 1:

                frag_map = {ai: fi for fi, f in enumerate(frags) for ai in f}
                frag_idx = np.array([frag_map[i] for i in range(mol.GetNumAtoms())])
                diff_frags = frag_idx[nax,:] != frag_idx[:,nax]

                can_bond = np.array([a.GetExplicitValence() <
                                     pt.GetDefaultValence(a.GetAtomicNum())
                                     for a in mol.GetAtoms()])
                can_bond = can_bond[nax,:] & can_bond[nax,:]

                cond_dist2 = np.where(diff_frags & can_bond & dist2<25, dist2, np.inf)

                if not np.any(np.isfinite(cond_dist2)):
                    break # no possible bond meets the conditons

                a1, a2 = np.unravel_index(cond_dist2.argmin(), dist2.shape)
                mol.AddBond(int(a1), int(a2), Chem.BondType.SINGLE)
                mol.UpdatePropertyCache() # update explicit valences

                frags = Chem.GetMolFrags(mol)
                n_frags = len(frags)
        
        # check validity after bond adding
        n_frags, error, valid = get_rd_mol_validity(mol)
        m.loc[idx, prefix+'_fit_add_n_frags'] = n_frags
        m.loc[idx, prefix+'_fit_add_error']   = error
        m.loc[idx, prefix+'_fit_add_valid']   = valid

        # energy minimization with UFF
        true_uffE = get_rd_mol_uff_energy(true_mol)
        min_mol, init_uffE, min_uffE, _ = uff_minimize_rd_mol(mol)

        m.loc[idx, prefix+'_uffE']             = true_uffE
        m.loc[idx, prefix+'_fit_add_uffE']     = init_uffE
        m.loc[idx, prefix+'_fit_add_min_uffE'] = min_uffE

        # TODO get RMSD to true struct, and before-after minimizing
        try:
            rmsd = AllChem.GetBestRMS(min_mol, true_mol)
        except RuntimeError:
            rmsd = np.nan
        m.loc[idx, prefix+'_fit_add_RMSD_true'] = rmsd

        try:
            rmsd = AllChem.GetBestRMS(mol, min_mol)
        except RuntimeError:
            rmsd = np.nan
        m.loc[idx, prefix+'_fit_add_RMSD_min']  = rmsd

        # convert to smiles string
        true_smi = Chem.MolToSmiles(true_mol, canonical=True)
        smi      = Chem.MolToSmiles(mol,      canonical=True)

        if not prefix.endswith('_gen'):
            m.loc[idx, prefix+'_SMILES'] = true_smi
        m.loc[idx, prefix+'_fit_add_SMILES'] = smi
        m.loc[idx, prefix+'_fit_add_SMILES_match'] = smi == true_smi

        # fingerprint similarity
        m.loc[idx, prefix+'_fit_add_morgan_sim'] = get_rd_mol_similarity(true_mol, mol, 'morgan')
        m.loc[idx, prefix+'_fit_add_rdkit_sim']  = get_rd_mol_similarity(true_mol, mol, 'rdkit')
        m.loc[idx, prefix+'_fit_add_maccs_sim']  = get_rd_mol_similarity(true_mol, mol, 'maccs')

        # other molecular descriptors
        if not prefix.endswith('_gen'):
            m.loc[idx, prefix+'_MW']   = Chem.Descriptors.MolWt(true_mol)
            m.loc[idx, prefix+'_QED']  = Chem.QED.default(true_mol)
            m.loc[idx, prefix+'_logP'] = Crippen.MolLogP(true_mol)
            m.loc[idx, prefix+'_SAS']  = sascorer.calculateScore(true_mol)
            m.loc[idx, prefix+'_NPS']  = npscorer.scoreMol(true_mol, self.nps_model)

        m.loc[idx, prefix+'_fit_add_min_MW']   = Chem.Descriptors.MolWt(mol)
        m.loc[idx, prefix+'_fit_add_min_QED']  = Chem.QED.default(mol)
        m.loc[idx, prefix+'_fit_add_min_logP'] = Crippen.MolLogP(mol)
        m.loc[idx, prefix+'_fit_add_min_SAS']  = sascorer.calculateScore(mol)
        m.loc[idx, prefix+'_fit_add_min_NPS']  = npscorer.scoreMol(mol, self.nps_model)



def get_rd_mol_similarity(rd_mol1, rd_mol2, fingerprint):
    if fingerprint == 'morgan':
        fgp1 = AllChem.GetMorganFingerprintAsBitVect(rd_mol1, 2, 1024)
        fgp2 = AllChem.GetMorganFingerprintAsBitVect(rd_mol2, 2, 1024)
    elif fingerprint == 'rdkit':
        fgp1 = Chem.Fingerprints.FingerprintMols.FingerprintMol(rd_mol1)
        fgp2 = Chem.Fingerprints.FingerprintMols.FingerprintMol(rd_mol2)
    elif fingerprint == 'maccs':
        fgp1 = AllChem.GetMACCSKeysFingerprint(rd_mol1)
        fgp2 = AllChem.GetMACCSKeysFingerprint(rd_mol2)
    return DataStructs.TanimotoSimilarity(fgp1, fgp2)


def uff_minimize_rd_mol(rd_mol, max_iters=1000):

    if AllChem.UFFHasAllMoleculeParams(rd_mol):
        rd_mol_H = Chem.AddHs(rd_mol, addCoords=True)
        E_init = AllChem.UFFGetMoleculeForceField(rd_mol_H).CalcEnergy()
        converged = AllChem.UFFOptimizeMolecule(rd_mol_H, maxIters=max_iters) == 0
        E_final = AllChem.UFFGetMoleculeForceField(rd_mol_H).CalcEnergy()
        rd_mol = Chem.RemoveHs(rd_mol_H)
        return rd_mol, E_init, E_final, converged
    else:
        return rd_mol, np.nan, np.nan, np.nan


def get_rd_mol_uff_energy(rd_mol): # TODO do we need to add H for true mol?
    energy = AllChem.UFFGetMoleculeForceField(rd_mol).CalcEnergy()


def get_rd_mol_validity(rd_mol):
    n_frags = len(Chem.GetMolFrags(rd_mol))
    try:
        Chem.SanitizeMol(rd_mol)
    except Exception as e:
        error = e
    else:
        error = None
    valid = n_frags == 1 and error is None
    return n_frags, error, valid


def ob_mol_to_rd_mol(ob_mol):

    n_atoms = ob_mol.NumAtoms()
    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(n_atoms)

    for ob_atom in ob.OBMolAtomIter(ob_mol):
        rd_atom = Chem.Atom(ob_atom.GetAtomicNum())
        #TODO copy format charge
        i = rd_mol.AddAtom(rd_atom)
        ob_coords = ob_atom.GetVector()
        x = ob_coords.GetX()
        y = ob_coords.GetY()
        z = ob_coords.GetZ()
        rd_coords = Geometry.Point3D(x, y, z)
        rd_conf.SetAtomPosition(i, rd_coords)

    rd_mol.AddConformer(rd_conf)

    for ob_bond in ob.OBMolBondIter(ob_mol):
        i = ob_bond.GetBeginAtomIdx()-1
        j = ob_bond.GetEndAtomIdx()-1
        bond_order = ob_bond.GetBondOrder()
        if ob_bond.IsAromatic():
            rd_mol.AddBond(i, j, Chem.BondType.AROMATIC)
        elif bond_order == 1:
            rd_mol.AddBond(i, j, Chem.BondType.SINGLE)
        elif bond_order == 2:
            rd_mol.AddBond(i, j, Chem.BondType.DOUBLE)
        elif bond_order == 3:
            rd_mol.AddBond(i, j, Chem.BondType.TRIPLE)
        else:
            raise Exception('unknown bond order {}'.format(bond_order))

    return rd_mol


def make_ob_mol(xyz, c, bonds, channels):
    '''
    Return an OpenBabel molecule from an array of
    xyz atom positions, channel indices, a bond matrix,
    and a list of atom type channels.
    '''
    mol = ob.OBMol()

    n_atoms = 0
    for (x, y, z), c_ in zip(xyz, c):
        atomic_num = channels[c_].atomic_num
        atom = mol.NewAtom()
        atom.SetAtomicNum(atomic_num)
        atom.SetVector(x, y, z)
        n_atoms += 1

    if np.any(bonds):
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
        if i == 0:
            conv.WriteFile(mol, sdf_file)
        else:
            conv.Write(mol)
    conv.CloseOutFile()


def write_rd_mols_to_sdf_file(sdf_file, mols):
    writer = Chem.SDWriter(sdf_file)
    for mol in mols:
        writer.write(mol)


def read_rd_mols_from_sdf_file(sdf_file):
    if sdf_file.endswith('.gz'):
        open = gzip.open
    with open(sdf_file) as f:
        suppl = Chem.ForwardSDMolSupplier(f)
        return [mol for mol in suppl]


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


def get_atom_density2(atom_pos, atom_radius, points, radius_multiple):
    return np.exp(-2*np.sum((points - atom_pos)**2, axis=1)/atom_radius**2)


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


def get_bond_length_energy(distance, bond_length, bonds):
    '''
    Compute the interatomic potential energy between an atom and a set of atoms.
    '''
    exp = np.exp(bond_length - distance)
    return (1 - exp)**2 * bonds


def get_bond_length_gradient(distance, bond_length, bonds):
    '''
    Compute the derivative of interatomic potential energy between an atom
    and a set of atoms with respect to the position of the first atom.
    '''
    exp = np.exp(bond_length - distance)
    return 2 * (1 - exp) * exp * bonds
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


def conv_grid(grid, kernel):
    # convolution theorem: g * grid = F-1(F(g)F(grid))
    F_h = np.fft.fftn(kernel)
    F_grid = np.fft.fftn(grid)
    return np.real(np.fft.ifftn(F_grid * F_h))


def weiner_invert_kernel(kernel, noise_ratio=0.0):
    F_h = np.fft.fftn(kernel)
    conj_F_h = np.conj(F_h)
    F_g = conj_F_h / (F_h*conj_F_h + noise_ratio)
    return np.real(np.fft.ifftn(F_g))


def wiener_deconv_grid(grid, kernel, noise_ratio=0.0):
    '''
    Applies a convolution to the input grid that approximates the inverse
    of the operation that converts a set of atom positions to a grid of
    atom density.
    '''
    # we want a convolution g such that g * grid = a, where a is the atom positions
    # we assume that grid = h * a, so g is the inverse of h: g * (h * a) = a
    # take F() to be the Fourier transform, F-1() the inverse Fourier transform
    # convolution theorem: g * grid = F-1(F(g)F(grid))
    # Wiener deconvolution: F(g) = 1/F(h) |F(h)|^2 / (|F(h)|^2 + noise_ratio)
    F_h = np.fft.fftn(kernel)
    F_grid = np.fft.fftn(grid)
    conj_F_h = np.conj(F_h)
    F_g = conj_F_h / (F_h*conj_F_h + noise_ratio)
    return np.real(np.fft.ifftn(F_grid * F_g))


def wiener_deconv_grids(grids, channels, resolution, radius_multiple, noise_ratio=0.0, radius_factor=1.0):

    deconv_grids = np.zeros_like(grids)
    points = get_grid_points(grids.shape[1:], 0, resolution)

    for i, grid in enumerate(grids):

        r = channels[i].atomic_radius*radius_factor
        kernel = get_atom_density(resolution/2, r, points, radius_multiple).reshape(grid.shape)
        kernel = np.roll(kernel, shift=[d//2 for d in grid.shape], axis=range(grid.ndim))
        deconv_grids[i,...] = wiener_deconv_grid(grid, kernel, noise_ratio)

    return np.stack(deconv_grids, axis=0)


def get_grid_points(shape, center, resolution):
    '''
    Return an array of points for a grid with
    the given shape, center, and resolution.
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


def get_atom_density_kernel(shape, resolution, atom_radius, radius_mult):
    '''
    Return atom density function as a grid with the
    given shape, resolution, and atom radius.
    '''
    center = np.zeros(len(shape))
    points = get_grid_points(shape, center, resolution)
    density = get_atom_density(center, atom_radius, points, radius_mult)
    return density.reshape(shape)


def fit_atoms_by_GD(points, density, xyz, c, bonds, atomic_radii, max_iter, 
                    lr, mo, lambda_E=0.0, radius_multiple=1.5, verbose=0,
                    density_pred=None, density_diff=None):
    '''
    Fit atom positions, provided by arrays xyz initial positions, c channel indices, 
    and optional bonds matrix, to arrays of points with the given channel density values.
    Minimize the L2 loss (and optionally interatomic energy) between the provided density
    and fitted density by gradient descent with momentum. Return the final atom positions
    and loss.
    '''
    n_atoms = len(xyz)

    xyz = np.array(xyz)
    d_loss_d_xyz = np.zeros_like(xyz)
    d_loss_d_xyz_prev = np.zeros_like(xyz)
    
    if density_pred is None:
        density_pred = np.zeros_like(density)

    if density_diff is None:
        density_diff = np.zeros_like(density)

    ax = np.newaxis
    if lambda_E:
        xyz_diff = np.zeros((n_atoms, n_atoms, 3))
        xyz_dist = np.zeros((n_atoms, n_atoms))
        bond_length = atomic_radii[:,ax] + atomic_radii[ax,:]

    # minimize loss by gradient descent
    loss = np.inf
    i = 0
    while True:
        loss_prev = loss

        # L2 loss between predicted and true density
        density_pred[...] = 0.0
        for j in range(n_atoms):
            density_pred[:,c[j]] += get_atom_density(xyz[j], atomic_radii[j], points, radius_multiple)

        density_diff[...] = density - density_pred
        loss = (density_diff**2).sum()

        # interatomic energy of predicted atom positions
        if lambda_E:
            xyz_diff[...] = xyz[:,ax,:] - xyz[ax,:,:]
            xyz_dist[...] = np.linalg.norm(xyz_diff, axis=2)
            for j in range(n_atoms):
                loss += lambda_E * get_bond_length_energy(xyz_dist[j,j+1:], bond_length[j,j+1:], bonds[j,j+1:]).sum()

        delta_loss = loss - loss_prev
        if verbose > 2:
            print('n_atoms = {}\titer = {}\tloss = {} ({})'.format(n_atoms, i, loss, delta_loss), file=sys.stderr)

        if n_atoms == 0 or i == max_iter or abs(delta_loss)/(abs(loss_prev) + 1e-8) < 1e-2:
            break

        # compute derivatives and descend loss gradient
        d_loss_d_xyz_prev[...] = d_loss_d_xyz
        d_loss_d_xyz[...] = 0.0

        for j in range(n_atoms):
            d_density_d_xyz = get_atom_gradient(xyz[j], atomic_radii[j], points, radius_multiple)
            d_loss_d_xyz[j] += (-2*density_diff[:,c[j],ax] * d_density_d_xyz).sum(axis=0)

        if lambda_E:
            for j in range(n_atoms-1):
                d_E_d_dist = get_bond_length_gradient(xyz_dist[j,j+1], bond_length[j,j+1], bonds[j,j+1:])
                d_E_d_xyz = xyz_diff[j,j+1:] * (d_E_d_dist / xyz_dist[j,j+1:])[:,ax]
                d_xyz[j] += lambda_E * d_E_d_xyz.sum(axis=0)
                d_xyz[j+1:,:] -= lambda_E * d_E_d_xyz

        xyz[...] -= lr*(mo*d_loss_d_xyz_prev + (1-mo)*d_loss_d_xyz)
        i += 1

    return xyz, density_pred, density_diff, loss


def get_top_n_index(tensor, n):
    '''
    Return indices of top-n values in tensor.
    '''
    idx = tensor.reshape(-1).topk(n).indices
    return np.unravel_index(idx, tensor.size)


def add_mol_density(xyz, t, r, density, points):
    raise NotImplementedError('TODO')
    for xyz_, t_, r_ in zip(xyz, t, r):
        density[:,t_] += get_atom_density(xyz_, r_, points)


def fit_L2_loss(xyz, t, r, density):
    raise NotImplementedError('TODO')
    return np.sum((density_true - density_pred)**2)


def grad_fit_L2_loss(xyz_pred, density_true):
    raise NotImplementedError('TODO')
    return density_true - density_pred


def fit_atoms_to_grid(grid, channels, center, resolution, beam_size, atom_init,
                      interm_iters, final_iters, lr, mo, lambda_E, radius_multiple,
                      bonded, max_init_bond_E, fit_channels, verbose, device):
    '''
    Fit atoms to grid by iteratively placing atoms and then optimizing their
    positions by gradient descent on L2 loss between the provided grid density
    and the density associated with the fitted atoms.
    '''
    t_start = time.time()

    center = torch.tensor(center, device=device)
    radii = torch.tensor([c.atomic_radius for c in channels], device=device)
    max_bonds = torch.tensor([atom_types.get_max_bonds(c.atomic_num) for c in channels], device=device)
    rm = radius_multiple

    # configure the gridmaker
    dimension = resolution*(grid.shape[1]-1)
    gmaker = molgrid.GridMaker()
    gmaker.set_resolution(resolution)
    gmaker.set_radius_type_indexed(False)
    c2grid = molgrid.Coords2Grid(gmaker, center)

    # convert input grid to tensor
    n_channels, grid_shape = grid.shape[0], grid.shape[1:]
    points = get_grid_points(grid_shape, center, resolution)
    grid_true = torch.tensor(grid, device=device)

    # get empty initial structure
    xyz = torch.zeros((0, 3), device=device)
    c = torch.zeros(0, dtype=int, device=device)
    bonds = torch.zeros((0, 0), device=device)

    # initial predicted grid, diff, and loss
    grid_pred = torch.zeros_like(grid_true, device=device)
    grid_diff = grid_true - grid_pred
    loss = (grid_diff**2).sum()

    # get centered atoms to make kernels
    xyz_kernel = center.unsqueeze(0).repeat(n_channels, device=device)
    c_kernel = torch.arange(n_channels, dtype=int, device=device)

    # init atom density kernels
    kernel_size = 15 # ceil(2*rm*max(radii)/resolution + 1)
    kernel_shape = (n_channels, n_channels, kernel_size, kernel_size, kernel_size)
    gmaker.set_dimension(resolution*(kernel_size-1)) # TODO check that this affects c2grid
    kernel = c2grid(xyz_kernel, c_kernel, radii)

    #################### PICK UP HERE ########################

    # select possible next atoms from remaining density
    next_idx = get_top_n_index(density_diff, beam_size)

    # keep track of a set of best structures
    best_structs = [(xyz, c, loss, next_idx)]
    
    best_structs_changed = True

    # search until the set of best structures doesn't change
    while best_structs_changed:

        # try to expand each current best structure
        new_structs = []
        best_structs_changed = False
        for xyz, c, loss, next_idx in best_structs:

            # evaluate each possible next atom
            for xyz_next_idx, c_next in zip(*next_idx):
                xyz_next = points[xyz_next_idx]

                # expand the stucture with the next atom
                xyz_new = torch.cat([xyz, xyz_next])
                c_new = torch.cat([c, c_next])

                # compute remaining density and loss
                xyz_new, density_pred, density_diff, loss_new = \
                    fit_atoms_by_GD(points, density_true, xyz_new, c_new, bonds, radii[c_new],
                                    interm_iters, lr=lr, mo=mo, lambda_E=lambda_E,
                                    radius_multiple=radius_multiple, verbose=verbose)

                # check if the structure is one of the best
                if any(loss_new < x[2] for x in best_structs):

                    # get next atom init locations
                    if atom_init == 'conv':

                        torch.nn.functional.conv3d()

                        conv = np.zeros_like(grid)
                        for i in range(n_channels):
                            conv[i] = conv_grid(density_diff[:,i].reshape(grid_shape), kernels[i])
                            conv[i] = np.roll(conv[i], np.array(grid_shape)//2, range(len(grid_shape)))
                        score = conv.reshape((n_channels, -1)).T

                    elif atom_init == 'deconv':
                        deconv = np.zeros_like(grid)
                        for i in range(n_channels):
                            deconv[i] = wiener_deconv_grid(density_diff[:,i].reshape(grid_shape), kernels[i])
                            deconv[i] = np.roll(deconv[i], np.array(grid_shape)//2, range(len(grid_shape)))
                        score = deconv.reshape((n_channels, -1)).T

                    else:
                        score = density_diff
                    next_idx_new = get_top_n_index(score, beam_size)

                    # keep track of the new structure
                    new_structs.append((xyz_new, c_new, loss_new, next_idx_new))

        # determine new set of best structures
        if new_structs: 
            best_structs += new_structs
            best_structs = sorted(best_structs, key=lambda x: x[2])[:beam_size]
            best_structs_changed = True

            # NOTE: When 0 < len(new_structs) < beam_size, not all
            # structs in best_structs will be different, so some
            # structs are expanded more than once.
            #
            # One way to handle this is would be to expand to diff
            # next atoms each time, by keeping track of next_idx
            # larger than beam_size, and counting the number of
            # times the structure has been expanded.

    while False:

        # optimize atom positions by gradient descent
        xyz, density_pred, density_diff, loss = \
            fit_atoms_by_GD(points, density, xyz, c, bonds, radii[c], interm_iters, lr=lr, mo=mo,
                            lambda_E=lambda_E, radius_multiple=radius_multiple, verbose=verbose,
                            density_pred=density_pred, density_diff=density_diff)

        if verbose > 1:
            print('n_atoms = {}\t\t\tloss = {}'.format(len(xyz), loss))

        # init next atom position on remaining density
        xyz_new = []
        c_new = []
        if fit_channels is not None:
            try:
                i = fit_channels[len(c)]
                conv = conv_grid(density_diff[:,i].reshape(grid_shape), kernels[i])
                conv = np.roll(conv, np.array(grid_shape)//2, range(len(grid_shape)))
                xyz_new.append(points[conv.argmax()])
                c_new.append(i)
            except IndexError:
                pass
        else:
            c_new = []
            for i in range(n_channels):
                conv = conv_grid(density_diff[:,i].reshape(grid_shape), kernels[i])
                conv = np.roll(conv, np.array(grid_shape)//2, range(len(grid_shape)))
                if np.any(conv > (kernels[i]**2).sum()/2): # check if L2 loss decreases
                    xyz_new.append(points[conv.argmax()])
                    c_new.append(i)

        # stop if a new atom was not added
        if not xyz_new:
            break

        xyz = np.vstack([xyz, xyz_new])
        c = np.append(c, c_new)

        if bonded: # add new bonds as row and column
            raise NotImplementedError('TODO add bonds_new')
            bonds = np.vstack([bonds, bonds_new])
            bonds = np.hstack([bonds, np.append(bonds_new, 0)])

    # get best structure
    xyz, c, loss, next_idx = best_structs[0]

    # final optimization
    xyz, density_pred, density_diff, loss = \
        fit_atoms_by_GD(points, density_true, xyz, c, bonds, radii[c],
                        final_iters, lr=lr, mo=mo, lambda_E=lambda_E,
                        radius_multiple=radius_multiple, verbose=verbose)

    grid_pred = density_pred.T.reshape(grid.shape)
    return xyz, c, bonds, grid_pred, loss, time.time() - t_start


def get_next_atom(points, density, xyz_init, c, atom_radius, bonded, bonds, max_n_bonds, max_init_bond_E=0.5):
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
        min_bond_length2 = bond_length2 - np.log(1 + np.sqrt(max_init_bond_E))
        max_bond_length2 = bond_length2 - np.log(1 - np.sqrt(max_init_bond_E))

    # can_bond[i] = xyz_init[i] has less than its max number of bonds
    can_bond = np.sum(bonds, axis=1) < max_n_bonds[c]

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
                    dist_min2 = min_bond_length2[c]
                    dist_max2 = max_bond_length2[c]
                else:
                    dist_min2 = atom_radius[c,np.newaxis]
                    dist_max2 = np.full_like(dist_min2, np.inf)

                # far_enough[i,j] = p is far enough from xyz_init[i] in channel[j]
                # near_enough[i,j] = p is near enough to xyz_init[i] in channel[j]
                far_enough = dist2[:,np.newaxis] > dist_min2
                near_enough = dist2[:,np.newaxis] < dist_max2

                # in_range[i] = p is far enough from all xyz_init and near_enough to
                # some xyz_init that can bond to make a bond in channel[i]
                in_range = np.all(far_enough, axis=0) & \
                           np.any(near_enough & can_bond[:,np.newaxis], axis=0)

                if np.any(in_range & more_density):
                    xyz_new = p
                    c_new = np.argmax(in_range*more_density*d)
                    if bonded:
                        bonds_new = near_enough[:,c_new] & can_bond
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


def combine_element_grids_and_channels(grids, channels):
    '''
    Return new grids and channels by combining channels
    of provided grids that are the same element.
    '''
    element_to_idx = dict()
    new_grid = []
    new_channels = []

    for grid, channel in zip(grids, channels):

        atomic_num = channel.atomic_num
        if atomic_num not in element_to_idx:

            element_to_idx[atomic_num] = len(element_to_idx)

            new_grid.append(np.zeros_like(grid))

            name = atom_types.get_name(atomic_num)
            symbol = channel.symbol
            atomic_radius = channel.atomic_radius

            new_channel = atom_types.channel(name, atomic_num, symbol, atomic_radius)
            new_channels.append(new_channel)

        new_grid[element_to_idx[atomic_num]] += grid

    return np.array(new_grid), new_channels


def write_pymol_script(pymol_file, dx_prefixes, struct_files, centers=[]):
    '''
    Write a pymol script with a map object for each of dx_files, a
    group of all map objects (if any), a rec_file, a lig_file, and
    an optional fit_file.
    '''
    with open(pymol_file, 'w') as f:
        for dx_prefix in dx_prefixes: # load densities
            dx_pattern = '{}_*.dx'.format(dx_prefix)
            dx_group = '{}_grids'.format(os.path.basename(dx_prefix))
            f.write('load_group {}, {}\n'.format(dx_pattern, dx_group))

        for struct_file in struct_files: # load structures
            obj_name = os.path.splitext(os.path.basename(struct_file))[0]
            m = re.match(r'^(.*_fit)_(\d+)$', obj_name)
            if m:
                obj_name = m.group(1)
                state = int(m.group(2)) + 1
                f.write('load {}, {}, state={}\n'.format(struct_file, obj_name, state))
            else:
                f.write('load {}, {}\n'.format(struct_file, obj_name))

        for struct_file, (x,y,z) in zip(struct_files, centers): # center structures
            obj_name = os.path.splitext(os.path.basename(struct_file))[0]
            f.write('translate [{},{},{}], {}, camera=0, state=0\n'.format(-x, -y, -z, obj_name))


def read_gninatypes_file(lig_file, channels):
    channel_names = [c.name for c in channels]
    channel_name_idx = {n: i for i, n in enumerate(channel_names)}
    xyz, c = [], []
    with open(lig_file, 'rb') as f:
        atom_bytes = f.read(16)
        while atom_bytes:
            x, y, z, t = struct.unpack('fffi', atom_bytes)
            smina_type = atom_types.smina_types[t]
            channel_name = 'Ligand' + smina_type.name
            if channel_name in channel_name_idx:
                c_ = channel_names.index(channel_name)
                xyz.append([x, y, z])
                c.append(c_)
            atom_bytes = f.read(16)
    assert xyz and c, lig_file
    return np.array(xyz), np.array(c)


def get_mol_center(mol):
    '''
    Compute the center of a molecule, ignoring hydrogen.
    '''
    return np.mean([a.coords for a in mol.atoms if a.atomicnum != 1], axis=0)


def get_n_atoms_from_sdf_file(sdf_file, idx=0):
    '''
    Count the number of atoms of each element in a molecule 
    from an .sdf file.
    '''
    mol = get_mols_from_sdf_file(sdf_file)[idx]
    return Counter(atom.GetSymbol() for atom in mol.GetAtoms())


def make_one_hot(x, n, dtype=None, device=None):
    y = torch.zeros(x.shape + (n,), dtype=dtype, device=device)
    for idx, last_idx in np.ndenumerate(x):
        y[idx + (int(last_idx),)] = 1
    return y


def write_structs_to_sdf_file(sdf_file, structs):
    mols = (s.to_ob_mol() for s in structs)
    write_ob_mols_to_sdf_file(sdf_file, mols)


def write_channels_to_file(channels_file, c, channels):
    with open(channels_file, 'w') as f:
        for c_ in c:
            channel = channels[c_]
            f.write(channel.name+'\n')


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
        n_bonds = 0
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                if bonds[i,j]:
                    n_bonds += 1
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
            for j in range(i+1, n_atoms):
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
    if len(grid.shape) != 3 or len(set(grid.shape)) != 1:
        raise ValueError('grid must have three equal dimensions')
    if len(center) != 3:
        raise ValueError('center must be a vector of length 3')
    dim = grid.shape[0]
    origin = np.array(center) - resolution*(dim-1)/2.
    with open(dx_file, 'w') as f:
        f.write('object 1 class gridpositions counts {:d} {:d} {:d}\n'.format(dim, dim, dim))
        f.write('origin {:.5f} {:.5f} {:.5f}\n'.format(*origin))
        f.write('delta {:.5f} 0 0\n'.format(resolution))
        f.write('delta 0 {:.5f} 0\n'.format(resolution))
        f.write('delta 0 0 {:.5f}\n'.format(resolution))
        f.write('object 2 class gridconnections counts {:d} {:d} {:d}\n'.format(dim, dim, dim))
        f.write('object 3 class array type double rank 0 items [ {:d} ] data follows\n'.format(dim**3))
        total = 0
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    f.write('{:.10f}'.format(grid[i][j][k]))
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
    for grid, channel in zip(grids, channels):
        dx_file = '{}_{}.dx'.format(out_prefix, channel.name)
        write_grid_to_dx_file(dx_file, grid, center, resolution)
        dx_files.append(dx_file)
    return dx_files


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


def read_examples_from_data_file(data_file, data_root=''):
    '''
    Read list of (rec_file, lig_file) examples from
    data_file, optionally prepended with data_root.
    '''
    examples = []
    with open(data_file, 'r') as f:
        for line in f:
            rec_file, lig_file = line.rstrip().split()[2:4]
            if data_root:
                rec_file = os.path.join(data_root, rec_file)
                lig_file = os.path.join(data_root, lig_file)
            examples.append((rec_file, lig_file))
    return examples


def find_blobs_in_net(net, blob_pattern):
    '''
    Find all blob_names in net that match blob_pattern.
    '''
    return re.findall('^{}$'.format(blob_pattern), '\n'.join(net.blobs), re.MULTILINE)


def get_layer_index(net, layer_name):
    return net._layer_names.index(layer_name)


def lazy_forward(net, layer):
    '''
    Compute the forward pass of a layer by recursively
    calling forward on each input layer.
    '''
    raise NotImplementedError

    stack = [layer]
    visited = set()
    for bottom in net.layer_dict[layer].layer_param.bottom:
        for layer in input_map[bottom]:
            if layer not in visited:
                lazy_forward(net, layer)
                visited.add(layer)


def count_types(c, n_types, dtype=None):
    count = np.zeros(n_types, dtype=dtype)
    for i in c:
        count[i] += 1
    return count


def get_min_rmsd(xyz1, c1, xyz2, c2):
    '''
    Compute an RMSD between two sets of positions of the same
    atom types with no prior mapping between particular atom
    positions of a given type. Returns the minimum RMSD across
    all permutations of this mapping.
    '''
    # check that structs are same size
    if len(c1) != len(c2):
        raise ValueError('structs must have same num atoms')
    n_atoms = len(c1)

    # copy everything into arrays
    xyz1 = np.array(xyz1)
    xyz2 = np.array(xyz2)
    c1 = np.array(c1)
    c2 = np.array(c2)

    # check that types are compatible
    idx1 = np.argsort(c1)
    idx2 = np.argsort(c2)
    c1 = c1[idx1]
    c2 = c2[idx2]
    if any(c1 != c2):
        raise ValueError('structs must have same num atoms of each type')
    xyz1 = xyz1[idx1]
    xyz2 = xyz2[idx2]

    # find min rmsd by solving linear sum assignment
    # problem on squared dist matrix for each type
    ssd = 0.0
    nax = np.newaxis
    for c in set(c1): 
        xyz1_c = xyz1[c1 == c]
        xyz2_c = xyz2[c2 == c]
        dist2_c = ((xyz1_c[:,nax,:] - xyz2_c[nax,:,:])**2).sum(axis=2)
        idx1, idx2 = sp.optimize.linear_sum_assignment(dist2_c)
        ssd += dist2_c[idx1, idx2].sum()

    return np.sqrt(ssd/n_atoms)


def generate_from_model(gen_net, data_param, n_examples, args):
    '''
    Generate grids from specific blob(s) in gen_net for each
    ligand in examples, and possibly do atom fitting.
    '''
    device = ('cpu', 'cuda')[args.gpu]
    batch_size = gen_net.blobs['lig'].shape[0]

    rec_map = molgrid.FileMappedGninaTyper(data_param.recmap)
    lig_map = molgrid.FileMappedGninaTyper(data_param.ligmap)
    rec_channels = atom_types.get_channels_from_map(rec_map, name_prefix='Receptor')
    lig_channels = atom_types.get_channels_from_map(lig_map, name_prefix='Ligand')

    ex_provider = molgrid.ExampleProvider(rec_map, lig_map, data_root=args.data_root)
    ex_provider.populate(data_param.source)

    grid_maker = molgrid.GridMaker(data_param.resolution, data_param.dimension)
    grid_dims = grid_maker.grid_dimensions(rec_map.num_types() + lig_map.num_types())
    grid_true = torch.zeros(batch_size, *grid_dims, dtype=torch.float32, device=device)

    try: # find latent variable blobs for VAE
        latent_mean = find_blobs_in_net(gen_net, r'.+_latent_mean')[0]
        latent_std = find_blobs_in_net(gen_net, r'.+_latent_std')[0]
        latent_noise = find_blobs_in_net(gen_net, r'.+_latent_noise')[0]
        latent_sample = find_blobs_in_net(gen_net, r'.+_latent_sample')[0]
        variational = True
    except IndexError: # for standard AE
        latent_sample = find_blobs_in_net(gen_net, r'.+_latent_defc')[0]
        variational = False

    n_latent = gen_net.blobs[latent_sample].shape[1]

    # find first decoder blob
    dec_fc = find_blobs_in_net(gen_net, r'.+_dec_fc')[0]

    gen_net.forward() # this is necessary for proper latent sampling

    if args.parallel: # compute metrics and write output in a separate thread

        out_queue = mp.Queue()
        out_thread = threading.Thread(
            target=out_worker_main,
            args=(out_queue, args))
        out_thread.start()

        if args.fit_atoms: # fit atoms to grids in separate processes
            fit_queue = mp.Queue(args.n_fit_workers) # queue for atom fitting
            fit_procs = mp.Pool(
                processes=args.n_fit_workers,
                initializer=fit_worker_main,
                initargs=(fit_queue, out_queue, args))

    else: # compute metrics, write output, and fit atoms in single thread
        output = OutputWriter(args.out_prefix, args.output_dx, args.output_sdf, args.output_channels,
                              args.n_samples, args.blob_name, args.fit_atoms, verbose=args.verbose)

        if args.fit_atoms:
            fitter = AtomFitter(args.beam_size, args.beam_stride, args.atom_init, args.interm_iters, args.final_iters,
                                args.learning_rate, args.beta1, args.beta2, args.weight_decay,
                                args.constrain_types, args.r_factor, args.output_visited, args.output_kernel,
                                device=device, verbose=args.verbose)

    # generate density grids from generative model in main thread
    try:
        for example_idx in range(n_examples):

            for sample_idx in range(args.n_samples):

                # keep track of position in batch
                batch_idx = (example_idx*args.n_samples + sample_idx) % batch_size
                first_sample_idx = batch_idx - batch_idx%args.n_samples

                if batch_idx == 0: # forward next batch

                    # get next batch of structures
                    examples = ex_provider.next_batch(batch_size)

                    # convert structures to grids
                    for i, ex in enumerate(examples):
                        transform = molgrid.Transform(ex.coord_sets[1].center(),
                                                      args.random_translate,
                                                      args.random_rotation)
                        transform.forward(ex, ex)
                        grid_maker.forward(ex, grid_true[i])

                    rec = grid_true[:,:rec_map.num_types(),...].cpu()
                    lig = grid_true[:,rec_map.num_types():,...].cpu()

                    need_first = (args.encode_first or args.condition_first)
                    is_first = (example_idx == sample_idx == 0)

                    if need_first and is_first:
                        first_rec = np.array(rec[0:1])
                        first_lig = np.array(lig[0:1])

                    # set encoder input grids
                    if args.encode_first:
                        gen_net.blobs['rec'].data[...] = first_rec
                        gen_net.blobs['lig'].data[...] = first_lig
                    else:
                        gen_net.blobs['rec'].data[...] = rec
                        gen_net.blobs['lig'].data[...] = lig

                    # set conditional input grids
                    if 'cond_rec' in gen_net.blobs:
                        if args.condition_first:
                            gen_net.blobs['cond_rec'].data[...] = first_rec
                        else:
                            gen_net.blobs['cond_rec'].data[...] = rec

                    if args.interpolate: # copy grids that will be interpolated
                        for i in range(0, batch_size, args.n_samples):
                            gen_net.blobs['lig'].data[i:i+args.n_samples] = \
                                gen_net.blobs['lig'].data[i:i+1]
                            gen_net.blobs['rec'].data[i:i+args.n_samples] = \
                                gen_net.blobs['rec'].data[i:i+1]

                    # encode true grids to latent variable parameters
                    if variational:
                        if args.prior:
                            if args.mean:
                                gen_net.blobs[latent_mean].data[...] = 0.0
                                gen_net.blobs[latent_std].data[...]  = 0.0
                            else:
                                gen_net.blobs[latent_mean].data[...] = 0.0
                                gen_net.blobs[latent_std].data[...]  = 1.0
                        else: # posterior
                            if args.mean:
                                gen_net.forward(end=latent_mean)
                                gen_net.blobs[latent_std].data[...] = 0.0
                            else:
                                gen_net.forward(end=latent_std)

                        # sample latent variables
                        gen_net.forward(start=latent_noise, end=latent_sample)

                    else:
                        if args.prior:
                            gen_net.blobs[latent_sample] = np.random.randn(batch_size, n_latent)
                        else:
                            gen_net.forward(end=latent_sample)

                    if args.interpolate: # interpolate between latent samples
                        latent = np.array(gen_net.blobs[latent_sample].data)
                        for i in range(0, batch_size, args.n_samples):
                            j = (i + args.n_samples) % batch_size
                            gen_net.blobs[latent_sample].data[i:i+args.n_samples] = \
                                np.linspace(latent[i], latent[j], args.n_samples)

                    # decode latent samples to generate grids
                    gen_net.forward(start=dec_fc)

                # get current example ligand
                if args.interpolate:
                    ex = examples[first_sample_idx]
                else:
                    ex = examples[batch_idx]
                lig_coord_set = ex.coord_sets[1]

                # get lig source file and name
                lig_base = os.path.splitext(lig_coord_set.src)[0]
                lig_name = os.path.basename(lig_base)
                m = re.match(r'(.+)_ligand_(\d+)', lig_base)
                if m:
                    lig_base = m.group(1) + '_docked.sdf.gz'
                    idx = int(m.group(2))
                else:
                    lig_base = lig_base + '.sdf'
                    idx = 0
                lig_file = os.path.join(args.data_root, lig_base)

                # get true ligand source mol, struct, and types
                mol = read_rd_mols_from_sdf_file(lig_file)[idx]
                struct = MolStruct.from_coord_set(lig_coord_set, lig_channels, src_mol=mol)
                types = count_types(struct.c, lig_map.num_types(), dtype=np.int16)

                for blob_name in args.blob_name: # get data from blob and add to appropriate output

                    grid_blob = gen_net.blobs[blob_name]

                    if args.interpolate and blob_name in {'rec', 'lig'}:
                        grid_data = grid_blob.data[first_sample_idx]
                    else:
                        grid_data = grid_blob.data[batch_idx]

                    grid = MolGrid(values=np.array(grid_data),
                                   channels=struct.channels,
                                   center=struct.center,
                                   resolution=grid_maker.get_resolution())

                    grid_name = blob_name
                    grid_norm = np.linalg.norm(grid.values)

                    if grid_name == 'lig_gen':
                        latent_blob = gen_net.blobs[latent_sample]
                        latent_data = np.array(latent_blob.data)[batch_idx]
                        grid.info['latent_norm'] = np.linalg.norm(latent_data)

                    if args.verbose:
                        try:
                            gpu_usage = getGPUs()[0].memoryUtil
                        except:
                            gpu_usage = np.nan

                        print('main_thread produced {} {} {} (norm={}\tGPU={})'
                              .format(lig_name, grid_name.ljust(7), sample_idx, grid_norm, gpu_usage), flush=True)

                    if args.fit_atoms and blob_name.startswith('lig'):
                        if args.parallel:
                            fit_queue.put((lig_name, grid_name, sample_idx, grid, struct))
                        else:
                            output.write(lig_name, grid_name, sample_idx, grid, struct)
                            grid_fit, struct_fit = fitter.fit(grid, types, use_r_factor='gen' in grid_name)
                            output.write(lig_name, grid_name+'_fit', sample_idx, grid_fit, struct_fit)
                    else:
                        if args.parallel:
                            out_queue.put((lig_name, grid_name, sample_idx, grid, struct))
                        else:
                            output.write(lig_name, grid_name, sample_idx, grid, struct)
    finally:
        if args.parallel:

            if args.fit_atoms:
                #for i in range(args.n_fit_workers):
                #    fit_queue.put(None)
                fit_procs.close()
                fit_procs.join()

            out_queue.put(None)
            out_thread.join()

    if args.verbose:
        print('main_thread exit')


def fit_worker_main(fit_queue, out_queue, args):

    fitter = AtomFitter(args.beam_size, args.beam_stride, args.atom_init, args.interm_iters, args.final_iters,
                        args.learning_rate, args.beta1, args.beta2, args.weight_decay,
                        args.constrain_types, args.r_factor, args.output_visited, args.output_kernel,
                        device='cpu', verbose=args.verbose)
    while True:

        if args.verbose:
            print('fit_worker waiting')
        task = fit_queue.get()
        if task is None:
            break

        lig_name, grid_name, sample_idx, grid, struct = task
        if args.verbose:
            print('fit_worker got {} {} {}'.format(lig_name, grid_name, sample_idx))

        out_queue.put((lig_name, grid_name, sample_idx, grid, struct))

        types = count_types(struct.c, len(struct.channels), dtype=int16)
        grid_fit, struct_fit = fitter.fit(grid, types, use_r_factor='gen' in grid_name)
        grid_name += '_fit'
        if args.verbose:
            print('fit_worker produced {} {} {} ({} atoms, {}s)'
                  .format(lig_name, grid_name, sample_idx, struct_fit.n_atoms, struct_fit.fit_time))

        out_queue.put((lig_name, grid_name, sample_idx, grid_fit, struct_fit))

    if args.verbose:
        print('fit_worker exit')


def out_worker_main(out_queue, args):

    output = OutputWriter(args.out_prefix, args.output_dx, args.output_sdf, args.output_channels,
                          args.n_samples, args.blob_name, args.fit_atoms, verbose=args.verbose)
    while True:
        task = out_queue.get()
        if task is None:
            break
        output.write(*task)

    if args.verbose:
        print('out_worker exit')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Generate atomic density grids from generative model with Caffe')
    parser.add_argument('-d', '--data_model_file', required=True, help='prototxt file for data model')
    parser.add_argument('-g', '--gen_model_file', required=True, help='prototxt file for generative model')
    parser.add_argument('-w', '--gen_weights_file', default=None, help='.caffemodel weights for generative model')
    parser.add_argument('-r', '--rec_file', default=[], action='append', help='receptor file (relative to data_root)')
    parser.add_argument('-l', '--lig_file', default=[], action='append', help='ligand file (relative to data_root)')
    parser.add_argument('--data_file', default='', help='path to data file (generate for every example)')
    parser.add_argument('--data_root', default='', help='path to root for receptor and ligand files')
    parser.add_argument('-b', '--blob_name', default=[], action='append', help='blob(s) in model to generate from (default lig & lig_gen)')
    parser.add_argument('--n_samples', default=1, type=int, help='number of samples to generate for each input example')
    parser.add_argument('--prior', default=False, action='store_true', help='generate from prior instead of posterior distribution')
    parser.add_argument('--mean', default=False, action='store_true', help='generate mean of distribution instead of sampling')
    parser.add_argument('--encode_first', default=False, action='store_true', help='generate all output from encoding first example')
    parser.add_argument('--condition_first', default=False, action='store_true', help='condition all generated output on first example')
    parser.add_argument('--interpolate', default=False, action='store_true', help='interpolate between examples in latent space')
    parser.add_argument('-o', '--out_prefix', required=True, help='common prefix for output files')
    parser.add_argument('--output_dx', action='store_true', help='output .dx files of atom density grids for each channel')
    parser.add_argument('--output_sdf', action='store_true', help='output .sdf file of best fit atom positions')
    parser.add_argument('--output_visited', action='store_true', help='output every visited structure in .sdf files')
    parser.add_argument('--output_kernel', action='store_true', help='output .dx files for kernel used to intialize atoms during atom fitting')
    parser.add_argument('--output_channels', action='store_true', help='output channels of each fit structure in separate files')
    parser.add_argument('--fit_atoms', action='store_true', help='fit atoms to density grids and print the goodness-of-fit')
    parser.add_argument('--constrain_types', action='store_true', help='constrain atom fitting to use atom types of true ligand')
    parser.add_argument('--r_factor', type=float, default=1.0, help='radius multiple for fitting to generated grids')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate for Adam optimizer')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay for Adam optimizer')
    parser.add_argument('--beam_size', type=int, default=1, help='value of beam size N for atom fitting search')
    parser.add_argument('--beam_stride', type=int, default=1, help='stride of atom fitting beam search')
    parser.add_argument('--atom_init', type=str, default='none', help='function to apply to remaining density before atom init (none|conv|deconv)')
    parser.add_argument('--interm_iters', type=int, default=10, help='maximum number of iterations for atom fitting between atom inits')
    parser.add_argument('--final_iters', type=int, default=100, help='maximum number of iterations for atom fitting after atom inits')
    parser.add_argument('--lambda_E', type=float, default=0.0, help='interatomic bond energy loss weight for gradient descent atom fitting')
    parser.add_argument('--bonded', action='store_true', help="add atoms by creating bonds to existing atoms when atom fitting")
    parser.add_argument('--max_init_bond_E', type=float, default=0.5, help='maximum energy of bonds to consider when adding bonded atoms')
    parser.add_argument('--fit_GMM', action='store_true', help='fit atoms by a Gaussian mixture model instead of gradient descent')
    parser.add_argument('--noise_model', default='', help='noise model for GMM atom fitting (d|p)')
    parser.add_argument('--deconv_grids', action='store_true', help="apply Wiener deconvolution to atom density grids")
    parser.add_argument('--deconv_fit', action='store_true', help="apply Wiener deconvolution for atom fitting initialization")
    parser.add_argument('--noise_ratio', default=1.0, type=float, help="noise-to-signal ratio for Wiener deconvolution")
    parser.add_argument('--verbose', default=0, type=int, help="verbose output level")
    parser.add_argument('--gpu', action='store_true', help="generate grids from model on GPU")
    parser.add_argument('--random_rotation', default=False, action='store_true', help='randomly rotate input before generating grids')
    parser.add_argument('--random_translate', default=0.0, type=float, help='randomly translate up to #A before generating grids')
    parser.add_argument('--batch_rotate', default=False, action='store_true')
    parser.add_argument('--batch_rotate_yaw', type=float)
    parser.add_argument('--batch_rotate_roll', type=float)
    parser.add_argument('--batch_rotate_pitch', type=float)
    parser.add_argument('--fix_center_to_origin', default=False, action='store_true', help='fix input grid center to origin')
    parser.add_argument('--use_covalent_radius', default=False, action='store_true', help='force input grid to use covalent radius')
    parser.add_argument('--parallel', default=False, action='store_true', help='run atom fitting in separate worker processes')
    parser.add_argument('--n_fit_workers', default=8, type=int, help='number of worker processes for parallel atom fitting')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_colwidth', 100)
    try:
        display_width = get_terminal_size()[1]
    except:
        display_width = 185
    pd.set_option('display.width', display_width)

    if not args.blob_name:
        args.blob_name += ['lig', 'lig_gen']

    # read the model param files and set atom gridding params
    data_net_param = caffe_util.NetParameter.from_prototxt(args.data_model_file)
    gen_net_param = caffe_util.NetParameter.from_prototxt(args.gen_model_file)

    data_param = data_net_param.get_molgrid_data_param(caffe.TEST)
    data_param.random_rotation = args.random_rotation
    data_param.random_translate = args.random_translate
    data_param.fix_center_to_origin = args.fix_center_to_origin
    data_param.use_covalent_radius = args.use_covalent_radius
    data_param.shuffle = False
    data_param.balanced = False

    if args.batch_rotate_yaw:
        data_param.batch_rotate = True
        data_param.batch_rotate_yaw = 2*np.pi/data_param.batch_size

    if args.batch_rotate_pitch:
        data_param.batch_rotate = True
        data_param.batch_rotate_pitch = 2*np.pi/data_param.batch_size

    if args.batch_rotate_roll:
        data_param.batch_rotate = True
        data_param.batch_rotate_roll = 2*np.pi/data_param.batch_size

    if not args.data_file: # use the set of (rec_file, lig_file) examples
        assert len(args.rec_file) == len(args.lig_file)
        examples = list(zip(args.rec_file, args.lig_file))

    else: # use the examples in data_file
        examples = read_examples_from_data_file(args.data_file)

    data_file = get_temp_data_file(e for e in examples for i in range(args.n_samples))
    data_param.source = data_file

    # create the net in caffe
    caffe.set_mode_gpu()
    gen_net = caffe_util.Net.from_param(gen_net_param, args.gen_weights_file, phase=caffe.TEST)

    generate_from_model(gen_net, data_param, len(examples), args)


if __name__ == '__main__':
    main(sys.argv[1:])
