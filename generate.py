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
import torch.nn.functional as F
from GPUtil import getGPUs
from openbabel import openbabel as ob
from openbabel import pybel
import rdkit
from rdkit import Chem, Geometry, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED, Crippen
from rdkit.Chem.Fingerprints import FingerprintMols
from SA_Score import sascorer
from NP_Score import npscorer
nps_model = npscorer.readNPModel()

import molgrid
import caffe_util
import atom_types
import simple_fit
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
        return (size - 1) * resolution

    @classmethod
    def compute_size(cls, dimension, resolution):
        return int(np.ceil(dimension / resolution + 1))

    def to_dx(self, dx_prefix, center=None):
        write_grids_to_dx_files(
            out_prefix=dx_prefix,
            grids=self.values,
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
            raise ValueError('invalid channel index in MolStruct c')

        self.n_atoms = xyz.shape[0]
        self.xyz = xyz
        self.c = c
        self.channels = channels

        if bonds is not None:
            if bonds.shape != (self.n_atoms, self.n_atoms):
                raise ValueError('MolStruct bonds must have shape (n_atoms, n_atoms)')
            self.bonds = bonds
        else:
            self.bonds = np.zeros((self.n_atoms, self.n_atoms))

        if self.n_atoms > 0:
            self.center = self.xyz.mean(0)
            self.radius = max(np.linalg.norm(self.xyz - self.center, axis=1))
        else:
            self.center = np.full(3, np.nan)
            self.radius = np.nan

        self.info = info

    @classmethod
    def from_gninatypes(self, gtypes_file, channels, **info):
        xyz, c = read_gninatypes_file(gtypes_file, channels)
        return MolStruct(xyz, c, channels, **info)

    @classmethod
    def from_coord_set(self, coord_set, channels, **info):
        if not coord_set.has_indexed_types():
            raise ValueError(
                'can only make MolStruct from CoordinateSet with indexed types'
            )
        xyz = coord_set.coords.tonumpy()
        c = coord_set.type_index.tonumpy().astype(int)
        return MolStruct(xyz, c, channels, **info)

    def to_ob_mol(self):
        mol = make_ob_mol(self.xyz.astype(float), self.c, self.bonds, self.channels)
        return mol

    def to_rd_mol(self):
        mol = make_rd_mol(self.xyz.astype(float), self.c, self.bonds, self.channels)
        return mol

    def to_sdf(self, sdf_file):
        write_rd_mols_to_sdf_file(sdf_file, [self.to_rd_mol()])

    def add_bonds(self, tol=0.0):

        nax = np.newaxis
        channel_radii = np.array([c.atomic_radius for c in self.channels])

        atom_dist2 = ((self.xyz[nax,:,:] - self.xyz[:,nax,:])**2).sum(axis=2)
        max_bond_dist2 = channel_radii[self.c][nax,:] + channel_radii[self.c][:,nax]
        self.bonds = (atom_dist2 < max_bond_dist2 + tol)


class AtomFitter(object):

    def __init__(
        self,
        multi_atom,
        beam_size,
        apply_conv,
        threshold,
        peak_value,
        min_dist,
        constrain_types,
        constrain_frags,
        estimate_types,
        interm_gd_iters,
        final_gd_iters,
        gd_kwargs,
        dkoes_make_mol,
        output_kernel,
        device,
        verbose=0,
    ):

        # can place all detected atoms at once, or do beam search
        self.multi_atom = multi_atom
        self.beam_size = beam_size

        # settings for detecting next atoms to place on density grid
        self.apply_conv = apply_conv
        self.threshold = threshold
        self.peak_value = peak_value
        self.min_dist = min_dist

        # can constrain to find exact atom type counts or single fragment
        self.constrain_types = constrain_types
        self.constrain_frags = constrain_frags
        self.estimate_types = estimate_types

        # can perform gradient descent at each step and/or at final step
        self.interm_gd_iters = interm_gd_iters
        self.final_gd_iters = final_gd_iters
        self.gd_kwargs = gd_kwargs

        # use alternate bond adding method
        self.dkoes_make_mol = dkoes_make_mol

        self.output_kernel = output_kernel
        self.device = device
        self.verbose = verbose

        self.grid_maker = molgrid.GridMaker()
        self.c2grid = molgrid.Coords2Grid(self.grid_maker)
        self.kernel = None

    def init_kernel(self, channels, resolution, deconv=False):
        '''
        Initialize the convolution kernel that is used
        to propose next atom initializations on grids.
        '''
        n_channels = len(channels)

        # kernel is created by computing a molgrid from a
        # struct with one atom of each type at the center
        xyz = torch.zeros((n_channels, 3), device=self.device)
        c = torch.eye(n_channels, device=self.device) # one-hot vector types
        r = torch.tensor([ch.atomic_radius for ch in channels], device=self.device)

        # kernel must fit max radius atom
        kernel_radius = 1.5 * max(r).item()

        self.grid_maker.set_radii_type_indexed(True)
        self.grid_maker.set_dimension(2 * kernel_radius) 
        self.grid_maker.set_resolution(resolution)

        self.c2grid.center = (0.0, 0.0, 0.0)
        values = self.c2grid(xyz, c, r)

        if deconv:
            values = torch.tensor(
                weiner_invert_kernel(values.cpu(), noise_ratio=1),
                dtype=values.dtype,
                device=self.device,
            )

        self.kernel = MolGrid(
            values=values,
            channels=channels,
            center=torch.zeros(3, device=self.device),
            resolution=resolution,
        )

        if self.output_kernel:
            dx_prefix = 'deconv_kernel' if deconv else 'conv_kernel'
            if self.verbose:
                kernel_norm = np.linalg.norm(values.cpu())
                print('writing out {} (norm={})'.format(dx_prefix, kernel_norm))
            self.kernel.to_dx(dx_prefix)
            self.output_kernel = False # only write once

    def get_next_atoms(self, grid, channels, center, resolution, types=None):
        '''
        Get a set of atoms from a density grid by convolving
        with a kernel, applying a threshold, and then returning
        atom types and coordinates ordered by grid value.
        '''
        assert len(grid.shape) == 4
        assert len(set(grid.shape[1:])) == 1

        n_channels = grid.shape[0]
        grid_dim = grid.shape[1]

        apply_peak_value = self.peak_value is not None and self.peak_value < np.inf
        apply_threshold = self.threshold is not None and self.threshold > -np.inf
        suppress_non_max = self.min_dist is not None and self.min_dist > 0.0

        if apply_peak_value:
            peak_value = torch.full((n_channels,), self.peak_value, device=self.device)

        if apply_threshold:
            threshold = torch.full((n_channels,), self.threshold, device=self.device)

        if self.apply_conv: # convolve grid with kernel

            if self.kernel is None:
                self.init_kernel(channels, resolution)

            grid = F.conv3d(
                input=grid.unsqueeze(0),
                weight=self.kernel.values.unsqueeze(1),
                padding=self.kernel.values.shape[-1]//2,
                groups=n_channels,
            )[0]

            kernel_norm2 = (self.kernel.values**2).sum(dim=(1,2,3))

            if apply_peak_value:
                peak_value *= kernel_norm2

            if apply_threshold:
                threshold *= kernel_norm2

        # reflect grid values above peak value
        if apply_peak_value:
            peak_value = peak_value.view(n_channels, 1, 1, 1)
            grid = peak_value - (peak_value - grid).abs()

        # sort grid points by value
        values, idx = torch.sort(grid.flatten(), descending=True)

        # convert flattened grid index to channel and spatial index
        idx_z, idx = idx % grid_dim, idx // grid_dim
        idx_y, idx = idx % grid_dim, idx // grid_dim
        idx_x, idx = idx % grid_dim, idx // grid_dim
        idx_c, idx = idx % n_channels, idx // n_channels
        idx_xyz = torch.stack((idx_x, idx_y, idx_z), dim=1)

        # apply threshold to grid values
        if apply_threshold:
            above_thresh = values > threshold[idx_c]
            values = values[above_thresh]
            idx_xyz = idx_xyz[above_thresh]
            idx_c = idx_c[above_thresh]

        # exclude grid channels with no atoms left
        if self.constrain_types:
            has_atoms_left = types[idx_c] > 0
            values = values[has_atoms_left]
            idx_xyz = idx_xyz[has_atoms_left]
            idx_c = idx_c[has_atoms_left]

            #TODO this does not constrain the atoms types correctly
            # when doing multi_atom fitting, because it only omits
            # atom types that have 0 atoms left- i.e. we could still
            # return 2 atoms of a type that only has 1 atom left.
            # Need to exclude all atoms of type t beyond rank n_t
            # where n_t is the number of atoms left of type t

        # convert spatial index to atom coordinates
        origin = center - resolution * (float(grid_dim) - 1) / 2.0
        xyz = origin + resolution * idx_xyz.float()

        # suppress atoms too close to a higher-value point
        if suppress_non_max:

            r = torch.tensor(
                [ch.atomic_radius for ch in channels],
                device=self.device,
            )
            if len(idx_c) > 1000:

                xyz_max = xyz[0].unsqueeze(0)
                idx_c_max = idx_c[0].unsqueeze(0)

                for i in range(1, len(idx_c)):
                    bond_radius = r[idx_c[i]] + r[idx_c_max]
                    min_dist = self.min_dist * bond_radius
                    min_dist2 = min_dist**2
                    dist2 = ((xyz[i].unsqueeze(0) - xyz_max)**2).sum(dim=1)
                    if not (dist2 < min_dist2).any():
                        xyz_max = torch.cat([xyz_max, xyz[i].unsqueeze(0)])
                        idx_c_max = torch.cat([idx_c_max, idx_c[i].unsqueeze(0)])

                xyz = xyz_max
                idx_c = idx_c_max

            else:
                bond_radius = r[idx_c].unsqueeze(1) + r[idx_c].unsqueeze(0)
                min_dist = self.min_dist * bond_radius
                min_dist2 = min_dist**2
                dist2 = ((xyz.unsqueeze(1) - xyz.unsqueeze(0))**2).sum(dim=2)
                not_too_close = ~torch.tril(dist2 < min_dist2, diagonal=-1).any(dim=1)
                xyz = xyz[not_too_close]
                idx_c = idx_c[not_too_close]

        # limit number of atoms to beam size
        if not self.multi_atom:
            xyz = xyz[:self.beam_size]
            idx_c = idx_c[:self.beam_size]

        # convert atom type channel index to one-hot type vector
        c = F.one_hot(idx_c, n_channels).to(dtype=torch.float32, device=self.device)

        return xyz.detach(), c.detach()

    def get_types_estimate(self, grid, channels, resolution):
        '''
        Since atom density is additive and non-negative, estimate
        the atom type counts by dividing the total density in each
        grid channel by the total density in each kernel channel.
        '''
        if self.kernel is None:
            self.init_kernel(channels, resolution)

        kernel_sum = self.kernel.values.sum(dim=(1,2,3))
        grid_sum = grid.sum(dim=(1,2,3))
        return grid_sum / kernel_sum

    def fit(self, grid, types):
        '''
        Fit atomic structure to mol grid.
        '''
        t_start = time.time()

        # get true grid on appropriate device
        grid_true = MolGrid(
            values=torch.as_tensor(grid.values, device=self.device),
            channels=grid.channels,
            center=torch.as_tensor(grid.center, device=self.device),
            resolution=grid.resolution,
        )

        # get true atom type counts on appropriate device
        types = torch.tensor(types, dtype=torch.float32, device=self.device)

        if self.estimate_types: # estimate atom type counts from grid density
            types_est = self.get_types_estimate(
                grid_true.values,
                grid_true.channels,
                grid_true.resolution,
            )
            est_type_loss = (types - types_est).abs().sum().item()
            types = types_est
        else:
            est_type_loss = np.nan

        # initialize empty struct
        print('initializing empty struct 0')
        n_channels = len(grid.channels)
        xyz = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        c = torch.zeros((0, n_channels), dtype=torch.float32, device=self.device)

        L2_loss = (grid_true.values**2).sum() / 2.0
        type_loss = types.abs().sum()

        # to constrain types, order structs first by type diff, then by L2 loss
        if self.constrain_types:
            objective = (type_loss.item(), L2_loss.item())

        else: # otherwise, order structs only by L2 loss
            objective = L2_loss.item()

        # get next atom init locations and channels
        print('getting next atoms for struct 0')
        xyz_next, c_next = self.get_next_atoms(
            grid_true.values,
            grid_true.channels,
            grid_true.center,
            grid_true.resolution,
            types,
        )

        # keept track of best structures so far
        struct_id = 0
        best_structs = [(objective, struct_id, xyz, c, xyz_next, c_next)]
        found_new_best_struct = True

        # keep track of visited structures
        visited = set()
        visited_structs = []
        struct_count = 1

        # search until we can't find a better structure
        while found_new_best_struct:

            new_best_structs = []
            found_new_best_struct = False

            # try to expand each current best structure
            for objective, struct_id, xyz, c, xyz_next, c_next in best_structs:

                if struct_id in visited:
                    continue

                print('expanding struct {} to {} next atoms'.format(struct_id, len(c_next)))

                if self.multi_atom: # evaluate all next atoms simultaneously

                    xyz_new = torch.cat([xyz, xyz_next])
                    c_new = torch.cat([c, c_next])

                    # compute diff and loss after gradient descent
                    xyz_new, grid_pred, grid_diff, L2_loss = self.fit_gd(
                        grid_true, xyz_new, c_new, self.interm_gd_iters
                    )

                    type_diff = types - c_new.sum(dim=0)
                    type_loss = type_diff.abs().sum()

                    if self.constrain_types:
                        objective_new = (type_loss.item(), L2_loss.item())
                    else:
                        objective_new = L2_loss.item()

                    # check if new structure is one of the best yet
                    if any(objective_new < s[0] for s in best_structs):

                        print('found new best as struct {}'.format(struct_count))

                        xyz_new_next, c_new_next = self.get_next_atoms(
                            grid_diff,
                            grid_true.channels,
                            grid_true.center,
                            grid_true.resolution,
                            type_diff,
                        )
                        new_best_structs.append(
                            (
                                objective_new,
                                struct_count,
                                xyz_new,
                                c_new,
                                xyz_new_next,
                                c_new_next,
                            )
                        )
                        found_new_best_struct = True
                        struct_count += 1

                else: # evaluate each possible next atom individually

                    for xyz_next_, c_next_ in zip(xyz_next, c_next):

                        # add next atom to structure
                        xyz_new = torch.cat([xyz, xyz_next_.unsqueeze(0)])
                        c_new = torch.cat([c, c_next_.unsqueeze(0)])

                        # compute diff and loss after gradient descent
                        xyz_new, grid_pred, grid_diff, L2_loss = self.fit_gd(
                            grid_true, xyz_new, c_new, self.interm_gd_iters
                        )

                        type_diff = types - c_new.sum(dim=0)
                        type_loss = type_diff.abs().sum()

                        if self.constrain_types:
                            objective_new = (type_loss.item(), L2_loss.item())
                        else:
                            objective_new = L2_loss.item()

                        # check if new structure is one of the best yet
                        if any(objective_new < s[0] for s in best_structs):

                            xyz_new_next, c_new_next = self.get_next_atoms(
                                grid_diff,
                                grid_true.channels,
                                grid_true.center,
                                grid_true.resolution,
                                type_diff,
                            )
                            new_best_structs.append(
                                (
                                    objective_new,
                                    struct_count,
                                    xyz_new,
                                    c_new,
                                    xyz_new_next,
                                    c_new_next,
                                )
                            )
                            found_new_best_struct = True
                            struct_count += 1

                visited.add(struct_id)
                visited_structs.append(
                    (objective, struct_id, time.time()-t_start, xyz, c)
                )

            if found_new_best_struct: # determine new set of best structures

                if self.multi_atom:
                    best_structs = sorted(best_structs + new_best_structs)[:1]
                else:
                    best_structs = sorted(best_structs + new_best_structs)[:self.beam_size]

                best_objective = best_structs[0][0]
                best_id = best_structs[0][1]
                best_n_atoms = best_structs[0][2].shape[0]

                if self.verbose:
                    try:
                        gpu_usage = getGPUs()[0].memoryUtil
                    except:
                        gpu_usage = np.nan
                    print(
                        'best struct # {} (objective={}, n_atoms={}, GPU={})'.format(
                            best_id, best_objective, best_n_atoms, gpu_usage
                        )
                    )

        # done searching for atomic structures
        best_objective, best_id, xyz_best, c_best, _, _ = best_structs[0]
        type_loss = (types - c_best.sum(dim=0)).abs().sum().item()

        if self.final_gd_iters > 0: # perform final gradient descent

            xyz_best, grid_pred, grid_diff, L2_loss = self.fit_gd(
                grid_true, xyz_best, c_best, self.final_gd_iters
            )

            if self.constrain_types:
                best_objective = (type_loss.item(), L2_loss.item())
            else:
                best_objective = L2_loss.item()

            visited_structs.append(
                (best_objective, best_id+1, time.time()-t_start, xyz_best, c_best)
            )

        # finalize the visited atomic structures
        visited_structs_ = iter(visited_structs)
        visited_structs = []
        for objective, struct_id, fit_time, xyz, c in visited_structs_:

            struct = MolStruct(
                xyz=xyz.cpu().detach().numpy(),
                c=one_hot_to_index(c).cpu().detach().numpy(),
                channels=grid.channels,
                L2_loss=L2_loss,
                type_diff=type_diff,
                est_type_diff=est_type_loss,
                time=fit_time,
            )
            visited_structs.append(struct)

        # finalize the best fit atomic structure and density grid
        struct_best = MolStruct(
            xyz=xyz_best.cpu().detach().numpy(),
            c=one_hot_to_index(c_best).cpu().detach().numpy(),
            channels=grid.channels,
            L2_loss=L2_loss,
            type_diff=type_loss,
            est_type_diff=est_type_loss,
            time=time.time()-t_start,
        )
        self.validify(struct_best)

        grid_pred = MolGrid(
            values=grid_pred.cpu().detach().numpy(),
            channels=grid.channels,
            center=grid.center,
            resolution=grid.resolution,
            visited_structs=visited_structs,
            src_struct=struct_best,
        )

        return grid_pred

    def fit_gd(self, grid, xyz, c, n_iters):

        r = torch.tensor(
            [ch.atomic_radius for ch in grid.channels],
            device=self.device,
        )

        xyz = xyz.clone().detach().to(self.device)
        c = c.clone().detach().to(self.device)
        xyz.requires_grad = True

        solver = torch.optim.Adam((xyz,), **self.gd_kwargs)

        self.grid_maker.set_radii_type_indexed(True)
        self.grid_maker.set_dimension(grid.dimension)
        self.grid_maker.set_resolution(grid.resolution)
        self.c2grid.center = tuple(grid.center.cpu().numpy().astype(float))

        for i in range(n_iters + 1):
            solver.zero_grad()

            grid_pred = self.c2grid(xyz, c, r)
            grid_diff = grid.values - grid_pred
            loss = (grid_diff**2).sum() / 2.0

            if i == n_iters: # or converged
                break

            loss.backward()
            solver.step()

        return (
            xyz.detach(),
            grid_pred.detach(),
            grid_diff.detach(),
            loss.detach()
        )

    def validify(self, struct, use_ob=False):
        '''
        Attempt to construct a valid molecule from an atomic
        structure by inferring bonds, setting aromaticity
        and connecting fragments, then minimizing energy.
        '''
        # initial struct from atom fitting (no bonds)
        rd_mol = struct.to_rd_mol()
        visited_mols = [rd_mol]

        # connect the dots using openbabel
        ob_mol = rd_mol_to_ob_mol(rd_mol)
        ob_mol.ConnectTheDots()
        rd_mol = ob_mol_to_rd_mol(ob_mol)
        visited_mols.append(rd_mol)

        # perceive bonds in openbabel and evaluate "geometric validity"
        # i.e. validity of molecule inferred purely using openbabel
        ob_mol.PerceiveBondOrders()
        rd_mol = ob_mol_to_rd_mol(ob_mol)
        struct.info['ob_validity'] = get_rd_mol_validity(rd_mol)
        struct.info['ob_mol'] = rd_mol
        visited_mols.append(rd_mol)

        if not use_ob:

            if self.dkoes_make_mol:

                ob_mol, mismatches = simple_fit.make_mol(struct)
                rd_mol = ob_mol_to_rd_mol(ob_mol)
                struct.info['add_validity'] = get_rd_mol_validity(rd_mol)
                struct.info['add_mol'] = rd_mol
                struct.info['mismatches'] = mismatches
                visited_mols.append(rd_mol)

            else: # try to fix it up using fragment connection and channel info

                # connect fragments by adding min distance bonds
                rd_mol = Chem.RWMol(rd_mol)
                connect_rd_mol_frags(rd_mol)
                visited_mols.append(rd_mol)

                # make aromatic rings using channel info
                rd_mol = Chem.RWMol(rd_mol)
                set_rd_mol_aromatic(rd_mol, struct.c, struct.channels)
                struct.info['add_validity'] = get_rd_mol_validity(rd_mol)
                struct.info['add_mol'] = rd_mol
                visited_mols.append(rd_mol)

        # minimize final molecule with UFF
        rd_mol_min, E_init, E_min, error = uff_minimize_rd_mol(rd_mol)
        struct.info['min_result'] = (E_init, E_min, error)
        struct.info['min_mol'] = rd_mol_min
        visited_mols.append(rd_mol_min)

        struct.info['visited_mols'] = visited_mols


class DkoesAtomFitter(AtomFitter):

    def __init__(self, iters=25, tol=0.01, dkoes_make_mol=True):
        self.iters = iters
        self.tol = tol
        self.dkoes_make_mol = dkoes_make_mol

    def fit(self, grid, types):
        grid_pred = simple_fit.simple_atom_fit(
            mgrid=grid,
            types=types,
            iters=self.iters,
            tol=self.tol
        )
        self.validify(grid_pred.info['src_struct'])
        return grid_pred


class OutputWriter(object):
    '''
    A data structure for receiving and organizing MolGrids and
    MolStructs from a generative model or atom fitting algorithm,
    computing metrics, and writing files to disk as necessary.
    '''
    def __init__(
        self,
        out_prefix,
        output_dx,
        output_sdf,
        output_channels,
        output_latent,
        output_visited,
        n_samples,
        blob_names,
        fit_atoms,
        batch_metrics,
        verbose
    ):

        self.out_prefix = out_prefix
        self.output_dx = output_dx
        self.output_sdf = output_sdf
        self.output_channels = output_channels
        self.output_latent = output_latent
        self.output_visited = output_visited
        self.n_samples = n_samples
        self.blob_names = blob_names
        self.fit_atoms = fit_atoms
        self.batch_metrics = batch_metrics

        # organize grids by lig_name, sample_idx, grid_name
        self.grids = defaultdict(lambda: defaultdict(dict))

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

    def write(self, lig_name, grid_name, sample_idx, grid):
        '''
        Write output files for grid and compute metrics in
        data frame, if all necessary data is present.
        '''
        grid_prefix = '{}_{}_{}'.format(self.out_prefix, lig_name, grid_name)
        sample_prefix = grid_prefix + '_' + str(sample_idx)
        add_sample_prefix = grid_prefix + '_add_' + str(sample_idx)
        ob_sample_prefix = grid_prefix + '_ob_' + str(sample_idx)

        is_gen_grid = grid_name.endswith('_gen')
        is_fit_grid = grid_name.endswith('_fit')
        is_real_grid = not (is_gen_grid or is_fit_grid)
        has_struct = is_real_grid or is_fit_grid

        # write output files
        if self.output_dx: # write density grid files

            if self.verbose:
                print('Writing ' + sample_prefix + ' .dx files')

            grid.to_dx(sample_prefix, center=np.zeros(3))
            self.dx_prefixes.append(sample_prefix)

        if has_struct and self.output_sdf: # write structure files

            # write atomic structure
            mol_file = sample_prefix + '.sdf'
            if self.verbose:
                print('Writing ' + mol_file)

            struct = grid.info['src_struct']
            if is_fit_grid and self.output_visited:
                visited_structs = grid.info['visited_structs']
                rd_mols = [s.to_rd_mol() for s in visited_structs]
            else:
                rd_mols = [struct.to_rd_mol()]

            write_rd_mols_to_sdf_file(mol_file, rd_mols)
            self.struct_files.append(mol_file)
            self.centers.append(struct.center)

            # write molecules with openbabel bonds
            ob_mol_file = ob_sample_prefix + '.sdf'
            if self.verbose:
                print('Writing ' + ob_mol_file)

            rd_mols = [struct.info['ob_mol']]
            write_rd_mols_to_sdf_file(ob_mol_file, rd_mols)
            self.struct_files.append(ob_mol_file)
            self.centers.append(struct.center)

            # write molecules with custom added bonds
            add_mol_file = add_sample_prefix + '.sdf'
            if self.verbose:
                print('Writing ' + add_mol_file)

            rd_mols = struct.info['visited_mols']
            write_rd_mols_to_sdf_file(add_mol_file, rd_mols)
            self.struct_files.append(add_mol_file)
            self.centers.append(struct.center)

            # write atom type channels
            if self.output_channels:

                channels_file = '{}.channels'.format(sample_prefix)
                if self.verbose:
                    print('Writing ' + channels_file)

                write_channels_to_file(
                    channels_file, struct.c, struct.channels
                )

        # write latent vector
        if is_gen_grid and self.output_latent:

            latent_file = sample_prefix + '.latent'
            if self.verbose:
                print('Writing ' + latent_file)

            latent_vec = grid.info['latent_vec']
            write_latent_vecs_to_file(latent_file, [latent_vec])

        # store grid until ready to compute output metrics
        self.grids[lig_name][sample_idx][grid_name] = grid
        lig_grids = self.grids[lig_name]

        n_grids = len(self.blob_names) * 2 if self.fit_atoms else 1

        if self.batch_metrics: # store until grids for all samples are ready
            has_all_samples = len(lig_grids) == self.n_samples
            has_all_grids = all(len(lig_grids[i]) == n_grids for i in lig_grids)

            if has_all_samples and has_all_grids:

                # compute batch metrics
                if self.verbose:
                    print('Computing metrics for all ' + lig_name + ' samples')

                self.compute_metrics(lig_name, range(self.n_samples))

                if self.verbose:
                    print('Writing ' + self.metric_file)

                self.metrics.to_csv(self.metric_file, sep=' ')

                if self.verbose:
                    print('Writing ' + self.pymol_file)

                write_pymol_script(
                    self.pymol_file, self.out_prefix, self.dx_prefixes,
                    self.struct_files, self.centers
                )
                del self.grids[lig_name]

        else: # only store until grids for this sample are ready
            has_all_grids = len(lig_grids[sample_idx]) == n_grids

            if has_all_grids:

                # compute sample metrics
                if self.verbose:
                    print('Computing metrics for {} sample {}'.format(
                        lig_name, sample_idx
                    ))

                self.compute_metrics(lig_name, [sample_idx])

                if self.verbose:
                    print('Writing ' + self.metric_file)

                self.metrics.to_csv(self.metric_file, sep=' ')

                if self.verbose:
                    print('Writing ' + self.pymol_file)

                write_pymol_script(
                    self.pymol_file, self.out_prefix, self.dx_prefixes,
                    self.struct_files, self.centers
                )
                del self.grids[lig_name][sample_idx]

    def compute_metrics(self, lig_name, sample_idxs):
        '''
        Compute metrics for density grids, fit atoms, and valid
        molecules for a given ligand in metrics data frame.
        '''
        lig_grids = self.grids[lig_name]

        if self.batch_metrics:

            lig_grid_mean = sum(
                lig_grids[i]['lig'].values for i in sample_idxs
            ) / self.n_samples

            lig_gen_grid_mean = sum(
                lig_grids[i]['lig_gen'].values for i in sample_idxs
            ) / self.n_samples

            latent_mean = sum(
                lig_grids[i]['lig_gen'].info['latent_vec'] for i in sample_idxs
            ) / self.n_samples

        else:
            lig_grid_mean = None
            lig_gen_grid_mean = None
            latent_mean = None

        for sample_idx in sample_idxs:
            idx = (lig_name, sample_idx)

            lig_grid = lig_grids[sample_idx]['lig']
            lig_gen_grid = lig_grids[sample_idx]['lig_gen']
            latent_vec = lig_gen_grid.info['latent_vec']

            self.compute_grid_metrics(
                idx, 'lig', lig_grid, lig_gen_grid, latent_vec,
                lig_grid_mean, lig_gen_grid_mean, latent_mean
            )

            if self.fit_atoms:

                lig_fit_grid = lig_grids[sample_idx]['lig_fit']
                lig_gen_fit_grid = lig_grids[sample_idx]['lig_gen_fit']

                lig_struct = lig_grid.info['src_struct']
                lig_fit_struct = lig_fit_grid.info['src_struct']
                lig_gen_fit_struct = lig_gen_fit_grid.info['src_struct']

                self.compute_fit_metrics(
                    idx, 'lig', lig_grid, lig_fit_grid,
                    lig_struct, lig_fit_struct
                )
                self.compute_fit_metrics(
                    idx, 'lig_gen', lig_gen_grid, lig_gen_fit_grid,
                    lig_struct, lig_gen_fit_struct
                )

                self.compute_mol_validity(
                    idx, 'lig', lig_struct, lig_fit_struct
                )
                self.compute_mol_validity(
                    idx, 'lig_gen', lig_struct, lig_gen_fit_struct
                )

        if self.verbose:
            print(self.metrics.loc[lig_name].loc[sample_idxs])

    def compute_grid_metrics(
        self,
        idx,
        prefix,
        true_grid,
        gen_grid,
        latent_vec,
        true_grid_mean,
        gen_grid_mean,
        latent_mean,
    ):
        m = self.metrics

        # density magnitude
        m.loc[idx, prefix+'_norm'] = np.linalg.norm(true_grid.values)
        m.loc[idx, prefix+'_gen_norm'] = np.linalg.norm(gen_grid.values)

        # latent vector magnitude
        m.loc[idx, prefix+'_latent_norm'] = np.linalg.norm(latent_vec)

        # generated density L2 loss
        m.loc[idx, prefix+'_gen_L2_loss'] = (
            ((true_grid.values - gen_grid.values)**2).sum()/2
        ).item()

        if self.batch_metrics: # compute batch "variance"
            # (divide by n_samples (+1) for sample (population) variance)

            lig_variance = (
                (true_grid.values - true_grid_mean)**2
            ).sum().item()

            lig_gen_variance = (
                (gen_grid.values - gen_grid_mean)**2
            ).sum().item()

            latent_variance = (
                (latent_vec - latent_mean)**2
            ).sum()

        else:
            lig_variance = np.nan
            lig_gen_variance = np.nan
            latent_variance = np.nan

        m.loc[idx, prefix+'_variance'] = lig_variance
        m.loc[idx, prefix+'_gen_variance'] = lig_gen_variance
        m.loc[idx, prefix+'_latent_variance'] = latent_variance

    def compute_fit_metrics(
        self,
        idx,
        prefix,
        true_grid,
        fit_grid,
        true_struct,
        fit_struct
    ):
        m = self.metrics
        is_gen_grid = prefix.endswith('_gen')

        # fit density L2 loss
        m.loc[idx, prefix+'_fit_L2_loss'] = (
            ((true_grid.values - fit_grid.values)**2).sum()/2
        ).item()

        # number of atoms
        if not is_gen_grid:
            m.loc[idx, prefix+'_n_atoms'] = true_struct.n_atoms
        m.loc[idx, prefix+'_fit_n_atoms'] = fit_struct.n_atoms

        # fit structure radius
        if not is_gen_grid:
            m.loc[idx, prefix+'_radius'] = true_struct.radius
        m.loc[idx, prefix+'_fit_radius'] = fit_struct.radius

        n_types = len(true_struct.channels)
        true_type_count = count_types(true_struct.c, n_types)
        fit_type_count  = count_types(fit_struct.c, n_types)

        # fit type difference
        m.loc[idx, prefix+'_fit_type_diff'] = np.linalg.norm(
            true_type_count - fit_type_count, ord=1
        )
        m.loc[idx, prefix+'_est_type_diff'] = fit_struct.info['est_type_diff']

        m.loc[idx, prefix+'_fit_exact_types'] = (
            m.loc[idx, prefix+'_fit_type_diff'] == 0
        )
        m.loc[idx, prefix+'_est_exact_types'] = (
            m.loc[idx, prefix+'_est_type_diff'] == 0
        )

        # fit minimum RMSD
        try:
            rmsd = get_min_rmsd(
                true_struct.xyz, true_struct.c, fit_struct.xyz, fit_struct.c
            )
        except (ValueError, ZeroDivisionError):
            rmsd = np.nan

        m.loc[idx, prefix+'_fit_RMSD'] = rmsd

        # fit time and number of visited structures
        m.loc[idx, prefix+'_fit_time'] = fit_struct.info['time']
        m.loc[idx, prefix+'_fit_n_visited'] = len(fit_grid.info['visited_structs'])

    def compute_mol_validity(self, idx, prefix, true_struct, fit_struct):

        m = self.metrics

        from_gen_grid = prefix.endswith('_gen')

        true_mol = true_struct.info['ob_mol']
        true_mol_min = true_struct.info['min_mol']

        mol = fit_struct.info['add_mol']
        mol_min = fit_struct.info['min_mol']

        # sanity check- validity of true molecule
        if not from_gen_grid:
            n_frags, error, valid = true_struct.info['ob_validity']
            m.loc[idx, prefix+'_ob_n_frags'] = n_frags
            m.loc[idx, prefix+'_ob_error'] = error
            m.loc[idx, prefix+'_ob_valid'] = valid

        # check mol validity of openbabel bond connecting
        n_frags, error, valid = fit_struct.info['ob_validity']
        m.loc[idx, prefix+'_fit_ob_n_frags'] = n_frags
        m.loc[idx, prefix+'_fit_ob_error'] = error
        m.loc[idx, prefix+'_fit_ob_valid'] = valid

        # check validity after custom bond adding
        n_frags, error, valid = fit_struct.info['add_validity']
        m.loc[idx, prefix+'_fit_add_n_frags'] = n_frags
        m.loc[idx, prefix+'_fit_add_error'] = error
        m.loc[idx, prefix+'_fit_add_valid'] = valid
        m.loc[idx, prefix+'_fit_add_mismatches'] = fit_struct.info.get('mismatches', np.nan)

        # convert to SMILES string
        true_smi = Chem.MolToSmiles(true_mol, canonical=True)
        smi = Chem.MolToSmiles(mol,  canonical=True)

        if not from_gen_grid:
            m.loc[idx, prefix+'_add_SMILES'] = true_smi

        m.loc[idx, prefix+'_fit_add_SMILES'] = smi
        m.loc[idx, prefix+'_fit_add_SMILES_match'] = (smi == true_smi)

        # fingerprint similarity
        m.loc[idx, prefix+'_fit_add_ob_sim']  = get_ob_smi_similarity(
            true_smi, smi
        )
        m.loc[idx, prefix+'_fit_add_morgan_sim'] = get_rd_mol_similarity(
            true_mol, mol, 'morgan'
        )
        m.loc[idx, prefix+'_fit_add_rdkit_sim']  = get_rd_mol_similarity(
            true_mol, mol, 'rdkit'
        )
        m.loc[idx, prefix+'_fit_add_maccs_sim']  = get_rd_mol_similarity(
            true_mol, mol, 'maccs'
        )

        # other molecular descriptors
        if not from_gen_grid:
            m.loc[idx, prefix+'_add_MW'] = get_rd_mol_weight(true_mol)
            m.loc[idx, prefix+'_add_logP'] = get_rd_mol_logP(true_mol)
            m.loc[idx, prefix+'_add_QED'] = get_rd_mol_QED(true_mol)
            m.loc[idx, prefix+'_add_SAS'] = get_rd_mol_SAS(true_mol)
            m.loc[idx, prefix+'_add_NPS'] = get_rd_mol_NPS(true_mol, nps_model)

        m.loc[idx, prefix+'_fit_add_MW'] = get_rd_mol_weight(mol)
        m.loc[idx, prefix+'_fit_add_logP'] = get_rd_mol_logP(mol)
        m.loc[idx, prefix+'_fit_add_QED'] = get_rd_mol_QED(mol)
        m.loc[idx, prefix+'_fit_add_SAS'] = get_rd_mol_SAS(mol)
        m.loc[idx, prefix+'_fit_add_NPS'] = get_rd_mol_NPS(mol, nps_model)

        # UFF energy minimization
        E_init_t, E_min_t, error = true_struct.info['min_result']
        if not from_gen_grid:
            m.loc[idx, prefix+'_add_E'] = E_init_t
            m.loc[idx, prefix+'_add_min_E'] = E_min_t
            m.loc[idx, prefix+'_add_dE_min'] = E_min_t - E_init_t
            m.loc[idx, prefix+'_add_min_error'] = error
            m.loc[idx, prefix+'_add_RMSD_min'] = get_aligned_rmsd(true_mol_min, true_mol)

        E_init, E_min, error = fit_struct.info['min_result']
        m.loc[idx, prefix+'_fit_add_E'] = E_init
        m.loc[idx, prefix+'_fit_add_min_E'] = E_min
        m.loc[idx, prefix+'_fit_add_dE_min'] = E_min - E_init
        m.loc[idx, prefix+'_fit_add_min_error'] = error
        m.loc[idx, prefix+'_fit_add_RMSD_min']  = get_aligned_rmsd(mol_min, mol)

        # compare energy to true mol, pre- and post-minimize
        m.loc[idx, prefix+'_fit_add_dE_true'] = E_init - E_init_t
        m.loc[idx, prefix+'_fit_add_min_dE_true'] = E_min - E_min_t

        # get aligned RMSD to true mol, pre-minimize
        m.loc[idx, prefix+'_fit_add_RMSD_true'] = get_aligned_rmsd(true_mol, mol)

        # get aligned RMSD to true mol, post-minimize
        m.loc[idx, prefix+'_fit_add_min_RMSD_true'] = get_aligned_rmsd(true_mol_min, mol_min)


def catch_exc(func, exc=Exception, default=np.nan):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except exc as e:
            return default
    return wrapper


get_rd_mol_weight = catch_exc(Chem.Descriptors.MolWt)
get_rd_mol_logP   = catch_exc(Chem.Crippen.MolLogP)
get_rd_mol_QED    = catch_exc(Chem.QED.default)
get_rd_mol_SAS    = catch_exc(sascorer.calculateScore)
get_rd_mol_NPS    = catch_exc(npscorer.scoreMol)
get_aligned_rmsd  = catch_exc(AllChem.GetBestRMS)


@catch_exc
def get_gpu_usage(gpu_id=0):
    return getGPUs()[gpu_id].memoryUtil


def set_rd_mol_aromatic(rd_mol, c, channels):

    # get aromatic carbon channels
    aroma_c_channels = set()
    for i, channel in enumerate(channels):
        if 'AromaticCarbon' in channel.name:
            aroma_c_channels.add(i)

    # get aromatic carbon atoms
    aroma_c_atoms = set()
    for i, c_ in enumerate(c):
        if c_ in aroma_c_channels:
            aroma_c_atoms.add(i)

    # make aromatic rings using channel info
    rings = Chem.GetSymmSSSR(rd_mol)
    for ring_atoms in rings:
        ring_atoms = set(ring_atoms)
        if len(ring_atoms & aroma_c_atoms) == 0: #TODO test < 3 instead, and handle heteroatoms
            continue
        if (len(ring_atoms) - 2)%4 != 0:
            continue
        for bond in rd_mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            if atom1.GetIdx() in ring_atoms and atom2.GetIdx() in ring_atoms:
                atom1.SetIsAromatic(True)
                atom2.SetIsAromatic(True)
                bond.SetBondType(Chem.BondType.AROMATIC)


def connect_rd_mol_frags(rd_mol):

    # try to connect fragments by adding min distance bonds
    frags = Chem.GetMolFrags(rd_mol)
    n_frags = len(frags)
    if n_frags > 1:

        nax = np.newaxis
        xyz = rd_mol.GetConformer(0).GetPositions()
        dist2 = ((xyz[nax,:,:] - xyz[:,nax,:])**2).sum(axis=2)

        pt = Chem.GetPeriodicTable()
        while n_frags > 1:

            frag_map = {ai: fi for fi, f in enumerate(frags) for ai in f}
            frag_idx = np.array([frag_map[i] for i in range(rd_mol.GetNumAtoms())])
            diff_frags = frag_idx[nax,:] != frag_idx[:,nax]

            can_bond = np.array([a.GetExplicitValence() <
                                 pt.GetDefaultValence(a.GetAtomicNum())
                                 for a in rd_mol.GetAtoms()])
            can_bond = can_bond[nax,:] & can_bond[:,nax]

            cond_dist2 = np.where(diff_frags & can_bond & (dist2<25), dist2, np.inf)

            if not np.any(np.isfinite(cond_dist2)):
                break # no possible bond meets the conditions

            a1, a2 = np.unravel_index(cond_dist2.argmin(), dist2.shape)
            rd_mol.AddBond(int(a1), int(a2), Chem.BondType.SINGLE)
            try:
                rd_mol.UpdatePropertyCache() # update explicit valences
            except:
                pass

            frags = Chem.GetMolFrags(rd_mol)
            n_frags = len(frags)


@catch_exc
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


@catch_exc
def get_ob_smi_similarity(smi1, smi2):
    fgp1 = pybel.readstring('smi', smi1).calcfp()
    fgp2 = pybel.readstring('smi', smi2).calcfp()
    return fgp1 | fgp2


def uff_minimize_rd_mol(rd_mol, max_iters=1000):
    try:
        rd_mol_H = Chem.AddHs(rd_mol, addCoords=True)
        ff = AllChem.UFFGetMoleculeForceField(rd_mol_H, confId=0)
        ff.Initialize()
        E_init = ff.CalcEnergy()
        try:
            res = ff.Minimize(maxIts=max_iters)
            E_final = ff.CalcEnergy()
            rd_mol = Chem.RemoveHs(rd_mol_H)
            if res == 0:
                e = None
            else:
                e = RuntimeError('minimization not converged')
            return rd_mol, E_init, E_final, e
        except RuntimeError as e:
            return Chem.RWMol(rd_mol), E_init, np.nan, e
    except Exception as e:
        return Chem.RWMol(rd_mol), np.nan, np.nan, e


@catch_exc
def get_rd_mol_uff_energy(rd_mol): # TODO do we need to add H for true mol?
    rd_mol_H = Chem.AddHs(rd_mol, addCoords=True)
    ff = AllChem.UFFGetMoleculeForceField(rd_mol_H, confId=0)
    return ff.CalcEnergy()


def get_rd_mol_validity(rd_mol):
    n_frags = len(Chem.GetMolFrags(rd_mol))
    try:
        Chem.SanitizeMol(rd_mol)
        error = None
    except Exception as e:
        error = e
    valid = n_frags == 1 and error is None
    return n_frags, error, valid


def ob_mol_to_rd_mol(ob_mol):

    n_atoms = ob_mol.NumAtoms()
    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(n_atoms)

    for ob_atom in ob.OBMolAtomIter(ob_mol):
        rd_atom = Chem.Atom(ob_atom.GetAtomicNum())
        rd_atom.SetIsAromatic(ob_atom.IsAromatic())
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
            bond_type = Chem.BondType.AROMATIC
        elif bond_order == 1:
            bond_type = Chem.BondType.SINGLE
        elif bond_order == 2:
            bond_type = Chem.BondType.DOUBLE
        elif bond_order == 3:
            bond_type = Chem.BondType.TRIPLE
        else:
            raise Exception('unknown bond order {}'.format(bond_order))
        rd_mol.AddBond(i, j, bond_type)

    return rd_mol


def rd_mol_to_ob_mol(rd_mol):

    ob_mol = ob.OBMol()
    rd_conf = rd_mol.GetConformer(0)

    for i, rd_atom in enumerate(rd_mol.GetAtoms()):
        atomic_num = rd_atom.GetAtomicNum()
        rd_coords = rd_conf.GetAtomPosition(i)
        x = rd_coords.x
        y = rd_coords.y
        z = rd_coords.z
        ob_atom = ob_mol.NewAtom()
        ob_atom.SetAtomicNum(atomic_num)
        ob_atom.SetAromatic(rd_atom.GetIsAromatic())
        ob_atom.SetVector(x, y, z)

    for k, rd_bond in enumerate(rd_mol.GetBonds()):
        i = rd_bond.GetBeginAtomIdx()+1
        j = rd_bond.GetEndAtomIdx()+1
        bond_type = rd_bond.GetBondType()
        bond_flags = 0
        if bond_type == Chem.BondType.AROMATIC:
            bond_order = 1
            bond_flags |= ob.OB_AROMATIC_BOND
        elif bond_type == Chem.BondType.SINGLE:
            bond_order = 1
        elif bond_type == Chem.BondType.DOUBLE:
            bond_order = 2
        elif bond_type == Chem.BondType.TRIPLE:
            bond_order = 3
        ob_mol.AddBond(i, j, bond_order, bond_flags)

    return ob_mol
                   

def make_rd_mol(xyz, c, bonds, channels):

    rd_mol = Chem.RWMol()

    for c_ in c:
        atomic_num = channels[c_].atomic_num
        rd_atom = Chem.Atom(atomic_num)
        rd_mol.AddAtom(rd_atom)

    n_atoms = rd_mol.GetNumAtoms()
    rd_conf = Chem.Conformer(n_atoms)

    for i, (x, y, z) in enumerate(xyz):
        rd_coords = Geometry.Point3D(x, y, z)
        rd_conf.SetAtomPosition(i, rd_coords)

    rd_mol.AddConformer(rd_conf)

    if np.any(bonds):
        n_bonds = 0
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                if bonds[i,j]:
                    rd_mol.AddBond(i, j, Chem.BondType.SINGLE)
                    n_bonds += 1

    return rd_mol


def make_ob_mol(xyz, c, bonds, channels):
    '''
    Return an OpenBabel molecule from an array of
    xyz atom positions, channel indices, a bond matrix,
    and a list of atom type channels.
    '''
    ob_mol = ob.OBMol()

    n_atoms = 0
    for (x, y, z), c_ in zip(xyz, c):
        atomic_num = channels[c_].atomic_num
        atom = ob_mol.NewAtom()
        atom.SetAtomicNum(atomic_num)
        atom.SetVector(x, y, z)
        n_atoms += 1

    if np.any(bonds):
        n_bonds = 0
        for i in range(n_atoms):
            atom_i = ob_mol.GetAtom(i)
            for j in range(i+1, n_atoms):
                atom_j = ob_mol.GetAtom(j)
                if bonds[i,j]:
                    bond = ob_mol.NewBond()
                    bond.Set(n_bonds, atom_i, atom_j, 1, 0)
                    n_bonds += 1
    return ob_mol


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
    writer.SetKekulize(False)
    for mol in mols:
        writer.write(mol)


def read_rd_mols_from_sdf_file(sdf_file):
    if sdf_file.endswith('.gz'):
        f = gzip.open(sdf_file)
        suppl = Chem.ForwardSDMolSupplier(f)
    else:
        suppl = Chem.SDMolSupplier(sdf_file)
    return [mol for mol in suppl]


def write_latent_vecs_to_file(latent_file, latent_vecs):

    with open(latent_file, 'w') as f:
        for v in latent_vecs:
            line = ' '.join('{:.5f}'.format(x) for x in v) + '\n'
            f.write(line)


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


def write_pymol_script(pymol_file, out_prefix, dx_prefixes, struct_files, centers=[]):
    '''
    Write a pymol script with a map object for each of dx_files, a
    group of all map objects (if any), a rec_file, a lig_file, and
    an optional fit_file.
    '''
    with open(pymol_file, 'w') as f:

        for dx_prefix in dx_prefixes: # load densities
            dx_pattern = '{}_*.dx'.format(dx_prefix)
            m = re.match('^{}_(.*)$'.format(out_prefix), dx_prefix)
            group_name = m.group(1) + '_grids'
            f.write('load_group {}, {}\n'.format(dx_pattern, group_name))

        for struct_file in struct_files: # load structures
            m = re.match('^{}_(.*)\\.sdf$'.format(out_prefix), struct_file)
            obj_name = m.group(1)
            f.write('load {}, {}\n'.format(struct_file, obj_name))

        for struct_file, (x,y,z) in zip(struct_files, centers): # center structures
            m = re.match('^{}_(.*)\\.sdf$'.format(out_prefix), struct_file)
            obj_name = m.group(1)
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


def one_hot_to_index(x):
    if len(x) > 0:
        return torch.argmax(x, dim=1)
    else:
        return torch.zeros((0,))


def write_structs_to_sdf_file(sdf_file, structs):
    mols = (s.to_ob_mol() for s in structs)
    write_ob_mols_to_sdf_file(sdf_file, mols)


def write_channels_to_file(channels_file, c, channels):
    with open(channels_file, 'w') as f:
        for c_ in c:
            channel = channels[c_]
            f.write(channel.name+'\n')


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


def slerp(v0, v1, t):
    '''
    Spherical linear interpolation between
    vectors v0 and v1 at points t.
    '''
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

    print('Creating example provider')
    ex_provider = molgrid.ExampleProvider(
        rec_map,
        lig_map,
        data_root=args.data_root,
        recmolcache=data_param.recmolcache,
        ligmolcache=data_param.ligmolcache,
    )

    print('Populating example provider')
    ex_provider.populate(data_param.source)

    print('Creating grid maker')
    grid_maker = molgrid.GridMaker(data_param.resolution, data_param.dimension)
    grid_dims = grid_maker.grid_dimensions(rec_map.num_types() + lig_map.num_types())
    grid_true = torch.zeros(batch_size, *grid_dims, dtype=torch.float32, device=device)

    print('Finding important blobs')
    try: # find receptor encoder blobs
        rec_enc_start = find_blobs_in_net(gen_net, 'rec')[0]
        try:
            rec_enc_end = find_blobs_in_net(gen_net, 'rec_latent_std')[0]
            rec_enc_is_var = True
        except IndexError:
            rec_enc_end = find_blobs_in_net(gen_net, 'rec_latent_fc')[0]
            rec_enc_is_var = False
        has_rec_enc = True
    except IndexError:
        has_rec_enc = False

    if args.verbose:
        print('has_rec_enc = {}'.format(has_rec_enc))
        if has_rec_enc:
            print('\trec_enc_is_var = {}'.format(rec_enc_is_var))
            print('\trec_enc_start = {}'.format(repr(rec_enc_start)))
            print('\trec_enc_end = {}'.format(repr(rec_enc_end)))

    try: # find ligand encoder blobs
        lig_enc_start = find_blobs_in_net(gen_net, 'lig')[0]
        try:
            lig_enc_end = find_blobs_in_net(gen_net, 'lig_latent_std')[0]
            lig_enc_is_var = True
        except IndexError:
            try:
                lig_enc_end = find_blobs_in_net(gen_net, 'lig_latent_defc')[0]
            except IndexError:
                lig_enc_end = find_blobs_in_net(gen_net, 'lig_latent_fc')[0]
            lig_enc_is_var = False
        has_lig_enc = True
    except IndexError:
        has_lig_enc = False

    if args.verbose:
        print('has_lig_enc = {}'.format(has_lig_enc))
        if has_lig_enc:
            print('\tlig_enc_is_var = {}'.format(lig_enc_is_var))
            print('\tlig_enc_start = {}'.format(repr(lig_enc_start)))
            print('\tlig_enc_end = {}'.format(repr(lig_enc_end)))

    # must have at least one encoder
    assert (has_rec_enc or has_lig_enc)

    # only one encoder can be variational
    if has_rec_enc and has_lig_enc:
        assert not (rec_enc_is_var and lig_enc_is_var)

    try: # find latent variable blobs
        latent_prefix = ('lig' if has_lig_enc else 'rec') + '_latent'
        latent_mean = find_blobs_in_net(gen_net, latent_prefix+'_mean')[0]
        latent_std = find_blobs_in_net(gen_net, latent_prefix+'_std')[0]
        latent_noise = find_blobs_in_net(gen_net, latent_prefix+'_noise')[0]
        latent_sample = find_blobs_in_net(gen_net, latent_prefix+'_sample')[0]
        variational = True
    except IndexError:
        variational = False

    if args.verbose:
        print('variational = {}'.format(variational))
        if variational:
            print('\tlatent_mean = {}'.format(repr(latent_mean)))
            print('\tlatent_std = {}'.format(repr(latent_std)))
            print('\tlatent_noise = {}'.format(repr(latent_noise)))
            print('\tlatent_sample = {}'.format(repr(latent_sample)))

    # find ligand decoder blobs (required)
    if has_rec_enc and has_lig_enc:
        lig_dec_start = find_blobs_in_net(gen_net, 'latent_concat')[0]
    else:
        lig_dec_start = find_blobs_in_net(gen_net, 'lig_dec_fc')[0]
    lig_dec_end = find_blobs_in_net(gen_net, 'lig_gen')[0]

    if args.verbose:
        print('has_lig_dec = True')
        print('\tlig_dec_start = {}'.format(repr(lig_dec_start)))
        print('\tlig_dec_end = {}'.format(repr(lig_dec_end)))

    n_latent = gen_net.blobs[latent_sample].shape[1]

    print('Testing generator forward')
    gen_net.forward() # this is necessary for proper latent sampling

    print('Creating atom fitter and output writer')

    if args.parallel: # compute metrics and write output in separate thread

        out_queue = mp.Queue()
        out_thread = threading.Thread(
            target=out_worker_main,
            args=(out_queue, args),
        )
        out_thread.start()

        if args.fit_atoms: # fit atoms to grids in separate processes
            fit_queue = mp.Queue(args.n_fit_workers) # queue for atom fitting
            fit_procs = mp.Pool(
                processes=args.n_fit_workers,
                initializer=fit_worker_main,
                initargs=(fit_queue, out_queue, args),
            )

    else: # compute metrics, write output, and fit atoms in single thread

        out_writer = OutputWriter(
            out_prefix=args.out_prefix,
            output_dx=args.output_dx,
            output_sdf=args.output_sdf,
            output_channels=args.output_channels,
            output_latent=args.output_latent,
            output_visited=args.output_visited,
            n_samples=args.n_samples,
            batch_metrics=args.batch_metrics,
            blob_names=args.blob_name,
            fit_atoms=args.fit_atoms,
            verbose=args.verbose,
        )

        if args.fit_atoms:

            if args.dkoes_simple_fit:
                atom_fitter = DkoesAtomFitter()
            else:
                atom_fitter = AtomFitter(
                    multi_atom=args.multi_atom, 
                    beam_size=args.beam_size,
                    apply_conv=args.apply_conv,
                    threshold=args.threshold,
                    peak_value=args.peak_value,
                    min_dist=args.min_dist,
                    constrain_types=args.constrain_types,
                    constrain_frags=False,
                    estimate_types=args.estimate_types,
                    interm_gd_iters=args.interm_gd_iters,
                    final_gd_iters=args.final_gd_iters,
                    gd_kwargs=dict(
                        lr=args.learning_rate,
                        betas=(args.beta1, args.beta2),
                        weight_decay=args.weight_decay,
                    ),
                    dkoes_make_mol=args.dkoes_make_mol,
                    output_kernel=args.output_kernel,
                    device=device,
                    verbose=args.verbose,
                )

    # generate density grids from generative model in main thread
    print('Starting to generate grids')
    try:
        for example_idx in range(n_examples):

            for sample_idx in range(args.n_samples):

                # keep track of position in batch
                batch_idx = (example_idx*args.n_samples + sample_idx) % batch_size

                if args.interpolate:
                    endpoint_idx = 0 if batch_idx < batch_size//2 else -1

                if batch_idx == 0: # forward next batch

                    # get next batch of structures
                    print('Getting batch of examples')
                    examples = ex_provider.next_batch(batch_size)

                    # convert structures to grids
                    print('Transforming and gridding examples')
                    for i, ex in enumerate(examples):
                        transform = molgrid.Transform(
                            ex.coord_sets[1].center(),
                            args.random_translate,
                            args.random_rotation,
                        )
                        transform.forward(ex, ex)
                        grid_maker.forward(ex, grid_true[i])

                    rec = grid_true[:,:rec_map.num_types(),...].cpu()
                    lig = grid_true[:,rec_map.num_types():,...].cpu()

                    need_first = (args.encode_first or args.condition_first)
                    is_first = (example_idx == sample_idx == 0)

                    if need_first and is_first:
                        first_rec = np.array(rec[:1])
                        first_lig = np.array(lig[:1])

                    print('Calling generator forward')

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

                    if args.interpolate: # copy true grids that will be interpolated
                        start_rec = np.array(rec[:1])
                        start_lig = np.array(lig[:1])
                        end_rec = np.array(rec[-1:])
                        end_lig = np.array(lig[-1:])
                        gen_net.blobs['rec'].data[:batch_size//2] = start_rec
                        gen_net.blobs['lig'].data[:batch_size//2] = start_lig
                        gen_net.blobs['rec'].data[batch_size//2:] = end_rec
                        gen_net.blobs['lig'].data[batch_size//2:] = end_lig

                    if has_rec_enc: # forward receptor encoder
                        if rec_enc_is_var:
                            if args.prior:
                                gen_net.blobs[latent_mean].data[...] = 0.0
                                if args.mean:
                                    gen_net.blobs[latent_std].data[...] = 0.0
                                else:
                                    gen_net.blobs[latent_std].data[...] = 1.0
                            else:
                                gen_net.forward(start=rec_enc_start, end=rec_enc_end)
                                if args.mean:
                                    gen_net.blobs[latent_std].data[...] = 0.0
                        else:
                            gen_net.forward(start=rec_enc_start, end=rec_enc_end)

                    if has_lig_enc: # forward ligand encoder
                        if lig_enc_is_var:
                            if args.prior:
                                gen_net.blobs[latent_mean].data[...] = 0.0
                                if args.mean:
                                    gen_net.blobs[latent_std].data[...] = 0.0
                                else:
                                    gen_net.blobs[latent_std].data[...] = 1.0
                            else: # posterior
                                if args.mean:
                                    gen_net.forward(start=lig_enc_start, end=latent_mean)
                                    gen_net.blobs[latent_std].data[...] = 0.0
                                else:
                                    gen_net.forward(start=lig_enc_start, end=lig_enc_end)
                        else:
                            gen_net.forward(start=lig_enc_start, end=lig_enc_end)

                    if variational: # sample latent variables
                        gen_net.forward(start=latent_noise, end=latent_sample)

                    if args.interpolate: # interpolate between latent samples

                        latent = gen_net.blobs[latent_sample].data
                        start_latent = np.array(latent[0])
                        end_latent = np.array(latent[-1])

                        if args.spherical:
                            gen_net.blobs[latent_sample].data[...] = slerp(
                                start_latent,
                                end_latent,
                                np.linspace(0, 1, batch_size, endpoint=True)
                            )
                        else:
                            gen_net.blobs[latent_sample].data[...] = np.linspace(
                                start_latent, end_latent, batch_size, endpoint=True
                            )

                    # decode latent samples to generate grids
                    gen_net.forward(start=lig_dec_start, end=lig_dec_end)

                print('Getting true molecule for current example')

                # get current example ligand
                if args.interpolate:
                    ex = examples[endpoint_idx]
                else:
                    ex = examples[batch_idx]

                lig_coord_set = ex.coord_sets[1]
                lig_src_file = lig_coord_set.src
                struct = MolStruct.from_coord_set(lig_coord_set, lig_channels)
                types = count_types(struct.c, lig_map.num_types(), dtype=np.int16)

                lig_src_no_ext = os.path.splitext(lig_src_file)[0]
                lig_name = os.path.basename(lig_src_no_ext)

                try: # get true mol from the original sdf file
                    m = re.match(r'(.+)_ligand_(\d+)', lig_src_no_ext)
                    if m:
                        lig_sdf_base = m.group(1) + '_docked.sdf.gz'
                        idx = int(m.group(2))
                    else:
                        lig_sdf_base = lig_src_no_ext + '.sdf'
                        idx = 0
                    lig_sdf_file = os.path.join(args.data_root, lig_sdf_base)
                    mol = read_rd_mols_from_sdf_file(lig_sdf_file)[idx]
                    print('Found true molecule in data root')
                    struct.info['src_mol'] = mol

                except Exception as e: # get true mol from openbabel
                    print('Did not find true molecule in data root')
                    struct.info['src_mol'] = None

                atom_fitter.validify(struct, use_ob=True)
                print('True molecule for {} has {} atoms'.format(lig_name, struct.n_atoms))

                # get latent vector for current example
                latent_vec = np.array(gen_net.blobs[latent_sample].data[batch_idx])

                # get data from blob, process, and write output
                for blob_name in args.blob_name:

                    print('Getting grid from {} blob'.format(blob_name))

                    grid_blob = gen_net.blobs[blob_name]
                    grid_needs_fit = args.fit_atoms and blob_name.startswith('lig')

                    if blob_name == 'rec':
                        grid_channels = rec_channels
                    elif blob_name in {'lig', 'lig_gen'}:
                        grid_channels = lig_channels
                    else:
                        grid_channels = atom_types.get_n_unknown_channels(
                            grid_blob.shape[1]
                        )

                    if args.interpolate and blob_name in {'rec', 'lig'}:
                        grid_data = grid_blob.data[endpoint_idx]
                    else:
                        grid_data = grid_blob.data[batch_idx]

                    grid = MolGrid(
                        values=np.array(grid_data),
                        channels=grid_channels,
                        center=struct.center,
                        resolution=grid_maker.get_resolution(),
                    )
                    grid_name = blob_name
                    grid_norm = np.linalg.norm(grid.values)

                    if grid_name == 'lig': # store true structure for input ligand grids
                        grid.info['src_struct'] = struct

                    elif grid_name == 'lig_gen': # store latent vector for generated grids
                        grid.info['latent_vec'] = latent_vec

                    if args.verbose:
                        gpu_usage = get_gpu_usage(0)

                        print('Produced {} {} {} (norm={}\tGPU={})'.format(
                            lig_name, grid_name.ljust(7), sample_idx, grid_norm, gpu_usage
                        ), flush=True)

                    if args.parallel:
                        out_queue.put((lig_name, grid_name, sample_idx, grid))
                        if grid_needs_fit:
                            fit_queue.put((lig_name, grid_name, sample_idx, grid, types))
                    else:
                        out_writer.write(lig_name, grid_name, sample_idx, grid)
                        if grid_needs_fit:
                            grid_fit = atom_fitter.fit(grid, types)
                            grid_fit_name = grid_name + '_fit'
                            out_writer.write(lig_name, grid_fit_name, sample_idx, grid_fit)
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
        print('Main thread exit')


def fit_worker_main(fit_queue, out_queue, args):

    if args.dkoes_simple_fit:
        atom_fitter = DkoesAtomFitter()
    else:
        atom_fitter = AtomFitter(
            multi_atom=args.multi_atom,
            beam_size=args.beam_size,
            apply_conv=args.apply_conv,
            threshold=args.threshold,
            peak_value=args.peak_value,
            min_dist=args.min_dist,
            constrain_types=args.constrain_types,
            constrain_frags=False,
            estimate_types=args.estimate_types,
            interm_gd_iters=args.interm_gd_iters,
            final_gd_iters=args.final_gd_iters,
            gd_kwargs=dict(
                lr=args.learning_rate,
                betas=(args.beta1, args.beta2),
                weight_decay=args.weight_decay,
            ),
            dkoes_make_mol=args.dkoes_make_mol,
            output_kernel=args.output_kernel,
            device='cpu', # can't fit on gpu in multiple threads
            verbose=args.verbose,
        )

    while True:
        if args.verbose:
            print('Fit worker waiting')

        task = fit_queue.get()
        if task is None:
            break

        lig_name, grid_name, sample_idx, grid, types = task

        if args.verbose:
            print('Fit worker got {} {} {}'.format(lig_name, grid_name, sample_idx))

        out_queue.put((lig_name, grid_name, sample_idx, grid))

        grid_fit = atom_fitter.fit(grid, types)
        grid_fit_name = grid_name + '_fit'

        if args.verbose:
            print('Fit worker produced {} {} {} ({} atoms, {}s)'.format(
                lig_name, grid_name, sample_idx, struct_fit.n_atoms, struct_fit.fit_time
            ), flush=True)

        out_queue.put((lig_name, grid_fit_name, sample_idx, grid_fit))

    if args.verbose:
        print('Fit worker exit')


def out_worker_main(out_queue, args):

    out_writer = OutputWriter(
        out_prefix=args.out_prefix,
        output_dx=args.output_dx,
        output_sdf=args.output_sdf,
        output_channels=args.output_channels,
        output_latent=args.output_latent,
        output_visited=args.output_visited,
        n_samples=args.n_samples,
        batch_metrics=args.batch_metrics,
        blob_names=args.blob_name,
        fit_atoms=args.fit_atoms,
        verbose=args.verbose,
    )

    while True:
        task = out_queue.get()
        if task is None:
            break
        out_writer.write(*task)

    if args.verbose:
        print('Output worker exit')


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
    parser.add_argument('--all_blobs', default=False, action='store_true', help='generate from all blobs in generative model')
    parser.add_argument('--n_samples', default=1, type=int, help='number of samples to generate for each input example')
    parser.add_argument('--prior', default=False, action='store_true', help='generate from prior instead of posterior distribution')
    parser.add_argument('--mean', default=False, action='store_true', help='generate mean of distribution instead of sampling')
    parser.add_argument('--encode_first', default=False, action='store_true', help='generate all output from encoding first example')
    parser.add_argument('--condition_first', default=False, action='store_true', help='condition all generated output on first example')
    parser.add_argument('--interpolate', default=False, action='store_true', help='interpolate between examples in latent space')
    parser.add_argument('--spherical', default=False, action='store_true', help='use spherical interpolation instead of linear')
    parser.add_argument('-o', '--out_prefix', required=True, help='common prefix for output files')
    parser.add_argument('--output_dx', action='store_true', help='output .dx files of atom density grids for each channel')
    parser.add_argument('--output_sdf', action='store_true', help='output .sdf file of best fit atom positions')
    parser.add_argument('--output_visited', action='store_true', help='output every visited structure in .sdf files')
    parser.add_argument('--output_kernel', action='store_true', help='output .dx files for kernel used to intialize atoms during atom fitting')
    parser.add_argument('--output_channels', action='store_true', help='output channels of each fit structure in separate files')
    parser.add_argument('--output_latent', action='store_true', help='output latent vectors for each generated density grid')
    parser.add_argument('--fit_atoms', action='store_true', help='fit atoms to density grids and print the goodness-of-fit')
    parser.add_argument('--dkoes_simple_fit', default=False, action='store_true', help='fit atoms using alternate functions by dkoes')
    parser.add_argument('--dkoes_make_mol', default=False, action='store_true', help="validify molecule using alternate functions by dkoes")
    parser.add_argument('--constrain_types', action='store_true', help='constrain atom fitting to find atom types of true ligand (or estimate)')
    parser.add_argument('--estimate_types', action='store_true', help='estimate atom type counts using the total grid density per channel')
    parser.add_argument('--multi_atom', default=False, action='store_true', help='add all next atoms to grid simultaneously at each atom fitting step')
    parser.add_argument('--beam_size', type=int, default=1, help='number of best structures to track during atom fitting beam search')
    parser.add_argument('--apply_conv', default=False, action='store_true', help='apply convolution to grid before detecting next atoms')
    parser.add_argument('--threshold', type=float, default=0.1, help='threshold value for detecting next atoms on grid')
    parser.add_argument('--peak_value', type=float, default=1.5, help='reflect grid values higher than this value before detecting next atoms')
    parser.add_argument('--min_dist', type=float, default=0.0, help='minimum distance between detected atoms, in terms of covalent bond length')
    parser.add_argument('--interm_gd_iters', type=int, default=10, help='number of gradient descent iterations after each step of atom fitting')
    parser.add_argument('--final_gd_iters', type=int, default=100, help='number of gradient descent iterations after final step of atom fitting')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate for Adam optimizer')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay for Adam optimizer')
    parser.add_argument('--batch_metrics', default=False, action='store_true', help="compute variance metrics across different samples")
    parser.add_argument('--verbose', default=0, type=int, help="verbose output level")
    parser.add_argument('--debug', default=False, action='store_true', help='debug mode')
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
    data_param.root_folder = args.data_root

    if args.gpu:
        print('Setting caffe to GPU mode')
        caffe.set_mode_gpu()
    else:
        print('Setting caffe to CPU mode')
        caffe.set_mode_cpu()

    # create the net in caffe
    print('Constructing generator in caffe')
    gen_net = caffe_util.Net.from_param(
        gen_net_param, args.gen_weights_file, phase=caffe.TEST
    )

    if args.all_blobs:
        args.blob_name = [b for b in gen_net.blobs]

    generate_from_model(gen_net, data_param, len(examples), args)


if __name__ == '__main__':
    main(sys.argv[1:])

