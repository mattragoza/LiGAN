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
import traceback
try:
    from itertools import izip
except ImportError:
    izip = zip
from functools import partial
from scipy.stats import multivariate_normal

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from GPUtil import getGPUs



class Encoder(cu.CaffeSubNet):

    def __init__(self, net, start, end, variational):
        super().__init__(net, start, end)
        self.variational = variational

    @classmethod
    def find_in_net(cls, net, input_):

        start = find_blobs_in_net(net, input_)[0]
        try:
            end = find_blobs_in_net(net, input_+'_latent_std')[0]
            variational = True
        except IndexError:
            try:
                end = find_blobs_in_net(net, input_+'_latent_defc')[0]
            except IndexError:
                end = find_blobs_in_net(net, input_+'_latent_fc')[0]
            variational = False

        return cls(net, start, end, variational)


class LatentSpace(cu.CaffeSubNet):

    @classmethod
    def find_in_net(cls, net, input_):
        try:
            start = end = find_blobs_in_net(net, input_+'_latent_defc')[0]
        except IndexError:
            start = end = find_blobs_in_net(net, input_+'_latent_fc')[0]
        return cls(net, start, end)

    @property
    def size(self):
        return self.net.blobs[self.end].shape[1]


class LatentVariable(LatentSpace):

    def __init__(self, net, mean, std, noise, sample):
        super().__init__(net, start=mean, end=sample)
        self.mean = mean
        self.std = std
        self.noise = noise
        self.sample = sample

    @classmethod
    def find_in_net(cls, net, input_):
        mean = find_blobs_in_net(net, input_+'_mean')[0]
        std = find_blobs_in_net(net, input_+'_std')[0]
        noise = find_blobs_in_net(net, input_+'_noise')[0]
        sample = find_blobs_in_net(net, input_+'_sample')[0]
        return cls(net, mean, std, noise, sample)


class Decoder(cu.CaffeSubNet):

    @classmethod
    def find_in_net(cls, net, n_inputs, output):

        if n_inputs > 1:
            start = find_blobs_in_net(net, 'latent_concat')[0]
        else:
            start = find_blobs_in_net(net, output+'_dec_fc')[0]

        end = find_blobs_in_net(net, output+'_gen')[0]

        return cls(net, start, end)


class Generator(object):

    def __init__(self, net, forward=True, verbose=False):

        # net should be a CaffeNet
        self.net = net
        self.verbose = verbose

        self.variational = False
        self.encoders = {}
        self.latent = None
        self.decoder = None
        self.losses = {}

        if verbose:
            print('Finding important blobs in generator')
        self.find_sub_nets()

        if forward:
            if verbose:
                print('Testing generator forward')
            # this is necessary for proper latent sampling
            self.forward()

    def find_sub_nets(self):

        self.variational = False

        # find rec, lig, and/or rec+lig (data) encoders
        self.encoders = {}
        for encoder_input in ['rec', 'lig', 'data']:
            try:
                encoder = Encoder.find_in_net(self.net, encoder_input)
            except IndexError:
                continue

            assert not (self.variational and encoder.variational), \
                'cannot have more than one variational encoder'

            self.encoders[encoder_input] = encoder

            if encoder.variational:
                self.variational = True

            latent_input = encoder_input

        assert len(self.encoders) > 0, \
            'must have at least one encoder'

        # find latent space
        if self.variational:
            self.latent = LatentVariable.find_in_net(self.net, latent_input)
        else:
            self.latent = LatentSpace.find_in_net(self.net, latent_input)

        # find lig decoder
        self.decoder = Decoder.find_in_net(
            self.net, n_inputs=len(self.encoders), output='lig'
        )

        # find loss functions
        self.losses = {}
        for l in find_blobs_in_net(self.net, '.*_loss'):
            self.losses[l] = cu.CaffeSubNet(self.net, start='lig_gen', end=l)

    def forward(self, prior=False, **kwargs):

        for i in self.encoders:
            self.encoders[i].forward(kwargs.get(i, None))

        if self.variational:
            self.latent.forward()

        self.decoder.forward()

        for l in self.losses:
            self.losses[l].forward()

    def backward(self):

        for l in self.losses:
            self.losses[l].backward()

        self.decoder.backward()

        if self.variational:
            self.latent.backward()

        for i in self.encoders:
            self.encoders[i].backward()

    def generate(self, data, n_examples, n_samples):
        pass # TODO

    def print_norms(self):
        print('data_norm diff_norm blob_name')
        for b in self.net.blobs:
            data_norm = np.linalg.norm(self.net.blobs[b].data)
            diff_norm = np.linalg.norm(self.net.blobs[b].diff)
            print('{:9.2f} {:9.2f} {}'.format(
                data_norm, diff_norm, b  
            ))




class OutputWriter(object):
    '''
    A data structure for receiving and organizing AtomGrids and
    AtomStructs from a generative model or atom fitting algorithm,
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
        output_conv,
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
        self.output_conv = output_conv
        self.n_samples = n_samples
        self.blob_names = blob_names
        self.fit_atoms = fit_atoms
        self.batch_metrics = batch_metrics

        # organize grids by lig_name, sample_idx, grid_type
        self.grids = defaultdict(lambda: defaultdict(dict))

        # accumulate metrics in dataframe
        self.metric_file = '{}.gen_metrics'.format(out_prefix)
        columns = ['lig_name', 'sample_idx']
        self.metrics = pd.DataFrame(columns=columns).set_index(columns)

        # write a pymol script when finished
        self.pymol_file = '{}.pymol'.format(out_prefix)
        self.dx_prefixes = []
        self.sdf_files = []
        self.centers = []

        self.verbose = verbose
        
        self.out_files = dict() # one file for all samples of given mol/grid

    def write(self, lig_name, grid_type, sample_idx, grid):
        '''
        Write output files for grid and compute metrics in
        data frame, if all necessary data is present.
        '''
        grid_prefix = '{}_{}_{}'.format(self.out_prefix, lig_name, grid_type)

        sample_prefix = grid_prefix + '_' + str(sample_idx)
        src_sample_prefix = grid_prefix + '_src_' + str(sample_idx)
        add_sample_prefix = grid_prefix + '_add_' + str(sample_idx)
        src_uff_sample_prefix = grid_prefix + '_src_uff_' + str(sample_idx)
        add_uff_sample_prefix = grid_prefix + '_add_uff_' + str(sample_idx)
        conv_sample_prefix = grid_prefix + '_conv_' + str(sample_idx)

        is_gen_grid = grid_type.endswith('_gen')
        is_fit_grid = grid_type.endswith('_fit')
        is_real_grid = not (is_gen_grid or is_fit_grid)
        has_struct = is_real_grid or is_fit_grid
        has_conv_grid = not is_fit_grid
        is_generated = is_gen_grid or grid_type.endswith('gen_fit')

        # write output files
        if self.output_dx: # write out density grids

            if self.verbose:
                print('Writing ' + sample_prefix + ' .dx files')

            grid.to_dx(sample_prefix, center=np.zeros(3))
            self.dx_prefixes.append(sample_prefix)

            if has_conv_grid and self.output_conv:

                if self.verbose:
                    print('Writing' + conv_sample_prefix + ' .dx files')

                grid.info['conv_grid'].to_dx(conv_sample_prefix, center=np.zeros(3))
                self.dx_prefixes.append(conv_sample_prefix)

        if has_struct and self.output_sdf: # write out structures

            struct = grid.info['src_struct']

            # write atom type channels
            if self.output_channels:

                channels_file = sample_prefix + '.channels'
                if self.verbose:
                    print('Writing ' + channels_file)

                write_channels_to_file(
                    channels_file, struct.c, struct.channels
                )
               
            src_fname = grid_prefix + '_src.sdf.gz'         
            start_fname = grid_prefix + '.sdf.gz'
            add_fname = grid_prefix + '_add.sdf.gz'
            min_fname = grid_prefix + '_uff.sdf.gz'

            def write_sdfs(fname, out, mol):
                '''
                Write out sdf files as necessary.
                '''
                if sample_idx == 0:
                    self.sdf_files.append(fname)
                    self.centers.append(struct.center)

                if sample_idx == 0 or not is_real_grid:

                    if self.verbose:
                        print('Writing %s %d'%(fname, sample_idx))
                        
                    if mol == struct:
                        if self.output_visited and 'visited_structs' in struct.info:
                            rd_mols = [s.to_rd_mol() for s in struct.info['visited_structs']]
                        else:
                            rd_mols = [struct.to_rd_mol()]
                    else:
                        if self.output_visited and 'visited_mols' in mol.info:
                            rd_mols = mol.info['visited_mols']
                        else:
                            rd_mols = [mol]
                    molecules.write_rd_mols_to_sdf_file(out, rd_mols, str(sample_idx))
                
                if sample_idx+1 == self.n_samples or is_real_grid:
                    out.close()

            if grid_prefix not in self.out_files:

                self.out_files[grid_prefix] = {}

                # open output files for generated samples                
                self.out_files[grid_prefix][start_fname] = gzip.open(start_fname, 'wt')
                self.out_files[grid_prefix][add_fname] = gzip.open(add_fname, 'wt')                                
                self.out_files[grid_prefix][min_fname] = gzip.open(min_fname, 'wt')

                # only output src file once
                if is_real_grid: # write real input molecule                  
                    src_file = gzip.open(src_fname, 'wt')
                    src_mol = struct.info['src_mol']
                    write_sdfs(src_fname, src_file, src_mol)

            # write typed atom structure
            write_sdfs(start_fname, self.out_files[grid_prefix][start_fname], struct)

            # write molecule with added bonds
            add_mol = struct.info['add_mol']
            write_sdfs(add_fname, self.out_files[grid_prefix][add_fname], add_mol)                            

            if add_mol.info['min_mol']: # write minimized molecule
                write_sdfs(min_fname, self.out_files[grid_prefix][min_fname], add_mol.info['min_mol'])                   

        # write latent vector
        if is_gen_grid and self.output_latent:

            latent_file = sample_prefix + '.latent'
            if self.verbose:
                print('Writing ' + latent_file)

            latent_vec = grid.info['latent_vec']
            write_latent_vecs_to_file(latent_file, [latent_vec])

        # store grid until ready to compute output metrics
        self.grids[lig_name][sample_idx][grid_type] = grid
        lig_grids = self.grids[lig_name]

        # determine how many grid_types to expect
        n_grid_types = len(self.blob_names)
        if self.fit_atoms:
            n_grid_types *= 2

        if self.batch_metrics: # store until grids for all samples are ready

            has_all_samples = (len(lig_grids) == self.n_samples)
            has_all_grids = all(len(lig_grids[i]) == n_grid_types for i in lig_grids)

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
                    self.pymol_file,
                    self.out_prefix,
                    self.dx_prefixes,
                    self.sdf_files,
                    self.centers,
                )
                del self.grids[lig_name]

        else: # only store until grids for this sample are ready

            has_all_grids = len(lig_grids[sample_idx]) == n_grid_types

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
                    self.pymol_file,
                    self.out_prefix,
                    self.dx_prefixes,
                    self.sdf_files,
                    self.centers,
                )
                del self.grids[lig_name][sample_idx]

    def compute_metrics(self, lig_name, sample_idxs):
        '''
        Compute metrics for density grids, fit atom types, and
        bonded molecules for a given ligand in metrics data frame.
        '''
        lig_grids = self.grids[lig_name]

        if self.batch_metrics:

            lig_grid_mean = sum(
                lig_grids[i]['lig'].values for i in sample_idxs
            ) / self.n_samples

            lig_gen_grid_mean = sum(
                lig_grids[i]['lig_gen'].values for i in sample_idxs
            ) / self.n_samples

            lig_latent_mean = sum(
                lig_grids[i]['lig_gen'].info['latent_vec'] for i in sample_idxs
            ) / self.n_samples

        else:
            lig_grid_mean = None
            lig_gen_grid_mean = None
            lig_latent_mean = None

        for sample_idx in sample_idxs:
            idx = (lig_name, sample_idx)

            lig_grid = lig_grids[sample_idx]['lig']
            lig_gen_grid = lig_grids[sample_idx]['lig_gen']

            self.compute_grid_metrics(idx, 'lig', lig_grid, mean_grid=lig_grid_mean)
            self.compute_grid_metrics(idx, 'lig_gen', lig_gen_grid, lig_grid, lig_gen_grid_mean)

            lig_latent = lig_gen_grid.info['latent_vec']
            self.compute_latent_metrics(idx, 'lig', lig_latent, lig_latent_mean)

            if self.fit_atoms:

                lig_fit_grid = lig_grids[sample_idx]['lig_fit']
                lig_gen_fit_grid = lig_grids[sample_idx]['lig_gen_fit']

                self.compute_grid_metrics(idx, 'lig_fit', lig_fit_grid, lig_grid)
                self.compute_grid_metrics(idx, 'lig_gen_fit', lig_gen_fit_grid, lig_gen_grid)

                lig_struct = lig_grid.info['src_struct']
                lig_fit_struct = lig_fit_grid.info['src_struct']
                lig_gen_fit_struct = lig_gen_fit_grid.info['src_struct']

                self.compute_struct_metrics(idx, 'lig', lig_struct)
                self.compute_struct_metrics(idx, 'lig_fit', lig_fit_struct, lig_struct)
                self.compute_struct_metrics(idx, 'lig_gen_fit', lig_gen_fit_struct, lig_struct)

                lig_mol = lig_struct.info['src_mol']
                lig_add_mol = lig_struct.info['add_mol']
                lig_fit_add_mol = lig_fit_struct.info['add_mol']
                lig_gen_fit_add_mol = lig_gen_fit_struct.info['add_mol']

                self.compute_mol_metrics(idx, 'lig', lig_mol)
                self.compute_mol_metrics(idx, 'lig_add', lig_add_mol, lig_mol)
                self.compute_mol_metrics(idx, 'lig_fit_add', lig_fit_add_mol, lig_mol)
                self.compute_mol_metrics(idx, 'lig_gen_fit_add', lig_gen_fit_add_mol, lig_mol)

        if self.verbose:
            print(self.metrics.loc[lig_name].loc[sample_idxs])

    def compute_grid_metrics(self, idx, grid_type, grid, ref_grid=None, mean_grid=None):
        m = self.metrics

        # density magnitude
        m.loc[idx, grid_type+'_norm'] = np.linalg.norm(grid.values)

        if mean_grid is not None:

            # density variance
            # (divide by n_samples (+1) for sample (population) variance)
            variance = (
                (grid.values - mean_grid)**2
            ).sum().item()
        else:
            variance = np.nan

        m.loc[idx, grid_type+'_variance'] = variance

        if ref_grid is not None:

            # density L2 loss
            m.loc[idx, grid_type+'_L2_loss'] = (
                (ref_grid.values - grid.values)**2
            ).sum().item() / 2

            # density L1 loss
            m.loc[idx, grid_type+'_L1_loss'] = (
                np.abs(ref_grid.values - grid.values)
            ).sum().item()

    def compute_latent_metrics(self, idx, latent_type, latent, mean_latent=None):
        m = self.metrics

        # latent vector magnitude
        m.loc[idx, latent_type+'_latent_norm'] = np.linalg.norm(latent)

        if mean_latent is not None:

            # latent vector variance
            variance = (
                (latent - mean_latent)**2
            ).sum()
        else:
            variance = np.nan

        m.loc[idx, latent_type+'_latent_variance'] = variance

    def compute_struct_metrics(self, idx, struct_type, struct, ref_struct=None):
        m = self.metrics

        # number of atoms
        m.loc[idx, struct_type+'_n_atoms'] = struct.n_atoms

        # maximum radius
        m.loc[idx, struct_type+'_radius'] = struct.radius

        if ref_struct is not None:

            # get atom type counts
            n_types = len(struct.channels)
            types = count_types(struct.c, n_types)
            ref_types = count_types(ref_struct.c, n_types)

            # type count difference
            m.loc[idx, struct_type+'_type_diff'] = np.linalg.norm(
                ref_types - types, ord=1
            )
            m.loc[idx, struct_type+'_exact_types'] = (
                m.loc[idx, struct_type+'_type_diff'] == 0
            )

            # minimum typed-atom RMSD
            try:
                rmsd = get_min_rmsd(
                    ref_struct.xyz, ref_struct.c, struct.xyz, struct.c
                )
            except (ValueError, ZeroDivisionError):
                rmsd = np.nan

            m.loc[idx, struct_type+'_RMSD'] = rmsd

        if struct_type.endswith('_fit'):

            # fit time and number of visited structures
            m.loc[idx, struct_type+'_time'] = struct.info['time']
            m.loc[idx, struct_type+'_n_visited'] = len(struct.info['visited_structs'])

            # accuracy of estimated type counts, whether or not
            # they were actually used to constrain atom fitting
            est_type = struct_type[:-4] + '_est'
            m.loc[idx, est_type+'_type_diff'] = struct.info.get('est_type_diff', np.nan)
            m.loc[idx, est_type+'_exact_types'] = (
                m.loc[idx, est_type+'_type_diff'] == 0
            )

    def compute_mol_metrics(self, idx, mol_type, mol, ref_mol=None):
        m = self.metrics

        # standardize mols
        mol_info = mol.info
        mol = Chem.RemoveHs(mol, sanitize=False)
        mol.info = mol_info

        # check molecular validity
        n_frags, error, valid = get_rd_mol_validity(mol)
        m.loc[idx, mol_type+'_n_frags'] = n_frags
        m.loc[idx, mol_type+'_error'] = error
        m.loc[idx, mol_type+'_valid'] = valid

        # other molecular descriptors
        m.loc[idx, mol_type+'_MW'] = get_rd_mol_weight(mol)
        m.loc[idx, mol_type+'_logP'] = get_rd_mol_logP(mol)
        m.loc[idx, mol_type+'_QED'] = get_rd_mol_QED(mol)
        m.loc[idx, mol_type+'_SAS'] = get_rd_mol_SAS(mol)
        m.loc[idx, mol_type+'_NPS'] = get_rd_mol_NPS(mol, nps_model)

        # convert to SMILES string
        smi = get_smiles_string(mol)
        m.loc[idx, mol_type+'_SMILES'] = smi

        if ref_mol is not None: # compare to ref_mol

            ref_mol_info = ref_mol.info
            ref_mol = Chem.RemoveHs(ref_mol, sanitize=False)
            ref_mol.info = ref_mol_info

            # get reference SMILES strings
            ref_smi = get_smiles_string(ref_mol)
            m.loc[idx, mol_type+'_SMILES_match'] = (smi == ref_smi)

            # fingerprint similarity
            m.loc[idx, mol_type+'_ob_sim']  = get_ob_smi_similarity(
                ref_smi, smi
            )
            m.loc[idx, mol_type+'_morgan_sim'] = get_rd_mol_similarity(
                ref_mol, mol, 'morgan'
            )
            m.loc[idx, mol_type+'_rdkit_sim']  = get_rd_mol_similarity(
                ref_mol, mol, 'rdkit'
            )
            m.loc[idx, mol_type+'_maccs_sim']  = get_rd_mol_similarity(
                ref_mol, mol, 'maccs'
            )

        # UFF energy minimization
        min_mol = mol.info['min_mol']
        E_init = mol.info['E_init']
        E_min = mol.info['E_min']

        m.loc[idx, mol_type+'_E'] = E_init
        m.loc[idx, mol_type+'_min_E'] = E_min
        m.loc[idx, mol_type+'_dE_min'] = E_min - E_init
        m.loc[idx, mol_type+'_min_error'] = mol.info['min_error']
        m.loc[idx, mol_type+'_min_time'] = mol.info['min_time']
        m.loc[idx, mol_type+'_RMSD_min']  = get_aligned_rmsd(min_mol, mol)

        if ref_mol is not None:

            # compare energy to ref mol, pre and post-minimization
            min_ref_mol = ref_mol.info['min_mol']
            E_init_ref = ref_mol.info['E_init']
            E_min_ref = ref_mol.info['E_init']

            m.loc[idx, mol_type+'_dE_ref'] = E_init - E_init_ref
            m.loc[idx, mol_type+'_min_dE_ref'] = E_min - E_min_ref

            # get aligned RMSD to ref mol, pre-minimize
            m.loc[idx, mol_type+'_RMSD_ref'] = get_aligned_rmsd(ref_mol, mol)

            # get aligned RMSD to true mol, post-minimize
            m.loc[idx, mol_type+'_min_RMSD_ref'] = get_aligned_rmsd(min_ref_mol, min_mol)
               

def find_real_mol_in_data_root(data_root, lig_src_no_ext):
    '''
    Try to find the real molecule in data_root using the
    source path found in the data file, without extension.
    '''
    try: # docked PDBbind ligands are gzipped together
        m = re.match(r'(.+)_ligand_(\d+)', lig_src_no_ext)
        lig_mol_base = m.group(1) + '_docked.sdf.gz'
        idx = int(m.group(2))
        lig_mol_file = os.path.join(data_root, lig_mol_base)
        lig_mol = molecules.read_rd_mols_from_sdf_file(lig_mol_file)[idx]

    except AttributeError:

        try: # cross-docked set has extra underscore
            lig_mol_base = lig_src_no_ext + '_.sdf'
            lig_mol_file = os.path.join(data_root, lig_mol_base)
            lig_mol = molecules.read_rd_mols_from_sdf_file(lig_mol_file)[0]

        except OSError:
            lig_mol_base = lig_src_no_ext + '.sdf'
            lig_mol_file = os.path.join(data_root, lig_mol_base)
            lig_mol = molecules.read_rd_mols_from_sdf_file(lig_mol_file)[0]

    return lig_mol


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


def write_pymol_script(pymol_file, out_prefix, dx_prefixes, sdf_files, centers=[]):
    '''
    Write a pymol script that loads all .dx files with a given
    prefix into a single group, then loads a set of sdf_files
    and translates them to the origin, if centers are provided.
    '''
    with open(pymol_file, 'w') as f:

        for dx_prefix in dx_prefixes: # load densities
            dx_pattern = '{}_*.dx'.format(dx_prefix)
            m = re.match('^({}_.*)$'.format(out_prefix), dx_prefix)
            group_name = m.group(1) + '_grids'
            f.write('load_group {}, {}\n'.format(dx_pattern, group_name))

        for sdf_file in sdf_files: # load structures
            m = re.match(r'^({}_.*)\.sdf(\.gz)?$'.format(out_prefix), sdf_file)
            obj_name = m.group(1)
            f.write('load {}, {}\n'.format(sdf_file, obj_name))

        for sdf_file, (x,y,z) in zip(sdf_files, centers): # center structures
            m = re.match(r'^({}_.*)\.sdf(\.gz)?$'.format(out_prefix), sdf_file)
            obj_name = m.group(1)
            f.write('translate [{},{},{}], {}, camera=0, state=0\n'.format(-x, -y, -z, obj_name))





def write_channels_to_file(channels_file, c, channels):
    with open(channels_file, 'w') as f:
        for c_ in c:
            channel = channels[c_]
            f.write(channel.name+'\n')


def read_channels_from_file(channels_file, channels):
    channel_map = {
        ch.name: i for i, ch in enumerate(channels)
    }
    c = []
    with open(channels_file, 'r') as f:
        for line in f:
            channel_name = line.rstrip()
            c.append(channel_map[channel_name])
    return np.array(c)


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


def read_examples_from_data_file(data_file, data_root='', n=None):
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
            if n is not None and len(examples) == n:
                break
    return examples


def find_blobs_in_net(net, blob_pattern):
    '''
    Find all blob_names in net that match blob_pattern.
    '''
    return re.findall('^{}$'.format(blob_pattern), '\n'.join(net.blobs), re.MULTILINE)


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

    generator = MolGridGenerator(gen_net)

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
            output_conv=args.output_conv,
            n_samples=args.n_samples,
            batch_metrics=args.batch_metrics,
            blob_names=args.blob_name,
            fit_atoms=args.fit_atoms,
            verbose=args.verbose,
        )

        if args.fit_atoms or args.output_conv:

            if args.dkoes_simple_fit:
                atom_fitter = DkoesAtomFitter(args.dkoes_make_mol, args.use_openbabel)
            else:
                atom_fitter = AtomFitter(
                    multi_atom=args.multi_atom,
                    n_atoms_detect=args.n_atoms_detect,
                    beam_size=args.beam_size,
                    apply_conv=args.apply_conv,
                    threshold=args.threshold,
                    peak_value=args.peak_value,
                    min_dist=args.min_dist,
                    constrain_types=args.constrain_types,
                    constrain_frags=False,
                    estimate_types=args.estimate_types,
                    fit_L1_loss=args.fit_L1_loss,
                    interm_gd_iters=args.interm_gd_iters,
                    final_gd_iters=args.final_gd_iters,
                    gd_kwargs=dict(
                        lr=args.learning_rate,
                        betas=(args.beta1, args.beta2),
                        weight_decay=args.weight_decay,
                    ),
                    dkoes_make_mol=args.dkoes_make_mol,
                    use_openbabel=args.use_openbabel,
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

                    if args.interpolate: # interpolate between real grids

                        start_rec = np.array(rec[:1])
                        start_lig = np.array(lig[:1])
                        end_rec = np.array(rec[-1:])
                        end_lig = np.array(lig[-1:])

                        gen_net.blobs['rec'].data[...] = np.linspace(
                            start_rec, end_rec, batch_size, endpoint=True
                        )
                        gen_net.blobs['lig'].data[...] = np.linspace(
                            start_lig, end_lig, batch_size, endpoint=True
                        )

                    if has_rec_enc: # forward receptor encoder
                        if rec_enc_is_var:
                            if args.prior:
                                gen_net.blobs[latent_mean].data[...] = 0.0
                                gen_net.blobs[latent_std].data[...] = args.var_factor
                            else: # posterior
                                gen_net.forward(start=rec_enc_start, end=rec_enc_end)
                                if args.var_factor != 1.0:
                                    gen_net.blobs[latent_std].data[...] *= args.var_factor
                        else:
                            gen_net.forward(start=rec_enc_start, end=rec_enc_end)

                    if has_lig_enc: # forward ligand encoder
                        if lig_enc_is_var:
                            if args.prior:
                                gen_net.blobs[latent_mean].data[...] = 0.0
                                gen_net.blobs[latent_std].data[...] = args.var_factor
                            else: # posterior
                                gen_net.forward(start=lig_enc_start, end=lig_enc_end)
                                if args.var_factor != 1.0:
                                    gen_net.blobs[latent_std].data[...] *= args.var_factor
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

                # get current example ligand
                if args.interpolate:
                    ex = examples[endpoint_idx]
                else:
                    ex = examples[batch_idx]

                print('Getting real atom types and coords')
                lig_coord_set = ex.coord_sets[1]
                lig_src = lig_coord_set.src
                lig_struct = AtomStruct.from_coord_set(lig_coord_set, lig_channels)
                types = count_types(lig_struct.c, lig_map.num_types(), dtype=np.int16)

                lig_src_no_ext = os.path.splitext(lig_src)[0]
                lig_name = os.path.basename(lig_src_no_ext)

                if not args.gen_only:
                    print('Getting real molecule from data root')
                    lig_mol = find_real_mol_in_data_root(args.data_root, lig_src_no_ext)

                    if args.fit_atoms:
                        print('Real molecule for {} has {} atoms'.format(lig_name, lig_struct.n_atoms))

                        print('Minimizing real molecule')
                        atom_fitter.uff_minimize(lig_mol)
                        lig_struct.info['src_mol'] = lig_mol

                        print('Validifying real atom types and coords')
                        atom_fitter.validify(lig_struct)

                # get latent vector for current example
                latent_vec = np.array(gen_net.blobs[latent_sample].data[batch_idx])

                # get data from blob, process, and write output
                for blob_name in args.blob_name:

                    print('Getting grid from {} blob'.format(blob_name))
                    if args.gen_only and blob_name != 'lig_gen':
                        continue

                    grid_type = blob_name
                    grid_blob = gen_net.blobs[blob_name]
                    grid_needs_fit = args.fit_atoms and blob_name in {'lig', 'lig_gen'}

                    if blob_name == 'rec':
                        grid_channels = rec_channels
                    elif blob_name in {'lig', 'lig_gen'}:
                        grid_channels = lig_channels
                    else:
                        grid_channels = atom_types.get_n_unknown_channels(
                            grid_blob.shape[1]
                        )

                    grid_data = grid_blob.data[batch_idx]

                    grid = AtomGrid(
                        values=np.array(grid_data),
                        channels=grid_channels,
                        center=lig_struct.center,
                        resolution=grid_maker.get_resolution(),
                    )
                    grid_norm = np.linalg.norm(grid.values)

                    if grid_type == 'lig': # store true structure for input ligand grids
                        grid.info['src_struct'] = lig_struct

                    elif grid_type == 'lig_gen': # store latent vector for generated grids
                        grid.info['latent_vec'] = latent_vec

                    if args.verbose:
                        gpu_usage = get_gpu_usage(0)

                        print('Produced {} {} {} (norm={}\tGPU={})'.format(
                            lig_name, grid_type.ljust(7), sample_idx, grid_norm, gpu_usage
                        ), flush=True)

                    if args.output_conv:
                        grid.info['conv_grid'] = grid.new_like(
                            values=atom_fitter.convolve(
                                torch.tensor(grid.values, device=atom_fitter.device),
                                grid.channels,
                                grid.resolution,
                            ).cpu().detach().numpy()
                        )

                    if args.parallel:
                        out_queue.put((lig_name, grid_type, sample_idx, grid))
                        if grid_needs_fit:
                            fit_queue.put((lig_name, grid_type, sample_idx, grid, types))
                    else:
                        out_writer.write(lig_name, grid_type, sample_idx, grid)
                        if grid_needs_fit:
                            grid = atom_fitter.fit(grid, types)
                            grid_type = grid_type + '_fit'
                            out_writer.write(lig_name, grid_type, sample_idx, grid)
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
        atom_fitter = DkoesAtomFitter(
            dkoes_make_mol=args.dkoes_make_mol,
            use_openbabel=args.use_openbabel,
        )
    else:
        atom_fitter = AtomFitter(
            multi_atom=args.multi_atom,
            n_atoms_detect=args.n_atoms_detect,
            beam_size=args.beam_size,
            apply_conv=args.apply_conv,
            threshold=args.threshold,
            peak_value=args.peak_value,
            min_dist=args.min_dist,
            constrain_types=args.constrain_types,
            constrain_frags=False,
            estimate_types=args.estimate_types,
            fit_L1_loss=args.fit_L1_loss,
            interm_gd_iters=args.interm_gd_iters,
            final_gd_iters=args.final_gd_iters,
            gd_kwargs=dict(
                lr=args.learning_rate,
                betas=(args.beta1, args.beta2),
                weight_decay=args.weight_decay,
            ),
            dkoes_make_mol=args.dkoes_make_mol,
            use_openbabel=args.use_openbabel,
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

        lig_name, grid_type, sample_idx, grid, types = task

        if args.verbose:
            print('Fit worker got {} {} {}'.format(lig_name, grid_type, sample_idx))

        out_queue.put((lig_name, grid_type, sample_idx, grid))

        grid = atom_fitter.fit(grid, types)
        grid_type = grid_type + '_fit'

        if args.verbose:
            print('Fit worker produced {} {} {} ({} atoms, {}s)'.format(
                lig_name, grid_type, sample_idx, struct_fit.n_atoms, struct_fit.fit_time
            ), flush=True)

        out_queue.put((lig_name, grid_type, sample_idx, grid))

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
        output_conv=args.output_conv,
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
    parser.add_argument('--var_factor', default=1.0, type=float, help='factor by which to multiply standard deviation of latent samples')
    parser.add_argument('--encode_first', default=False, action='store_true', help='generate all output from encoding first example')
    parser.add_argument('--condition_first', default=False, action='store_true', help='condition all generated output on first example')
    parser.add_argument('--interpolate', default=False, action='store_true', help='interpolate between examples in latent space')
    parser.add_argument('--spherical', default=False, action='store_true', help='use spherical interpolation instead of linear')
    parser.add_argument('-o', '--out_prefix', required=True, help='common prefix for output files')
    parser.add_argument('--output_dx', action='store_true', help='output .dx files of atom density grids for each channel')
    parser.add_argument('--output_sdf', action='store_true', help='output .sdf file of best fit atom positions')
    parser.add_argument('--output_conv', action='store_true', help='output .dx files of atom density grids convolved with kernel')
    parser.add_argument('--output_visited', action='store_true', help='output every visited structure in .sdf files')
    parser.add_argument('--output_kernel', action='store_true', help='output .dx files for kernel used to intialize atoms during atom fitting')
    parser.add_argument('--output_channels', action='store_true', help='output channels of each fit structure in separate files')
    parser.add_argument('--output_latent', action='store_true', help='output latent vectors for each generated density grid')
    parser.add_argument('--fit_atoms', action='store_true', help='fit atoms to density grids and print the goodness-of-fit')
    parser.add_argument('--dkoes_simple_fit', default=False, action='store_true', help='fit atoms using alternate functions by dkoes')
    parser.add_argument('--dkoes_make_mol', default=False, action='store_true', help="validify molecule using alternate functions by dkoes")
    parser.add_argument('--use_openbabel', default=False, action='store_true', help="validify molecule using OpenBabel only")
    parser.add_argument('--constrain_types', action='store_true', help='constrain atom fitting to find atom types of true ligand (or estimate)')
    parser.add_argument('--estimate_types', action='store_true', help='estimate atom type counts using the total grid density per channel')
    parser.add_argument('--multi_atom', default=False, action='store_true', help='add all next atoms to grid simultaneously at each atom fitting step')
    parser.add_argument('--n_atoms_detect', default=1, type=int, help='max number of atoms to detect in each atom fitting step')
    parser.add_argument('--beam_size', type=int, default=1, help='number of best structures to track during atom fitting beam search')
    parser.add_argument('--apply_conv', default=False, action='store_true', help='apply convolution to grid before detecting next atoms')
    parser.add_argument('--threshold', type=float, default=0.1, help='threshold value for detecting next atoms on grid')
    parser.add_argument('--peak_value', type=float, default=1.5, help='reflect grid values higher than this value before detecting next atoms')
    parser.add_argument('--min_dist', type=float, default=0.0, help='minimum distance between detected atoms, in terms of covalent bond length')
    parser.add_argument('--fit_L1_loss', default=False, action='store_true')
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
    parser.add_argument('--gen_only',action='store_true',help='Only produce generated molecules; do not perform fitting on true ligand')
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

    # read data params
    data_net_param = cu.NetParameter.from_prototxt(args.data_model_file)
    assert data_net_param.layer[0].type == 'MolGridData'
    data_param = data_net_param.layer[0].molgrid_data_param

    # read model params
    gen_net_param = caffe_util.NetParameter.from_prototxt(args.gen_model_file)

    data = MolGridData(
        data_root=args.data_root,
        batch_size=args.batch_size,
        rec_map_file=data_param.recmap,
        lig_map_file=data_param.ligmap,
        resolution=data_param.resolution,
        dimension=data_param.dimension,
        shuffle=args.shuffle,
        random_rotation=args.random_rotation,
        random_translate=args.random_translate,
        rec_molcache=data_param.rec_molcache,
        lig_molcache=data_param.lig_molcache,
    )

    if not args.data_file:
        assert len(args.rec_file) == len(args.lig_file)
        args.data_file = get_temp_data_file(zip(args.rec_file, args.lig_file))

    data.populate(args.data_file)

    if args.gpu:
        print('Setting caffe to GPU mode')
        cu.caffe.set_mode_gpu()
    else:
        print('Setting caffe to CPU mode')
        cu.caffe.set_mode_cpu()

    # create the net in caffe
    print('Constructing generator in caffe')
    gen_net = caffe_util.Net.from_param(
        gen_net_param, args.gen_weights_file, phase=caffe.TEST
    )

    mgrid_gen = MolGridGenerator(gen_net)
    mgrid_gen.generate()


if __name__ == '__main__':
    main(sys.argv[1:])

