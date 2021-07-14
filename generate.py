from __future__ import print_function
import sys, os, re, argparse, time, gzip, yaml, tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 32)
pd.set_option('display.max_rows', 120)
import torch
from rdkit import Chem

import liGAN
from liGAN import molecules as mols
from liGAN import metrics
from liGAN.atom_grids import AtomGrid
from liGAN.atom_structs import AtomStruct


MB = 1024 ** 2


class OutputWriter(object):
    '''
    A data structure for receiving and sorting AtomGrids and
    AtomStructs from a generative model or atom fitting algorithm,
    computing metrics, and writing files to disk as necessary.
    '''
    def __init__(
        self,
        out_prefix,
        output_dx,
        output_sdf,
        output_types,
        output_latent,
        output_visited,
        output_conv,
        n_samples,
        grid_types,
        batch_metrics,
        verbose
    ):
        self.out_prefix = out_prefix
        self.output_dx = output_dx
        self.output_sdf = output_sdf
        self.output_types = output_types
        self.output_latent = output_latent
        self.output_visited = output_visited
        self.output_conv = output_conv
        self.n_samples = n_samples
        self.grid_types = grid_types
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

        self.verbose = verbose

        # keep sdf files open so that all samples of a given
        #   struct or molecule can be written to one file
        self.open_files = dict()

        # create directories for output files
        if output_latent:
            self.latent_dir = Path('latents')
            self.latent_dir.mkdir(exist_ok=True)

        if output_dx:
            self.grid_dir = Path('grids')
            self.grid_dir.mkdir(exist_ok=True)

        if output_sdf:
            self.struct_dir = Path('structs')
            self.struct_dir.mkdir(exist_ok=True)

            self.mol_dir = Path('molecules')
            self.mol_dir.mkdir(exist_ok=True)

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def close_files(self):
        '''
        Close all open files that the output
        writer currently has a reference to
        and delete the references.
        '''
        for f, out in list(self.open_files.items()):
            out.close()
            del self.open_files[f]

    def write_sdf(self, sdf_file, mol, sample_idx, is_real):
        '''
        Append molecule or atom sturct to sdf_file.

        NOTE this method assumes that samples will be
        produced in sequential order (i.e. not async)
        because it opens the file on first sample_idx
        and closes it on the last one.
        '''
        if sdf_file not in self.open_files:
            self.open_files[sdf_file] = gzip.open(sdf_file, 'wt')
        out = self.open_files[sdf_file]

        if sample_idx == 0 or not is_real:
            self.print('Writing {} sample {}'.format(sdf_file, sample_idx))
                
            if isinstance(mol, AtomStruct):
                struct = mol
                if self.output_visited and 'visited_structs' in struct.info:
                    visited_structs = struct.info['visited_structs']
                    rd_mols = [s.to_rd_mol() for s in visited_structs]
                else:
                    rd_mols = [struct.to_rd_mol()]

            else: # molecule
                if self.output_visited and 'visited_mols' in mol.info:
                    rd_mols = mol.info['visited_mols']
                else:
                    rd_mols = [mol]

            try:
                mols.write_rd_mols_to_sdf_file(
                    out, rd_mols, str(sample_idx), kekulize=False
                )
            except ValueError:
                print(sdf_file, sample_idx, is_real)
                raise

        if sample_idx == 0:
            self.sdf_files.append(sdf_file)
        
        if sample_idx + 1 == self.n_samples or is_real:
            out.close()
            del self.open_files[sdf_file]

    def write_atom_types(self, types_file, atom_types):

        self.print('Writing ' + str(types_file))
        write_atom_types_to_file(types_file, atom_types)

    def write_dx(self, dx_prefix, grid):

        self.print('Writing {} .dx files'.format(dx_prefix))

        grid.to_dx(dx_prefix)
        self.dx_prefixes.append(dx_prefix)

    def write_latent(self, latent_file, latent_vec):

        self.print('Writing ' + str(latent_file))
        write_latent_vecs_to_file(latent_file, [latent_vec])

    def write(self, lig_name, grid_type, sample_idx, grid):
        '''
        Write output files for grid and compute metrics in
        data frame, if all necessary data is present.
        '''
        grid_prefix = '{}_{}_{}'.format(self.out_prefix, lig_name, grid_type)
        i = str(sample_idx)

        assert grid_type in {
            'rec', 'lig', 'lig_gen', 'lig_fit', 'lig_gen_fit'
        }
        is_lig_grid = grid_type.startswith('lig')
        is_gen_grid = grid_type.endswith('_gen')
        is_fit_grid = grid_type.endswith('_fit')

        is_real_grid = not (is_gen_grid or is_fit_grid)
        is_first_real_grid = (is_real_grid and sample_idx == 0)
        has_struct = (is_real_grid or is_fit_grid)
        has_conv_grid = not is_fit_grid # and is_lig_grid ?

        # write atomic structs and molecules
        if has_struct and self.output_sdf:

            # get struct that created this grid (via molgrid.GridMaker)
            #   note that depending on the grid_type, this can either be
            #   from atom fitting OR from typing a real molecule
            struct = grid.info['src_struct']

            # the real (source) molecule and atom types don't change
            #   between different samples, so only write them once

            # and we don't apply bond adding to the receptor struct,
            #   so only ligand structs have add_mol and min_mol

            if is_first_real_grid: # write real molecule

                sdf_file = self.mol_dir / (grid_prefix + '_src.sdf.gz')
                src_mol = struct.info['src_mol']
                self.write_sdf(sdf_file, src_mol, sample_idx, is_real=True)

                if 'min_mol' in src_mol.info:
                    sdf_file = self.mol_dir / (grid_prefix+'_src_uff.sdf.gz')
                    min_mol = src_mol.info['min_mol']
                    self.write_sdf(sdf_file, min_mol, sample_idx, is_real=True)

            # write typed atomic structure (real or fit)
            if is_first_real_grid or is_fit_grid:

                sdf_file = self.struct_dir / (grid_prefix + '.sdf.gz')
                self.write_sdf(sdf_file, struct, sample_idx, is_real_grid)

                if self.output_types: # write atom type channels

                    types_base = grid_prefix + '_' + i + '.atom_types'
                    types_file = self.struct_dir / types_base
                    self.write_atom_types(types_file, struct.atom_types)

            # write bond-added molecule (real or fit, no rec bond adding)
            if is_lig_grid and (
                is_first_real_grid or is_fit_grid
            ):
                sdf_file = self.mol_dir / (grid_prefix + '_add.sdf.gz')
                add_mol = struct.info['add_mol']
                self.write_sdf(sdf_file, add_mol, sample_idx, is_real_grid)                            

                sdf_file = self.mol_dir / (grid_prefix + '_add_uff.sdf.gz')
                min_mol = add_mol.info['min_mol']
                self.write_sdf(sdf_file, min_mol, sample_idx, is_real_grid)

        # write atomic density grids
        if self.output_dx:

            dx_prefix = self.grid_dir / (grid_prefix + '_' + i)
            self.write_dx(dx_prefix, grid)

            if has_conv_grid and self.output_conv: # write convolved grid

                dx_prefix = self.grid_dir / (grid_prefix + '_conv_' + i)
                self.write_dx(dx_prefix, grid.info['conv_grid'])                  

        # write latent vectors
        if is_gen_grid and self.output_latent:

            latent_file = self.latent_dir / (grid_prefix + '_' + i + '.latent')
            self.write_latent(latent_file, grid.info['src_latent'])

        # store grid until ready to compute output metrics
        self.grids[lig_name][sample_idx][grid_type] = grid
        lig_grids = self.grids[lig_name]

        if self.batch_metrics: # store until grids for all samples are ready

            has_all_samples = (len(lig_grids) == self.n_samples)
            has_all_grids = all(
                set(lig_grids[i]) == set(self.grid_types) for i in lig_grids
            )

            if has_all_samples and has_all_grids: # compute batch metrics

                self.print('Computing metrics for all '+lig_name+' samples')
                try:
                    self.compute_metrics(lig_name, range(self.n_samples))
                except:
                    self.close_files()
                    raise

                self.print('Writing ' + self.metric_file)
                self.metrics.to_csv(self.metric_file, sep=' ')

                self.print('Writing ' + self.pymol_file)
                write_pymol_script(
                    self.pymol_file,
                    self.out_prefix,
                    self.dx_prefixes,
                    self.sdf_files,
                )
                del self.grids[lig_name] # free memory

        else: # only store until grids for this sample are ready
            has_all_grids = set(lig_grids[sample_idx]) == set(self.grid_types)

            if has_all_grids: # compute sample metrics

                self.print('Computing metrics for {} sample {}'.format(
                    lig_name, sample_idx
                ))
                try:
                    self.compute_metrics(lig_name, [sample_idx])
                except:
                    self.close_files()
                    raise

                self.print('Writing ' + self.metric_file)
                self.metrics.to_csv(self.metric_file, sep=' ')

                self.print('Writing ' + self.pymol_file)
                write_pymol_script(
                    self.pymol_file,
                    self.out_prefix,
                    self.dx_prefixes,
                    self.sdf_files,
                )
                del self.grids[lig_name][sample_idx] # free memory

    def compute_metrics(self, lig_name, sample_idxs):
        '''
        Compute metrics for density grids, typed atomic structures,
        and molecules for a given ligand in metrics data frame.
        '''
        lig_grids = self.grids[lig_name]
        has_rec = ('rec' in self.grid_types)
        has_lig_gen = ('lig_gen' in self.grid_types)
        has_lig_fit = ('lig_fit' in self.grid_types)
        has_lig_gen_fit = ('lig_gen_fit' in self.grid_types)

        if self.batch_metrics: # compute mean grids and type counts

            def get_mean_type_counts(struct_batch):
                n = len(struct_batch)
                type_counts = sum([s.type_counts for s in struct_batch]) / n
                elem_counts = sum([s.elem_counts for s in struct_batch]) / n
                prop_counts = sum([s.prop_counts for s in struct_batch]) / n
                return type_counts, elem_counts, prop_counts

            lig_grid_batch = [lig_grids[i]['lig'].values for i in sample_idxs]
            lig_grid_mean = sum(lig_grid_batch) / self.n_samples

            lig_struct_batch = [lig_grids[i]['lig'].info['src_struct'] for i in sample_idxs]
            lig_mean_counts = get_mean_type_counts(lig_struct_batch)

            if has_lig_fit:
                lig_fit_struct_batch = [lig_grids[i]['lig_fit'].info['src_struct'] for i in sample_idxs]
                lig_fit_mean_counts = get_mean_type_counts(lig_fit_struct_batch)

            if has_lig_gen:
                lig_gen_grid_mean = sum(lig_grids[i]['lig_gen'].values for i in sample_idxs) / self.n_samples
                lig_latent_mean = sum(lig_grids[i]['lig_gen'].info['src_latent'] for i in sample_idxs) / self.n_samples

                if has_lig_gen_fit:
                    lig_gen_fit_struct_batch = [lig_grids[i]['lig_gen_fit'].info['src_struct'] for i in sample_idxs]
                    lig_gen_fit_mean_counts = get_mean_type_counts(lig_gen_fit_struct_batch)
        else:
            lig_grid_mean = None
            lig_mean_counts = None
            lig_fit_mean_counts = None
            lig_gen_grid_mean = None
            lig_latent_mean = None
            lig_gen_fit_mean_counts = None

        for sample_idx in sample_idxs:
            idx = (lig_name, sample_idx)

            rec_grid = lig_grids[sample_idx]['rec'] if has_rec else None
            lig_grid = lig_grids[sample_idx]['lig']
            self.compute_grid_metrics(idx,
                grid_type='lig',
                grid=lig_grid,
                mean_grid=lig_grid_mean,
                cond_grid=rec_grid,
            )

            lig_struct = lig_grid.info['src_struct']
            self.compute_struct_metrics(idx,
                struct_type='lig',
                struct=lig_struct,
                mean_counts=lig_mean_counts,
            )

            lig_mol = lig_struct.info['src_mol']
            self.compute_mol_metrics(idx,
                mol_type='lig', mol=lig_mol
            )

            if has_lig_fit or has_lig_gen_fit:

                lig_add_mol = lig_struct.info['add_mol']
                self.compute_mol_metrics(idx,
                    mol_type='lig_add', mol=lig_add_mol, ref_mol=lig_mol
                )

            if has_lig_gen:

                lig_gen_grid = lig_grids[sample_idx]['lig_gen']
                self.compute_grid_metrics(idx,
                    grid_type='lig_gen',
                    grid=lig_gen_grid,
                    ref_grid=lig_grid,
                    mean_grid=lig_gen_grid_mean,
                    cond_grid=rec_grid
                )

                lig_latent = lig_gen_grid.info['src_latent']
                self.compute_latent_metrics(idx,
                    latent_type='lig',
                    latent=lig_latent,
                    mean_latent=lig_latent_mean
                )

            if has_lig_fit:

                lig_fit_grid = lig_grids[sample_idx]['lig_fit']
                self.compute_grid_metrics(idx,
                    grid_type='lig_fit',
                    grid=lig_fit_grid,
                    ref_grid=lig_grid,
                    cond_grid=rec_grid,
                )

                lig_fit_struct = lig_fit_grid.info['src_struct']
                self.compute_struct_metrics(idx,
                    struct_type='lig_fit',
                    struct=lig_fit_struct,
                    ref_struct=lig_struct,
                    mean_counts=lig_fit_mean_counts,
                )

                lig_fit_add_mol = lig_fit_struct.info['add_mol']
                self.compute_mol_metrics(idx,
                    mol_type='lig_fit_add',
                    mol=lig_fit_add_mol,
                    ref_mol=lig_mol,
                )

                lig_fit_add_struct = lig_fit_add_mol.info['type_struct']
                self.compute_struct_metrics(idx,
                    struct_type='lig_fit_add',
                    struct=lig_fit_add_struct,
                    ref_struct=lig_struct,
                )

            if has_lig_gen_fit:

                lig_gen_fit_grid = lig_grids[sample_idx]['lig_gen_fit']
                self.compute_grid_metrics(idx,
                    grid_type='lig_gen_fit',
                    grid=lig_gen_fit_grid,
                    ref_grid=lig_gen_grid,
                    cond_grid=rec_grid,
                )

                lig_gen_fit_struct = lig_gen_fit_grid.info['src_struct']
                self.compute_struct_metrics(idx,
                    struct_type='lig_gen_fit',
                    struct=lig_gen_fit_struct,
                    ref_struct=lig_struct,
                    mean_counts=lig_gen_fit_mean_counts,
                )

                lig_gen_fit_add_mol = lig_gen_fit_struct.info['add_mol']
                self.compute_mol_metrics(idx,
                    mol_type='lig_gen_fit_add',
                    mol=lig_gen_fit_add_mol,
                    ref_mol=lig_mol,
                )

                lig_gen_fit_add_struct = lig_gen_fit_add_mol.info['type_struct']
                self.compute_struct_metrics(idx,
                    struct_type='lig_gen_fit_add',
                    struct=lig_gen_fit_add_struct,
                    ref_struct=lig_struct,
                )

        self.print(self.metrics.loc[lig_name].loc[sample_idxs].transpose())

    def compute_grid_metrics(
        self,
        idx,
        grid_type,
        grid,
        ref_grid=None,
        mean_grid=None,
        cond_grid=None,
    ):
        m = self.metrics

        # density magnitude
        m.loc[idx, grid_type+'_grid_norm'] = grid.values.norm().item()
        m.loc[idx, grid_type+'_grid_elem_norm'] = grid.elem_values.norm().item()
        m.loc[idx, grid_type+'_grid_prop_norm'] = grid.prop_values.norm().item()

        if mean_grid is not None:

            # density variance
            # (divide by n_samples (+1) for sample (population) variance)
            m.loc[idx, grid_type+'_grid_variance'] = (
                (grid.values - mean_grid)**2
            ).sum().item()

        if ref_grid is not None:

            # density L2 loss
            m.loc[idx, grid_type+'_L2_loss'] = (
                (ref_grid.values - grid.values)**2
            ).sum().item() / 2

            m.loc[idx, grid_type+'_elem_L2_loss'] = (
                (ref_grid.elem_values - grid.elem_values)**2
            ).sum().item() / 2

            m.loc[idx, grid_type+'_prop_L2_loss'] = (
                (ref_grid.prop_values - grid.prop_values)**2
            ).sum().item() / 2

        if cond_grid is not None:

            # density product
            m.loc[idx, grid_type+'_rec_prod'] = (
                cond_grid.values.sum(dim=0) * grid.values.sum(dim=0)
            ).sum().item()

            m.loc[idx, grid_type+'_rec_elem_prod'] = (
                cond_grid.elem_values.sum(dim=0) * grid.elem_values.sum(dim=0)
            ).sum().item()

            m.loc[idx, grid_type+'_rec_prop_prod'] = (
                cond_grid.prop_values.sum(dim=0) * grid.prop_values.sum(dim=0)
            ).sum().item()


    def compute_latent_metrics(
        self, idx, latent_type, latent, mean_latent=None
    ):
        m = self.metrics

        # latent vector magnitude
        m.loc[idx, latent_type+'_latent_norm'] = latent.norm().item()

        if mean_latent is not None:

            # latent vector variance
            variance = (
                (latent - mean_latent)**2
            ).sum().item()
        else:
            variance = np.nan

        m.loc[idx, latent_type+'_latent_variance'] = variance

    def compute_struct_metrics(
        self, idx, struct_type, struct, ref_struct=None, mean_counts=None,
    ):
        m = self.metrics

        m.loc[idx, struct_type+'_n_atoms'] = struct.n_atoms
        m.loc[idx, struct_type+'_radius'] = (
            struct.radius if struct.n_atoms > 0 else np.nan
        )

        if mean_counts is not None:

            mean_type_counts, mean_elem_counts, mean_prop_counts = \
                mean_counts

            m.loc[idx, struct_type+'_type_variance'] = (
                (struct.type_counts - mean_type_counts)**2
            ).sum().item()

            m.loc[idx, struct_type+'_elem_variance'] = (
                (struct.elem_counts - mean_elem_counts)**2
            ).sum().item()

            m.loc[idx, struct_type+'_prop_variance'] = (
                (struct.prop_counts - mean_prop_counts)**2
            ).sum().item()

        if ref_struct is not None:

            # difference in num atoms
            m.loc[idx, struct_type+'_n_atoms_diff'] = (
                ref_struct.n_atoms - struct.n_atoms
            )

            # overall type count difference
            m.loc[idx, struct_type+'_type_diff'] = (
                ref_struct.type_counts - struct.type_counts
            ).norm(p=1).item()

            # element type count difference
            m.loc[idx, struct_type+'_elem_diff'] = (
                ref_struct.elem_counts - struct.elem_counts
            ).norm(p=1).item()

            # property type count difference
            m.loc[idx, struct_type+'_prop_diff'] = (
                ref_struct.prop_counts - struct.prop_counts
            ).norm(p=1).item()

            # minimum atom-only RMSD (ignores properties)
            rmsd = metrics.compute_struct_rmsd(ref_struct, struct)
            m.loc[idx, struct_type+'_RMSD'] = rmsd

        if struct_type.endswith('_fit'):

            # fit time and number of visited structures
            m.loc[idx, struct_type+'_time'] = struct.info['time']
            m.loc[idx, struct_type+'_n_visited'] = len(
                struct.info['visited_structs']
            )

            # accuracy of estimated type counts, whether or not
            # they were actually used to constrain atom fitting
            est_type = struct_type[:-4] + '_est'
            m.loc[idx, est_type+'_type_diff'] = struct.info.get(
                'est_type_diff', np.nan
            )
            m.loc[idx, est_type+'_exact_types'] = (
                m.loc[idx, est_type+'_type_diff'] == 0
            )

    def compute_mol_metrics(self, idx, mol_type, mol, ref_mol=None):
        m = self.metrics

        # check molecular validity
        valid, reason = mol.validate()
        m.loc[idx, mol_type+'_n_atoms'] = mol.n_atoms
        m.loc[idx, mol_type+'_n_frags'] = mol.n_frags
        m.loc[idx, mol_type+'_valid'] = valid
        m.loc[idx, mol_type+'_reason'] = reason

        # other molecular descriptors
        m.loc[idx, mol_type+'_MW'] = mols.get_rd_mol_weight(mol)
        m.loc[idx, mol_type+'_logP'] = mols.get_rd_mol_logP(mol)
        m.loc[idx, mol_type+'_QED'] = mols.get_rd_mol_QED(mol)
        if valid:
            m.loc[idx, mol_type+'_SAS'] = mols.get_rd_mol_SAS(mol)
            m.loc[idx, mol_type+'_NPS'] = mols.get_rd_mol_NPS(mol)
        else:
            m.loc[idx, mol_type+'_SAS'] = np.nan
            m.loc[idx, mol_type+'_NPS'] = np.nan

        # convert to SMILES string
        smi = mol.to_smi()
        m.loc[idx, mol_type+'_SMILES'] = smi

        if ref_mol: # compare to ref_mol

            # difference in num atoms
            m.loc[idx, mol_type+'_n_atoms_diff'] = (
                ref_mol.n_atoms - mol.n_atoms
            )

            ref_valid, ref_reason = ref_mol.validate()

            # get reference SMILES strings
            ref_smi = ref_mol.to_smi()
            m.loc[idx, mol_type+'_SMILES_match'] = (smi == ref_smi)

            if valid and ref_valid: # fingerprint similarity

                m.loc[idx, mol_type+'_ob_sim'] = \
                    mols.get_ob_smi_similarity(ref_smi, smi)
                m.loc[idx, mol_type+'_rdkit_sim'] = \
                    mols.get_rd_mol_similarity(ref_mol, mol, 'rdkit')
                m.loc[idx, mol_type+'_morgan_sim'] = \
                    mols.get_rd_mol_similarity(ref_mol, mol, 'morgan')
                m.loc[idx, mol_type+'_maccs_sim'] = \
                    mols.get_rd_mol_similarity(ref_mol, mol, 'maccs')
            else:
                m.loc[idx, mol_type+'_ob_sim'] = np.nan
                m.loc[idx, mol_type+'_rdkit_sim'] = np.nan
                m.loc[idx, mol_type+'_morgan_sim'] = np.nan
                m.loc[idx, mol_type+'_maccs_sim'] = np.nan

        if 'min_mol' not in mol.info:
            return

        # UFF energy minimization
        min_mol = mol.info['min_mol']
        E_init = min_mol.info['E_init']
        E_min = min_mol.info['E_min']

        m.loc[idx, mol_type+'_E'] = E_init
        m.loc[idx, mol_type+'_min_E'] = E_min
        m.loc[idx, mol_type+'_dE_min'] = E_min - E_init
        m.loc[idx, mol_type+'_min_error'] = min_mol.info['min_error']
        m.loc[idx, mol_type+'_min_time'] = min_mol.info['min_time']
        m.loc[idx, mol_type+'_RMSD_min'] = min_mol.aligned_rmsd(mol)

        # compare energy to ref mol, before and after minimizing
        if ref_mol:

            min_ref_mol = ref_mol.info['min_mol']
            E_init_ref = min_ref_mol.info['E_init']
            E_min_ref = min_ref_mol.info['E_init']

            m.loc[idx, mol_type+'_dE_ref'] = E_init - E_init_ref
            m.loc[idx, mol_type+'_min_dE_ref'] = E_min - E_min_ref

            # get aligned RMSD to ref mol, pre-minimize
            m.loc[idx, mol_type+'_RMSD_ref'] = ref_mol.aligned_rmsd(mol)

            # get aligned RMSD to true mol, post-minimize
            m.loc[idx, mol_type+'_min_RMSD_ref'] = min_ref_mol.aligned_rmsd(min_mol)


def find_real_rec_in_data_root(data_root, rec_src_no_ext):

    # cross-docked set
    m = re.match(r'(.+)(_0)?', rec_src_no_ext)
    rec_mol_base = m.group(1) + '.pdb'
    rec_mol_file = os.path.join(data_root, rec_mol_base)
    rec_mol = mols.Molecule.from_pdb(rec_mol_file, sanitize=False)
    try:
        Chem.SanitizeMol(rec_mol)
    except Chem.MolSanitizeException:
        pass
    return rec_mol


def find_real_lig_in_data_root(data_root, lig_src_no_ext, use_ob=False):
    '''
    Try to find the real molecule in data_root using the
    source path in the data file, without file extension.
    '''
    try: # PDBbind
        m = re.match(r'(.+)_ligand_(\d+)', lig_src_no_ext)
        lig_mol_base = m.group(1) + '_docked.sdf.gz'
        idx = int(m.group(2))
        lig_mol_file = os.path.join(data_root, lig_mol_base)

    except AttributeError:
        try: # cross-docked set
            m = re.match(r'(.+)_(\d+)', lig_src_no_ext)
            lig_mol_base = m.group(1) + '.sdf'
            idx = int(m.group(2))
            lig_mol_file = os.path.join(data_root, lig_mol_base)
            os.stat(lig_mol_file)

        except OSError:
            lig_mol_base = lig_src_no_ext + '.sdf'
            lig_mol_file = os.path.join(data_root, lig_mol_base)
            idx = 0

    if use_ob: # read and add Hs with OpenBabel, then convert to RDkit
        lig_mol = mols.read_ob_mols_from_file(lig_mol_file, 'sdf')[idx]
        lig_mol.AddHydrogens()
        lig_mol = mols.Molecule.from_ob_mol(lig_mol)

    else: # read and add Hs with RDKit (need to sanitize before add Hs)
        lig_mol = mols.Molecule.from_sdf(lig_mol_file, sanitize=False, idx=idx)

    try: # need to do this to get ring info, etc.
        lig_mol.sanitize()
    except Chem.MolSanitizeException:
        pass

    if not use_ob: # add Hs with rdkit (after sanitize)
        lig_mol = lig_mol.add_hs()

    return lig_mol


def write_latent_vecs_to_file(latent_file, latent_vecs):

    with open(latent_file, 'w') as f:
        for v in latent_vecs:
            line = ' '.join('{:.5f}'.format(x) for x in v) + '\n'
            f.write(line)


def rec_and_lig_at_index_in_data_file(file, index):
    '''
    Read receptor and ligand names at a specific line number in a data file.
    '''
    with open(file, 'r') as f:
        line = f.readlines()[index]
    cols = line.rstrip().split()
    return cols[2], cols[3]


def n_lines_in_file(file):
    '''
    Count the number of lines in a file.
    '''
    with open(file, 'r') as f:
        return sum(1 for line in f)


def write_pymol_script(
    pymol_file, out_prefix, dx_prefixes, sdf_files
):
    '''
    Write a pymol script that loads all .dx files with a given
    prefix into a single group, then loads a set of sdf_files
    and translates them to the origin, if centers are provided.
    '''
    with open(pymol_file, 'w') as f:

        for dx_prefix in dx_prefixes: # load density grids
            try:
                m = re.match(
                    '^grids/({}_.*)$'.format(re.escape(out_prefix)),
                    str(dx_prefix)
                )
            except AttributeError:
                print(dx_prefix, file=sys.stderr)
                raise
            group_name = m.group(1) + '_grids'
            dx_pattern = '{}_*.dx'.format(dx_prefix)
            f.write('load_group {}, {}\n'.format(dx_pattern, group_name))

        for sdf_file in sdf_files: # load structs/molecules
            try:
                m = re.match(
                    r'^(molecules|structs)/({}_.*)\.sdf(\.gz)?$'.format(
                        re.escape(out_prefix)
                    ),
                    str(sdf_file)
                )
                obj_name = m.group(2)
            except AttributeError:
                print(sdf_file, file=sys.stderr)
                raise
            f.write('load {}, {}\n'.format(sdf_file, obj_name))


def write_atom_types_to_file(types_file, atom_types):
    with open(types_file, 'w') as f:
        f.write('\n'.join(str(a) for a in atom_types))
        

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


def slerp(v0, v1, t):
    '''
    Spherical linear interpolation between
    vectors v0 and v1 at steps t.
    '''
    norm_v0 = v0.norm()
    norm_v1 = v1.norm()
    dot_v0_v1 = (v0*v1).sum(1)
    cos_theta = dot_v0_v1 / (norm_v0 * norm_v1)
    theta = torch.acos(cos_theta) # angle between the vectors
    sin_theta = torch.sin(theta)
    k0 = torch.sin((1-t)*theta) / sin_theta
    k1 = torch.sin(t*theta) / sin_theta
    return (
        k0[:,None] * v0[None,:] + k1[:,None] * v1[None,:]
    )


def lerp(v0, v1, t):
    '''
    Linear interpolation between vectors v0 and v1 at steps t.
    '''
    k0, k1 = (1-t), t
    return (
        k0[:,None] * v0[None,:] + k1[:,None] * v1[None,:]
    )


def generate(
    data,
    gen_model,
    prior_model,
    atom_fitter,
    bond_adder,
    out_writer,
    n_examples,
    n_samples,
    fit_atoms,
    prior=False,
    stage2=False,
    z_score=None,
    truncate=None,
    var_factor=1.0,
    interpolate=False,
    spherical=False,
    verbose=True,
):
    '''
    Generate atomic density grids from gen_model for
    each example in data and fit atomic structures.
    '''
    batch_size = data.batch_size

    print('Starting to generate grids')
    for example_idx in range(n_examples): # iterate over data rows

        for sample_idx in range(n_samples): # multiple samples per data row

            # keep track of position in current batch
            full_idx = example_idx*n_samples + sample_idx
            batch_idx = full_idx % batch_size

            if interpolate:
                endpoint_idx = 0 if batch_idx < batch_size//2 else -1

            is_first = (example_idx == sample_idx == 0)
            need_next_batch = (batch_idx == 0)

            if need_next_batch: # forward next batch

                with torch.no_grad():

                    if verbose: print('Getting next batch of data')
                    grids, structs, _ = data.forward(split_rec_lig=True)
                    rec_structs, lig_structs = structs
                    rec_grids, lig_grids = grids
                    rec_lig_grids = data.grids

                    if gen_model:
                        if verbose:
                            print(f'Calling generator forward (prior={prior}, stage2={stage2})')

                        if stage2: # insert prior model
                            lig_gen_grids, _, _, _, latents, _, _ = \
                                gen_model.forward2(
                                    prior_model=prior_model,
                                    inputs=None if prior else rec_lig_grids,
                                    conditions=rec_grids,
                                    batch_size=batch_size,
                                    z_score=z_score,
                                    truncate=truncate,
                                    var_factor=var_factor,
                                )
                        else:
                            lig_gen_grids, latents, _, _ = gen_model(
                                inputs=None if prior else rec_lig_grids,
                                conditions=rec_grids,
                                batch_size=batch_size,
                                z_score=z_score,
                                truncate=truncate,
                                var_factor=var_factor,
                            )

                        # TODO interpolation here!
                        assert not interpolate, 'TODO'

                        for i, name in enumerate(data.lig_typer.get_type_names()):
                            print(name, '\t', (lig_gen_grids[:,i]**2).sum().item())

            rec_struct = rec_structs[batch_idx]
            lig_struct = lig_structs[batch_idx]
            lig_center = lig_struct.center

            # undo transform so structs are all aligned
            transform = data.transforms[batch_idx]
            transform.backward(rec_struct.coords, rec_struct.coords)
            transform.backward(lig_struct.coords, lig_struct.coords)

            # only process real rec/lig once, since they're
            # the same for all samples of a given ligand
            if sample_idx == 0:
                if verbose: print('Getting real molecule from data root')
                my_split_ext = lambda f: f.rsplit('.', 1 + f.endswith('.gz'))

                rec_src_file = rec_struct.info['src_file']
                rec_src_no_ext = my_split_ext(rec_src_file)[0]
                rec_mol = find_real_rec_in_data_root(
                    data.root_dir, rec_src_no_ext
                )

                lig_src_file = lig_struct.info['src_file']
                lig_src_no_ext = my_split_ext(lig_src_file)[0]
                lig_name = os.path.basename(lig_src_no_ext)
                lig_mol = find_real_lig_in_data_root(
                    data.root_dir, lig_src_no_ext, use_ob=True
                )

                if fit_atoms: # add bonds and minimize

                    if verbose: print('Minimizing real molecule')
                    lig_mol.info['min_mol'] = lig_mol.uff_minimize()

                    if verbose: print('Making molecule from real atoms')
                    lig_add_mol, lig_add_struct, _ = bond_adder.make_mol(lig_struct)
                    lig_add_mol.info['type_struct'] = lig_add_struct
                    lig_add_mol.info['min_mol'] = lig_add_mol.uff_minimize()

            rec_struct.info['src_mol'] = rec_mol
            lig_struct.info['src_mol'] = lig_mol
            if fit_atoms:
                lig_struct.info['add_mol'] = lig_add_mol

            grid_types = [
                ('rec', rec_grids, data.rec_typer),
                ('lig', lig_grids, data.lig_typer),  
            ]
            if gen_model:
                grid_types += [
                    ('lig_gen', lig_gen_grids, data.lig_typer)
                ]

            for grid_type, grids, atom_typer in grid_types:
                torch.cuda.reset_max_memory_allocated()

                is_lig_grid = (grid_type.startswith('lig'))
                is_gen_grid = (grid_type.endswith('gen'))
                grid_needs_fit = (is_lig_grid and fit_atoms and (
                    is_gen_grid or not gen_model
                ))

                grid = liGAN.atom_grids.AtomGrid(
                    values=grids[batch_idx],
                    typer=atom_typer,
                    center=lig_center,
                    resolution=data.resolution
                )

                if grid_type == 'rec':
                    grid.info['src_struct'] = rec_struct
                elif grid_type == 'lig':
                    grid.info['src_struct'] = lig_struct
                elif grid_type == 'lig_gen':
                    grid.info['src_latent'] = latents[batch_idx]

                # display progress
                index_str = (
                    '[example_idx={} sample_idx={} lig_name={} grid_type={}]'\
                        .format(example_idx, sample_idx, lig_name, grid_type)
                )
                value_str = 'norm={:.4f} gpu={:.4f}'.format(
                    grid.values.norm(),
                    torch.cuda.max_memory_allocated() / MB,
                )
                print(index_str + ' ' + value_str, flush=True)

                if out_writer.output_conv:
                    grid.info['conv_grid'] = grid.new_like(
                        values=torch.cat([
                            atom_fitter.convolve(
                                grid.elem_values, grid.resolution, grid.typer
                            ),
                            grid.prop_values
                        ], dim=0)
                    )

                out_writer.write(lig_name, grid_type, sample_idx, grid)

                if grid_needs_fit: # atom fitting, bond adding, minimize

                    fit_struct, fit_grid, visited_structs = atom_fitter.fit_struct(
                        grid, lig_struct.type_counts
                    )
                    fit_struct.info['visited_structs'] = visited_structs

                    if fit_struct.n_atoms > 0: # undo transform
                        transform.backward(fit_struct.coords, fit_struct.coords)

                    fit_add_mol, fit_add_struct, visited_mols = bond_adder.make_mol(fit_struct)
                    fit_add_mol.info['type_struct'] = fit_add_struct
                    fit_add_mol.info['min_mol'] = fit_add_mol.uff_minimize()
                    fit_struct.info['add_mol'] = fit_add_mol
                    fit_grid.info['src_struct'] = fit_struct

                    out_writer.write(
                        lig_name, grid_type + '_fit', sample_idx, fit_grid
                    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Generate atomic density grids from generative model'
    )
    parser.add_argument('config_file')
    parser.add_argument('--debug', default=False, action='store_true')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    device = 'cuda'
    liGAN.set_random_seed(config.get('random_seed', None))

    print('Loading data')
    data_file = config['data'].pop('data_file')
    data = liGAN.data.AtomGridData(
        device=device,
        n_samples=config['generate']['n_samples'],
        **config['data']
    )
    data.populate(data_file)

    if 'model_type' in config:
        print('Initializing generative model')
        gen_model_type = getattr(liGAN.models, config.pop('model_type'))
        gen_model_state = config['gen_model'].pop('state')
        gen_model = gen_model_type(
            n_channels_in=(data.n_lig_channels + data.n_rec_channels),
            n_channels_cond=data.n_rec_channels,
            n_channels_out=data.n_lig_channels,
            grid_size=data.grid_size,
            device=device,
            **config['gen_model']
        )
        print('Loading generative model state')
        state_dict = torch.load(gen_model_state)
        state_dict.pop('log_recon_var', None)
        gen_model.load_state_dict(state_dict)

        if gen_model_type.has_stage2:

            print('Initializing prior model')
            prior_model_state = config['prior_model'].pop('state')
            prior_model = liGAN.models.Stage2VAE(
                n_input=gen_model.n_latent,
                **config['prior_model']
            ).to(device)

            print('Loading prior model state')
            state_dict = torch.load(prior_model_state)
            state_dict.pop('log_recon_var', None)
            prior_model.load_state_dict(state_dict)
        else:
            prior_model = None
    else:
        gen_model = None
        prior_model = None
        print('No generative model, using real grids')

    print('Initializing atom fitter')
    atom_fitter = liGAN.atom_fitting.AtomFitter(
        device=device, **config.get('atom_fitting', {})
    )

    print('Initializing bond adder')
    bond_adder = liGAN.bond_adding.BondAdder(
       **config.get('bond_adding', {})
    )

    # determine generated grid types
    grid_types = ['rec', 'lig']
    if gen_model:
        grid_types += ['lig_gen']
        if config['generate']['fit_atoms']:
            grid_types += ['lig_gen_fit']
    else:
        if config['generate']['fit_atoms']:
            grid_types += ['lig_fit']

    print('Initializing output writer')
    out_writer = OutputWriter(
        out_prefix=config['out_prefix'],
        n_samples=config['generate']['n_samples'],
        grid_types=grid_types,
        verbose=config['verbose'],
        **config['output']
    )

    generate(
        data=data,
        gen_model=gen_model,
        prior_model=prior_model,
        atom_fitter=atom_fitter,
        bond_adder=bond_adder,
        out_writer=out_writer,
        verbose=config['verbose'],
        **config['generate']
    )
    print('Done')


if __name__ == '__main__':
    main(sys.argv[1:])
