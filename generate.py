from __future__ import print_function
import sys, os, re, argparse, time, gzip, yaml, tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem

import liGAN
from liGAN import molecules, metrics
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
        output_channels,
        output_latent,
        output_visited,
        output_conv,
        n_samples,
        n_grid_types,
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
        self.n_grid_types = n_grid_types
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
        # struct or molecule can be written to one file
        self.open_files = dict()

        # create directories for output grids and structs
        if output_dx:
            self.grid_dir = Path('grids')
            self.grid_dir.mkdir(exist_ok=True)

        if output_sdf:
            self.struct_dir = Path('structs')
            self.struct_dir.mkdir(exist_ok=True)

        if output_latent:
            self.latent_dir = Path('latents')
            self.latent_dir.mkdir(exist_ok=True)

    def write_sdf(self, sdf_file, mol, sample_idx, is_real):
        '''
        Append molecule or atom sturct tp sdf_file.

        NOTE this method assumes that samples will be
        produced in sequential order (i.e. not async)
        because it opens the file on first sample_idx
        and closes it on the last one.
        '''
        if sdf_file not in self.open_files:
            self.open_files[sdf_file] = gzip.open(sdf_file, 'wt')
        out = self.open_files[sdf_file]

        if sample_idx == 0 or not is_real:

            if self.verbose:
                print('Writing {} sample {}'.format(sdf_file, sample_idx))
                
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

            molecules.write_rd_mols_to_sdf_file(
                out, rd_mols, str(sample_idx), kekulize=False
            )

        if sample_idx == 0:
            self.sdf_files.append(sdf_file)
        
        if sample_idx + 1 == self.n_samples or is_real:
            out.close()

    def write_channels(self, channels_file, c):

        if self.verbose:
            print('Writing ' + str(channels_file))

        write_channels_to_file(channels_file, c)

    def write_dx(self, dx_prefix, grid):

        if self.verbose:
            print('Writing {} .dx files'.format(dx_prefix))

        grid.to_dx(grid_prefix, center=(0,0,0))
        self.dx_prefixes.append(dx_prefix)

    def write_latent(self, latent_file, latent_vec):

        if self.verbose:
            print('Writing ' + str(latent_file))

        write_latent_vecs_to_file(latent_file, [latent_vec])

    def write(self, lig_name, grid_type, sample_idx, grid):
        '''
        Write output files for grid and compute metrics in
        data frame, if all necessary data is present.
        '''
        grid_prefix = '{}_{}_{}'.format(self.out_prefix, lig_name, grid_type)
        i = str(sample_idx)

        is_lig_grid = grid_type.startswith('lig')
        is_gen_grid = grid_type.endswith('_gen')
        is_fit_grid = grid_type.endswith('_fit')
        is_real_grid = not (is_gen_grid or is_fit_grid)
        is_first_real_grid = (is_real_grid and sample_idx == 0)
        has_struct = (is_real_grid or is_fit_grid)
        has_conv_grid = not is_fit_grid
        is_generated = is_gen_grid or grid_type.endswith('gen_fit')

        # write atom type stucts and molecules
        if has_struct and self.output_sdf:

            # the struct that created this grid (via molgrid.GridMaker)
            # note that depending on the grid_type, this can be either
            # a fit structure OR from a real molecule
            struct = grid.info['src_struct']

            # note that the real (source) molecule and atom types don't
            # change between different samples, so only write them once

            # and we don't apply bond adding to the rec struct,
            # so only lig structs have add_mol and min_mol

            # write real molecule
            if is_first_real_grid:

                sdf_file = self.struct_dir / (grid_prefix + '_src.sdf.gz')
                src_mol = struct.info['src_mol']
                self.write_sdf(sdf_file, src_mol, sample_idx, is_real=True)

                if is_lig_grid: # no rec minimization
                    sdf_file = self.struct_dir/(grid_prefix+'_src_uff.sdf.gz')
                    min_mol = src_mol.info['min_mol']
                    self.write_sdf(sdf_file, min_mol, sample_idx, is_real=True)

            # write typed atomic structure (real or fit)
            if is_first_real_grid or is_fit_grid:

                sdf_file = self.struct_dir / (grid_prefix + '.sdf.gz')
                self.write_sdf(sdf_file, struct, sample_idx, is_real_grid)

                # write atom type channels
                if self.output_channels:

                    channels_base = grid_prefix + '_' + i + '.channels'
                    channels_file = self.struct_dir / channels_base
                    self.write_channels(channels_file, struct.c)

            # write bond-added molecule (real or fit, no rec bond adding)
            if is_lig_grid and (
                is_first_real_grid or is_fit_grid
            ):
                sdf_file = self.struct_dir / (grid_prefix + '_add.sdf.gz')
                add_mol = struct.info['add_mol']
                self.write_sdf(sdf_file, add_mol, sample_idx, is_real_grid)                            

                sdf_file = self.struct_dir / (grid_prefix + '_add_uff.sdf.gz')
                min_mol = add_mol.info['min_mol']
                self.write_sdf(sdf_file, min_mol, sample_idx, is_real_grid)

        # write atomic density grids
        if self.output_dx:

            dx_prefix = self.grid_dir / (grid_prefix + '_' + i)
            self.write_dx(dx_prefix, grid)

            if has_conv_grid and self.output_conv:

                dx_prefix = self.grid_dir / (grid_prefix + '_conv_' + i)
                self.write_dx(dx_prefix, grid.info['conv_grid'])                  

        # write latent vector
        if is_gen_grid and self.output_latent:

            latent_file = self.latent_dir / (grid_prefix + '_' + i + '.latent')
            self.write_latent(latent_file, grid.info['src_latent'])

        # store grid until ready to compute output metrics
        self.grids[lig_name][sample_idx][grid_type] = grid
        lig_grids = self.grids[lig_name]

        if self.batch_metrics: # store until grids for all samples are ready

            has_all_samples = (len(lig_grids) == self.n_samples)
            has_all_grids = all(
                len(lig_grids[i]) == self.n_grid_types for i in lig_grids
            )

            if has_all_samples and has_all_grids:

                # compute batch metrics
                if self.verbose:
                    print(
                        'Computing metrics for all ' + lig_name + ' samples'
                        )
                try:
                    self.compute_metrics(lig_name, range(self.n_samples))
                except:
                    for out in self.open_files.values():
                        out.close()
                    raise

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
                )
                del self.grids[lig_name]

        else: # only store until grids for this sample are ready
            has_all_grids = len(lig_grids[sample_idx]) == self.n_grid_types

            if has_all_grids:

                # compute sample metrics
                if self.verbose:
                    print('Computing metrics for {} sample {}'.format(
                        lig_name, sample_idx
                    ))

                try:
                    self.compute_metrics(lig_name, [sample_idx])
                except:
                    for open_files in self.out_files.values():
                        for out in open_files.values():
                            out.close()
                    raise

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
                )
                del self.grids[lig_name][sample_idx]

    def compute_metrics(self, lig_name, sample_idxs):
        '''
        Compute metrics for density grids, typed atomic structures,
        and molecules for a given ligand in metrics data frame.
        '''
        lig_grids = self.grids[lig_name]

        if self.batch_metrics: # compute mean grids

            lig_grid_mean = sum(
                lig_grids[i]['lig'].values for i in sample_idxs
            ) / self.n_samples

            lig_gen_grid_mean = sum(
                lig_grids[i]['lig_gen'].values for i in sample_idxs
            ) / self.n_samples

            lig_latent_mean = sum(
                lig_grids[i]['lig_gen'].info['src_latent'] for i in sample_idxs
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

            lig_latent = lig_gen_grid.info['src_latent']
            self.compute_latent_metrics(idx, 'lig', lig_latent, lig_latent_mean)

            if 'lig_gen_fit' in lig_grids[sample_idx]:

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

    def compute_grid_metrics(
        self, idx, grid_type, grid, ref_grid=None, mean_grid=None
    ):
        m = self.metrics

        # density magnitude
        m.loc[idx, grid_type+'_norm'] = grid.values.norm().item()

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
                (ref_grid.values - grid.values).abs()
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
        self, idx, struct_type, struct, ref_struct=None
    ):
        m = self.metrics

        m.loc[idx, struct_type+'_n_atoms'] = struct.n_atoms
        m.loc[idx, struct_type+'_radius'] = (
            struct.radius if struct.n_atoms > 0 else np.nan
        )

        if ref_struct is not None:

            # type count difference
            m.loc[idx, struct_type+'_type_diff'] = (
                ref_struct.type_counts - struct.type_counts
            ).norm(p=1).item()
            m.loc[idx, struct_type+'_exact_types'] = (
                m.loc[idx, struct_type+'_type_diff'] == 0
            )

            # minimum typed-atom RMSD
            rmsd = metrics.compute_atom_rmsd(ref_struct, struct)

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
        mols = molecules

        # standardize mols- make sure to copy over info
        mol_info = mol.info
        mol = mols.Molecule(Chem.RemoveHs(mol, sanitize=False))
        mol.info = mol_info

        # check molecular validity
        n_atoms, n_frags, error, valid = mols.get_rd_mol_validity(mol)
        m.loc[idx, mol_type+'_n_frags'] = n_frags
        m.loc[idx, mol_type+'_error'] = error
        m.loc[idx, mol_type+'_valid'] = valid

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
        smi = mols.get_smiles_string(mol)
        m.loc[idx, mol_type+'_SMILES'] = smi

        if ref_mol is not None: # compare to ref_mol

            # we have to sanitize the ref_mol here each time
            # since we copy before sanitizng on previous calls
            ref_mol_info = ref_mol.info
            ref_mol = mols.Molecule(Chem.RemoveHs(ref_mol, sanitize=False))
            ref_mol.info = ref_mol_info
            ref_valid = mols.get_rd_mol_validity(mol)[-1]

            # get reference SMILES strings
            ref_smi = mols.get_smiles_string(ref_mol)
            m.loc[idx, mol_type+'_SMILES_match'] = (smi == ref_smi)

            if valid and ref_valid: # fingerprint similarity
                m.loc[idx, mol_type+'_ob_sim'] = \
                    mols.get_ob_smi_similarity(ref_smi, smi)
                m.loc[idx, mol_type+'_morgan_sim'] = \
                    mols.get_rd_mol_similarity(ref_mol, mol, 'morgan')
                m.loc[idx, mol_type+'_rdkit_sim'] = \
                    mols.get_rd_mol_similarity(ref_mol, mol, 'rdkit')
                m.loc[idx, mol_type+'_maccs_sim'] = \
                    mols.get_rd_mol_similarity(ref_mol, mol, 'maccs')
            else:
                m.loc[idx, mol_type+'_ob_sim'] = np.nan
                m.loc[idx, mol_type+'_morgan_sim'] = np.nan
                m.loc[idx, mol_type+'_rdkit_sim'] = np.nan
                m.loc[idx, mol_type+'_maccs_sim'] = np.nan

        # UFF energy minimization
        min_mol = mol.info['min_mol']
        E_init = min_mol.info['E_init']
        E_min = min_mol.info['E_min']

        m.loc[idx, mol_type+'_E'] = E_init
        m.loc[idx, mol_type+'_min_E'] = E_min
        m.loc[idx, mol_type+'_dE_min'] = E_min - E_init
        m.loc[idx, mol_type+'_min_error'] = min_mol.info['min_error']
        m.loc[idx, mol_type+'_min_time'] = min_mol.info['min_time']
        m.loc[idx, mol_type+'_RMSD_min'] = mols.get_rd_mol_rmsd(min_mol, mol)

        # compare energy to ref mol, before and after minimizing
        if ref_mol is not None:

            min_ref_mol = ref_mol.info['min_mol']
            E_init_ref = min_ref_mol.info['E_init']
            E_min_ref = min_ref_mol.info['E_init']

            m.loc[idx, mol_type+'_dE_ref'] = E_init - E_init_ref
            m.loc[idx, mol_type+'_min_dE_ref'] = E_min - E_min_ref

            # get aligned RMSD to ref mol, pre-minimize
            m.loc[idx, mol_type+'_RMSD_ref'] = \
                mols.get_rd_mol_rmsd(ref_mol, mol)

            # get aligned RMSD to true mol, post-minimize
            m.loc[idx, mol_type+'_min_RMSD_ref'] = \
                mols.get_rd_mol_rmsd(min_ref_mol, min_mol)


def find_real_rec_in_data_root(data_root, rec_src_no_ext):

    # cross-docked set
    m = re.match(r'(.+)_0', rec_src_no_ext)
    rec_mol_base = m.group(1) + '.pdb'
    rec_mol_file = os.path.join(data_root, rec_mol_base)
    rec_mol = molecules.Molecule.from_pdb(rec_mol_file, sanitize=False)
    try:
        Chem.SanitizeMol(rec_mol)
    except Chem.MolSanitizeException:
        pass
    return rec_mol


def find_real_lig_in_data_root(data_root, lig_src_no_ext):
    '''
    Try to find the real molecule in data_root using the
    source path in the data file, without file extension.
    '''
    try: # PDBbind
        m = re.match(r'(.+)_ligand_(\d+)', lig_src_no_ext)
        lig_mol_base = m.group(1) + '_docked.sdf.gz'
        idx = int(m.group(2))
        lig_mol_file = os.path.join(data_root, lig_mol_base)
        lig_mol = molecules.Molecule.from_sdf(
            lig_mol_file, sanitize=False, idx=idx
        )
    except AttributeError:
        try: # cross-docked set
            m = re.match(r'(.+)_(\d+)', lig_src_no_ext)
            lig_mol_base = m.group(1) + '.sdf'
            idx = int(m.group(2))
            lig_mol_file = os.path.join(data_root, lig_mol_base)
            lig_mol = molecules.Molecule.from_sdf(
                lig_mol_file, sanitize=False, idx=idx
            )
        except OSError:
            lig_mol_base = lig_src_no_ext + '.sdf'
            lig_mol_file = os.path.join(data_root, lig_mol_base)
            lig_mol = molecules.read_rd_mols_from_sdf_file(
                lig_mol_file, sanitize=False, idx=0
            )
    try:
        Chem.SanitizeMol(lig_mol)
    except Chem.MolSanitizeException:
        pass
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
    pymol_file, out_prefix, dx_prefixes, sdf_files, centers=[]
):
    '''
    Write a pymol script that loads all .dx files with a given
    prefix into a single group, then loads a set of sdf_files
    and translates them to the origin, if centers are provided.
    '''
    with open(pymol_file, 'w') as f:

        for dx_prefix in dx_prefixes: # load densities
            dx_pattern = '{}_*.dx'.format(dx_prefix)
            m = re.match('^grids/.({}_.*)$'.format(out_prefix), dx_prefix)
            group_name = m.group(1) + '_grids'
            f.write('load_group {}, {}\n'.format(dx_pattern, group_name))

        for sdf_file in sdf_files: # load structures
            m = re.match(
                r'^structs/({}_.*)\.sdf(\.gz)?$'.format(out_prefix),
                str(sdf_file)
            )
            obj_name = m.group(1)
            f.write('load {}, {}\n'.format(sdf_file, obj_name))

        for sdf_file, (x,y,z) in zip(sdf_files, centers): # center structures
            m = re.match(
                r'^structs/({}_.*)\.sdf(\.gz)?$'.format(out_prefix),
                str(sdf_file)
            )
            obj_name = m.group(1)
            f.write('translate [{},{},{}], {}, camera=0, state=0\n'.format(
                -x, -y, -z, obj_name
            ))


def write_channels_to_file(channels_file, c):
    with open(channels_file, 'w') as f:
        f.write(' '.join(map(str, c)) + '\n')
        

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
    atom_fitter,
    out_writer,
    n_examples,
    n_samples,
    fit_atoms,
    prior=False,
    gen_only=False,
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
                    complex_grids = None if prior else data.grids

                    if verbose: print('Calling generator forward')
                    lig_gen_grids, latents, _, _ = gen_model(
                        complex_grids, rec_grids, batch_size
                    )
                    # TODO interpolation here!

            rec_struct = rec_structs[batch_idx]
            lig_struct = lig_structs[batch_idx]

            # undo transform so structs are all aligned
            transform = data.transforms[batch_idx]
            lig_center = lig_struct.center # store for atom fitting
            transform.backward(rec_struct.xyz, rec_struct.xyz)
            transform.backward(lig_struct.xyz, lig_struct.xyz)

            # only process real rec/lig once, since they're
            # the same for all samples of a given ligand

            if sample_idx == 0: # TODO re-implement gen_only

                if verbose: print('Getting real molecule from data root')

                rec_src_file = rec_struct.info['src_file']
                rec_src_no_ext = os.path.splitext(rec_src_file)[0]
                rec_mol = find_real_rec_in_data_root(
                    data.root_dir, rec_src_no_ext
                )

                lig_src_file = lig_struct.info['src_file']
                lig_src_no_ext = os.path.splitext(lig_src_file)[0]
                lig_name = os.path.basename(lig_src_no_ext)
                lig_mol = find_real_lig_in_data_root(
                    data.root_dir, lig_src_no_ext
                )

                if fit_atoms: # add bonds and minimize

                    if verbose: print('Minimizing real molecule')
                    lig_mol.info['min_mol'] = lig_mol.uff_minimize()

                    if verbose: print('Making molecule from real atoms')
                    lig_add_mol = lig_struct.make_mol(verbose)
                    lig_add_mol.info['min_mol'] = lig_add_mol.uff_minimize()

            rec_struct.info['src_mol'] = rec_mol
            lig_struct.info['src_mol'] = lig_mol
            lig_struct.info['add_mol'] = lig_add_mol

            grid_types = [
                ('rec', rec_grids, data.rec_channels),
                ('lig', lig_grids, data.lig_channels),
                ('lig_gen', lig_gen_grids, data.lig_channels)
            ]
            for grid_type, grids, grid_channels in grid_types:
                torch.cuda.reset_max_memory_allocated()

                is_lig_grid = (grid_type.startswith('lig'))
                grid_needs_fit = (is_lig_grid and fit_atoms)

                grid = liGAN.atom_grids.AtomGrid(
                    values=grids[batch_idx],
                    channels=grid_channels,
                    center=lig_center, # use original (transformed) center
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
                        atom_fitter.convolve(
                            grid.values, grid.channels, grid.resolution,
                        )
                    )

                out_writer.write(lig_name, grid_type, sample_idx, grid)

                if grid_needs_fit: # atom fitting, bond adding, minimize

                    fit_struct, fit_grid = atom_fitter.fit(
                        grid, lig_struct.type_counts
                    )

                    if fit_struct.n_atoms > 0: # undo transform
                        transform.backward(fit_struct.xyz, fit_struct.xyz)

                    fit_add_mol = fit_struct.make_mol(verbose)
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

    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_colwidth', 100)
    try:
        display_width = get_terminal_size()[1]
    except:
        display_width = 185
    pd.set_option('display.width', display_width)

    device = 'cuda'
    if 'random_seed' in config:
        liGAN.set_random_seed(config['random_seed'])
    else:
        liGAN.set_random_seed()

    print('Loading data')
    data_file = config['data'].pop('data_file')
    data = liGAN.data.AtomGridData(
        device=device,
        n_samples=config['generate']['n_samples'],
        **config['data']
    )
    data.populate(data_file)

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
    gen_model.load_state_dict(torch.load(gen_model_state))

    print('Initializing atom fitter')
    atom_fitter = liGAN.atom_fitting.AtomFitter(
        device=device, **config['atom_fitting']
    )

    print('Initializing output writer')
    out_writer = OutputWriter(
        out_prefix=config['out_prefix'],
        n_samples=config['generate']['n_samples'],
        n_grid_types=5 if config['generate']['fit_atoms'] else 3,
        verbose=config['verbose'],
        **config['output']
    )

    generate(
        data=data,
        gen_model=gen_model,
        atom_fitter=atom_fitter,
        out_writer=out_writer,
        verbose=config['verbose'],
        **config['generate']
    )
    print('Done')


if __name__ == '__main__':
    main(sys.argv[1:])

