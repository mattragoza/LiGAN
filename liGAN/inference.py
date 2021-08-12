import sys, os, re, time, gzip, itertools
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
from liGAN import models
from liGAN.atom_grids import AtomGrid
from liGAN.atom_structs import AtomStruct
from liGAN import molecules as mols

MB = 1024 ** 2


class MoleculeGenerator(object):
    '''
    Base class for generating 3D molecules
    using a generative model, atom fitting,
    and bond adding algorithms.
    '''
    # subclasses override these class attributes
    gen_model_type = None
    has_disc_model = False # not used...
    has_prior_model = False
    has_complex_input = False

    def __init__(
        self,
        out_prefix,
        n_samples,
        fit_atoms,
        data_kws={},
        gen_model_kws={},
        prior_model_kws={},
        atom_fitting_kws={},
        bond_adding_kws={},
        output_kws={},
        device='cuda',
        verbose=False,
        debug=False,
    ):
        super().__init__()
        self.device = device

        print('Loading data')
        self.init_data(device=device, n_samples=n_samples, **data_kws)

        if self.gen_model_type:
            print('Initializing generative model')
            self.init_gen_model(
                device=device, n_samples=n_samples, **gen_model_kws
            )

            if self.gen_model_type.has_stage2:
                print('Initializing prior model')
                self.init_prior_model(device=device, **prior_model_kws)
            else:
                self.prior_model = None
        else:
            print('No generative model')
            self.gen_model = None
            self.prior_model = None

        print('Initializing atom fitter')
        self.atom_fitter = liGAN.atom_fitting.AtomFitter(
            device=device, **atom_fitting_kws
        )
        print('Initializing bond adder')
        self.bond_adder = liGAN.bond_adding.BondAdder(
            debug=debug, **bond_adding_kws
        )

        # determine expected grid types
        grid_types = {'rec', 'lig'}
        if self.data.diff_cond_structs or self.data.diff_cond_transform:
            grid_types.add('cond_rec')
            grid_types.add('cond_lig')
        if self.gen_model:
            grid_types.add('lig_gen')
            if fit_atoms:
                grid_types.add('lig_gen_fit')
        elif fit_atoms:
            grid_types.add('lig_fit')

        print('Initializing output writer')
        self.out_writer = OutputWriter(
            out_prefix=out_prefix,
            n_samples=n_samples,
            grid_types=grid_types,
            verbose=verbose,
            **output_kws,
        )

    def init_data(self, device, n_samples, **data_kws):
        self.data = liGAN.data.AtomGridData(
            device=device, n_samples=n_samples, **data_kws
        )

    def init_gen_model(
        self,
        device,
        caffe_init=False,
        state=None,
        **gen_model_kws
    ):
        self.gen_model = self.gen_model_type(
            n_channels_in=self.n_channels_in,
            n_channels_cond=self.n_channels_cond,
            n_channels_out=self.n_channels_out,
            grid_size=self.data.grid_size,
            device=device,
            **gen_model_kws
        )
        if caffe_init:
            self.gen_model.apply(liGAN.models.caffe_init_weights)

        if state:
            print('Loading generative model state')
            state_dict = torch.load(state)
            state_dict.pop('log_recon_var', None)
            self.gen_model.load_state_dict(state_dict)

    def init_prior_model(
        self,
        device,
        caffe_init=False,
        state=None,
        **prior_model_kws
    ):
        self.prior_model = liGAN.models.Stage2VAE(
            n_input=self.gen_model.n_latent,
            **prior_model_kws
        ).to(device)

        if caffe_init:
            self.prior_model.apply(liGAN.models.caffe_init_weights)

        if state:
            print('Loading prior model state')
            state_dict = torch.load(state)
            state_dict.pop('log_recon_var', None)
            self.prior_model.load_state_dict(state_dict)

    @property
    def n_channels_in(self):
        if self.gen_model_type.has_input_encoder:
            data = self.data
            if self.has_complex_input:
                return data.n_rec_channels + data.n_lig_channels
            else:
                return data.n_lig_channels

    @property
    def n_channels_cond(self):
        if self.gen_model_type.has_conditional_encoder:
            return self.data.n_rec_channels

    @property
    def n_channels_out(self):
        return self.data.n_lig_channels

    @property
    def n_channels_disc(self):
        if self.has_disc_model:
            data = self.data
            if self.gen_model_type.has_conditional_encoder:
                return data.n_rec_channels + data.n_lig_channels
            else:
                return data.n_lig_channels

    def forward(
        self,
        prior,
        stage2,
        interpolate=False,
        spherical=False,
        **kwargs
    ):
        print(f'Calling generator forward')
        print(f'  prior = {prior}')
        print(f'  stage2 = {stage2}')

        print('Getting next batch of data')
        data = self.data
        input_grids, cond_grids, input_structs, cond_structs, transforms \
            = data.forward(interpolate=interpolate, spherical=spherical)[:5]
        input_rec_structs, input_lig_structs = input_structs
        cond_rec_structs, cond_lig_structs = cond_structs
        input_transforms, cond_transforms = transforms
        input_rec_grids, input_lig_grids = data.split_channels(input_grids)
        cond_rec_grids, cond_lig_grids = data.split_channels(cond_grids)

        posterior = not prior
        if posterior:
            if self.has_complex_input:
                gen_input_grids = torch.cat(
                    [input_rec_grids, input_lig_grids], dim=1
                )
            else:
                gen_input_grids = input_lig_grids
        else:
            gen_input_grids = None

        gen_cond_grids = cond_rec_grids

        if self.gen_model:

            with torch.no_grad():
                if stage2: # insert prior model
                    lig_gen_grids, _, _, _, latents, _, _ = \
                        self.gen_model.forward2(
                            prior_model=self.prior_model,
                            inputs=gen_input_grids,
                            conditions=gen_cond_grids,
                            interpolate=interpolate,
                            spherical=spherical,
                            batch_size=self.data.batch_size,
                            **kwargs
                        )
                else:
                    lig_gen_grids, latents, _, _ = self.gen_model(
                        inputs=gen_input_grids,
                        conditions=gen_cond_grids,
                        batch_size=self.data.batch_size,
                        interpolate=interpolate,
                        spherical=spherical,
                        **kwargs
                    )
        else:
            lig_gen_grids, latents = None, None

        def try_detach(x):
            try:
                return x.detach()
            except AttributeError:
                return x

        input_grids = try_detach(input_rec_grids), try_detach(input_lig_grids)
        cond_grids = try_detach(cond_rec_grids), try_detach(cond_lig_grids)
        latents, lig_gen_grids = try_detach(latents), try_detach(lig_gen_grids)
        return (
            input_grids, cond_grids,
            input_structs, cond_structs,
            latents, lig_gen_grids, transforms
        )

    def generate(
        self,
        n_examples,
        n_samples,
        prior=False,
        stage2=False,
        var_factor=1.0,
        post_factor=1.0,
        z_score=None,
        truncate=None,
        interpolate=False,
        spherical=False,
        fit_atoms=True,
        add_bonds=True,
        uff_minimize=True,
        gnina_minimize=True,
        fit_to_real=False,
        add_to_real=False,
        minimize_real=True,
        verbose=True,
    ):
        '''
        Generate atomic density grids from generative
        model for each example in data, fit atomic
        structures, and add bonds to make molecules.
        '''
        batch_size = self.data.batch_size

        print('Starting to generate grids')
        for example_idx, sample_idx in itertools.product(
            range(n_examples), range(n_samples)
        ):
            # keep track of position in current batch
            full_idx = example_idx*n_samples + sample_idx
            batch_idx = full_idx % batch_size
            print(example_idx, sample_idx, full_idx, batch_idx)

            need_real_input_mol = (sample_idx == 0)
            need_real_cond_mol = \
                (sample_idx == 0 and self.data.diff_cond_structs)
            need_next_batch = (batch_idx == 0)

            if need_next_batch: # forward next batch

                #if gnina_minimize: # copy to gpu
                #    self.gen_model.to('cuda')
                (
                    input_grids, cond_grids,
                    input_structs, cond_structs,
                    latents, lig_gen_grids, transforms
                ) = self.forward(
                    prior=prior,
                    stage2=stage2,
                    var_factor=var_factor,
                    post_factor=post_factor,
                    z_score=z_score,
                    truncate=truncate,
                    interpolate=interpolate,
                    spherical=spherical,
                )
                input_rec_grids, input_lig_grids = input_grids
                cond_rec_grids, cond_lig_grids = cond_grids
                input_rec_structs, input_lig_structs = input_structs
                cond_rec_structs, cond_lig_structs = cond_structs
                input_transforms, cond_transforms = transforms
                #if gnina_minimize: # copy to cpu
                #    self.gen_model.to('cpu')

            input_rec_struct = input_rec_structs[batch_idx]
            input_lig_struct = input_lig_structs[batch_idx]
            cond_rec_struct = cond_rec_structs[batch_idx]
            cond_lig_struct = cond_lig_structs[batch_idx]

            # in order to align gen structs with real structs,
            #   we need to apply the inverse of the transform
            #   that was used to create the density grid that
            #   is the reconstruction target (assume conditional)
            input_transform = input_transforms[batch_idx]
            cond_transform = cond_transforms[batch_idx]
            input_center = torch.as_tensor(tuple(
                input_transform.get_rotation_center()
            ))
            cond_center = torch.as_tensor(tuple(
                cond_transform.get_rotation_center()
            ))

            # only process real rec/lig once, since they're
            #   the same for all samples of a given ligand
            if need_real_input_mol:
                print('Getting real input molecule from data root')
                splitext = lambda x: x.rsplit('.', 1 + x.endswith('.gz'))

                input_rec_src_file = input_rec_struct.info['src_file']
                input_rec_mol = read_rec_from_pdb_file(input_rec_src_file)
                input_rec_name = \
                    splitext(os.path.basename(input_rec_src_file))[0]

                input_lig_src_file = input_lig_struct.info['src_file']
                input_lig_mol = read_lig_from_sdf_file(input_lig_src_file)
                input_lig_name = \
                    splitext(os.path.basename(input_lig_src_file))[0]

                if uff_minimize:
                    # real molecules don't need UFF minimization,
                    #   but we need their UFF metrics for reference
                    input_pkt_mol = \
                        input_rec_mol.get_pocket(lig_mol=input_lig_mol)
                    input_lig_mol.info['pkt_mol'] = input_pkt_mol
                    input_uff_mol = \
                        input_lig_mol.uff_minimize(rec_mol=input_pkt_mol)
                    input_lig_mol.info['uff_mol'] = input_uff_mol

                    if minimize_real:
                        print('Minimizing real molecule with gnina', flush=True)
                        # NOTE that we are not using the UFF mol here
                        input_lig_mol.info['gni_mol'] = \
                            input_lig_mol.gnina_minimize(rec_mol=input_rec_mol)

                if add_to_real: # evaluate bond adding in isolation
                    print('Adding bonds to real atoms')
                    lig_add_mol, lig_add_struct, _ = \
                        self.bond_adder.make_mol(input_lig_struct)
                    lig_add_mol.info['type_struct'] = lig_add_struct
                    lig_struct.info['add_mol'] = lig_add_mol

                    if uff_minimize:
                        print('Minimizing molecule from real atoms with UFF',
                            end='') # show number of tries inline
                        lig_add_pkt_mol = \
                            input_rec_mol.get_pocket(lig_mol=lig_add_mol)
                        lig_add_mol.info['pkt_mol'] = lig_add_pkt_mol
                        lig_add_uff_mol = \
                            lig_add_mol.uff_minimize(rec_mol=lig_add_pkt_mol)
                        lig_add_mol.info['uff_mol'] = lig_add_uff_mol

                        if gnina_minimize:
                            print('Minimizing molecule from real atoms with gnina', flush=True)
                            lig_add_mol.info['gni_mol'] = \
                                lig_add_uff_mol.gnina_minimize(rec_mol=input_rec_mol)

            else: # check that the molecules are the same
                assert input_rec_struct.info['src_file'] == input_rec_src_file
                assert input_lig_struct.info['src_file'] == input_lig_src_file

            # if the conditional molecule is different,
            #   we need to process it separately from input
            if need_real_cond_mol:

                cond_rec_src_file = cond_rec_struct.info['src_file']
                cond_rec_mol = read_rec_from_pdb_file(cond_rec_src_file)
                cond_rec_name = splitext(os.path.basename(cond_rec_src_file))[0]

                cond_lig_src_file = cond_lig_struct.info['src_file']
                cond_lig_mol = read_lig_from_sdf_file(cond_lig_src_file)
                cond_lig_name = splitext(os.path.basename(cond_lig_src_file))[0]

                if uff_minimize:
                    # real molecules don't need UFF minimization,
                    #   but we need their UFF metrics for reference
                    cond_pkt_mol = cond_rec_mol.get_pocket(lig_mol=cond_lig_mol)
                    cond_lig_mol.info['pkt_mol'] = cond_pkt_mol
                    cond_uff_mol = \
                        cond_lig_mol.uff_minimize(rec_mol=cond_pkt_mol)
                    cond_lig_mol.info['uff_mol'] = cond_uff_mol

                    if minimize_real:
                        print('Minimizing real molecule with gnina', flush=True)
                        # NOTE that we are not using the UFF mol here
                        cond_lig_mol.info['gni_mol'] = \
                            cond_lig_mol.gnina_minimize(rec_mol=cond_rec_mol)

            elif self.data.diff_cond_structs:
                assert cond_rec_struct.info['src_file'] == cond_rec_src_file
                assert cond_lig_struct.info['src_file'] == cond_lig_src_file
            else:
                cond_rec_name = input_rec_name
                cond_lig_name = input_lig_name

            # unique identifier for this data example
            example_info = (
                example_idx,
                input_rec_name,
                input_lig_name,
                cond_rec_name,
                cond_lig_name,
            )

            # done processing real mols/structs, so attach them
            input_rec_struct.info['src_mol'] = input_rec_mol
            input_lig_struct.info['src_mol'] = input_lig_mol
            if self.data.diff_cond_structs:
                cond_rec_struct.info['src_mol'] = cond_rec_mol
                cond_lig_struct.info['src_mol'] = cond_lig_mol

            # now process atomic density grids
            grid_types = [
                ('rec', input_rec_grids),
                ('lig', input_lig_grids),  
            ]
            if self.data.diff_cond_structs or self.data.diff_cond_transform:
                grid_types += [
                    ('cond_rec', cond_rec_grids),
                    ('cond_lig', cond_lig_grids)
                ]
            if self.gen_model:
                grid_types += [
                    ('lig_gen', lig_gen_grids)
                ]

            for grid_type, grids in grid_types:
                torch.cuda.reset_max_memory_allocated()

                is_cond_grid = grid_type.startswith('cond')
                is_lig_grid = ('lig' in grid_type)
                is_gen_grid = grid_type.endswith('gen')
                real_or_gen = 'generated' if is_gen_grid else 'real'

                if is_gen_grid:
                    grid_needs_fit = fit_atoms and is_lig_grid
                    center = cond_center
                elif is_cond_grid:
                    grid_need_fit = False
                    center = cond_center
                else:
                    grid_needs_fit = fit_to_real and is_lig_grid
                    center = input_center

                if is_lig_grid:
                    atom_typer = self.data.lig_typer
                else:
                    atom_typer = self.data.rec_typer

                grid = liGAN.atom_grids.AtomGrid(
                    values=grids[batch_idx],
                    typer=atom_typer,
                    center=center,
                    resolution=self.data.resolution
                )

                if grid_type == 'rec':
                    grid.info['src_struct'] = input_rec_struct
                elif grid_type == 'lig':
                    grid.info['src_struct'] = input_lig_struct
                elif grid_type == 'cond_rec':
                    grid.info['src_struct'] = cond_rec_struct
                elif grid_type == 'cond_lig':
                    grid.info['src_struct'] = cond_lig_struct
                elif grid_type == 'lig_gen':
                    grid.info['src_latent'] = latents[batch_idx]

                # display progress
                index_str = f'[example_idx={example_idx} sample_idx={sample_idx} grid_type={grid_type}]'
                value_str = 'norm={:.4f} gpu={:.4f}'.format(
                    grid.values.norm(),
                    torch.cuda.max_memory_allocated() / MB,
                )
                print(index_str + ' ' + value_str, flush=True)

                if is_lig_grid and self.out_writer.output_conv:
                    grid.info['conv_grid'] = grid.new_like(
                        values=torch.cat([
                            atom_fitter.convolve(
                                grid.elem_values, grid.resolution, grid.typer
                            ),
                            grid.prop_values
                        ], dim=0)
                    )

                self.out_writer.write(example_info, sample_idx, grid_type, grid)

                if grid_needs_fit: # perform atom fitting

                    print(f'Fitting atoms to {real_or_gen} grid')
                    fit_struct, fit_grid, visited_structs = self.atom_fitter.fit_struct(grid, cond_lig_struct.type_counts)
                    fit_struct.info['visited_structs'] = visited_structs
                    fit_grid.info['src_struct'] = fit_struct

                    if fit_struct.n_atoms > 0: # inverse transform
                        cond_transform.backward(fit_struct.coords, fit_struct.coords)

                    if add_bonds: # do bond adding
                        print(f'Adding bonds to atoms from {real_or_gen} grid')
                        fit_add_mol, fit_add_struct, visited_mols = \
                            self.bond_adder.make_mol(fit_struct)
                        fit_add_mol.info['type_struct'] = fit_add_struct
                        fit_struct.info['add_mol'] = fit_add_mol

                        if uff_minimize: # do UFF minimization
                            print(f'Minimizing molecule from {real_or_gen} grid with UFF', end='')
                            fit_pkt_mol = input_rec_mol.get_pocket(fit_add_mol)
                            fit_add_mol.info['pkt_mol'] = fit_pkt_mol
                            fit_uff_mol = \
                                fit_add_mol.uff_minimize(rec_mol=fit_pkt_mol)
                            fit_add_mol.info['uff_mol'] = fit_uff_mol

                            if gnina_minimize: # do gnina minimization
                                print(f'Minimizing molecule from {real_or_gen} grid with gnina', flush=True)
                                fit_add_mol.info['gni_mol'] = fit_uff_mol.gnina_minimize(rec_mol=input_rec_mol)

                        # minimize and score wrt conditional receptor too
                        if self.data.diff_cond_structs and uff_minimize:
                            print(f'Minimizing molecule from {real_or_gen} grid with UFF wrt conditional receptor', end='')
                            fit_pkt_mol = cond_rec_mol.get_pocket(fit_add_mol)
                            fit_add_mol.info['cond_pkt_mol'] = fit_pkt_mol
                            fit_uff_mol = \
                                fit_add_mol.uff_minimize(rec_mol=fit_pkt_mol)
                            fit_add_mol.info['cond_uff_mol'] = fit_uff_mol

                            if gnina_minimize: # do gnina minimization
                                print(f'Minimizing molecule from {real_or_gen} grid with gnina wrt conditional receptor', flush=True)
                                fit_add_mol.info['cond_gni_mol'] = \
                                    fit_uff_mol.gnina_minimize(rec_mol=cond_rec_mol)

                    grid_type += '_fit'
                    self.out_writer.write(
                        example_info, sample_idx, grid_type, fit_grid
                    )

        return self.out_writer.metrics


class AEGenerator(MoleculeGenerator):
    gen_model_type = liGAN.models.AE


class VAEGenerator(MoleculeGenerator):
    gen_model_type = liGAN.models.VAE


class CEGenerator(MoleculeGenerator):
    gen_model_type = liGAN.models.CE


class CVAEGenerator(MoleculeGenerator):
    gen_model_type = liGAN.models.CVAE
    has_complex_input = True


class GANGenerator(MoleculeGenerator):
    gen_model_type = liGAN.models.GAN
    has_disc_model = True


class CGANGenerator(MoleculeGenerator):
    gen_model_type = liGAN.models.CGAN
    has_disc_model = True


class VAEGANGenerator(MoleculeGenerator):
    gen_model_type = liGAN.models.VAE
    has_disc_model = True


class CVAEGANGenerator(MoleculeGenerator):
    gen_model_type = liGAN.models.CVAE
    has_complex_input = True
    has_disc_model = True


class VAE2Generator(MoleculeGenerator):
    gen_model_type = liGAN.models.VAE2
    has_prior_model = True


class CVAE2Generator(MoleculeGenerator):
    gen_model_type = liGAN.models.CVAE2
    has_complex_input = True
    has_prior_model = True



class OutputWriter(object):
    '''
    A data structure for receiving and sorting AtomGrids and
    AtomStructs from a generative model or atom fitting algorithm,
    computing metrics, and writing files to disk as necessary.
    '''
    def __init__(
        self,
        out_prefix,
        n_samples,
        grid_types,
        output_mols=True,
        output_structs=False,
        output_grids=False,
        output_latents=False,
        output_visited=False,
        output_conv=False,
        batch_metrics=False,
        verbose=False
    ):
        out_dir, out_prefix = os.path.split(out_prefix)
        self.out_prefix = out_prefix

        self.output_grids = output_grids
        self.output_structs = output_structs
        self.output_mols = output_mols
        self.output_latents = output_latents
        self.output_visited = output_visited
        self.output_conv = output_conv
        self.n_samples = n_samples
        self.grid_types = grid_types
        self.batch_metrics = batch_metrics

        # organize grids by (rec_name, lig_name), sample_idx, grid_type
        self.grids = defaultdict(lambda: defaultdict(dict))

        # accumulate metrics in dataframe
        self.metric_file = os.path.join(out_dir, f'{out_prefix}.gen_metrics')
        columns = [
            'example_idx',
            'input_rec_name',
            'input_lig_name',
            'cond_rec_name',
            'cond_lig_name',
            'sample_idx'
        ]
        self.metrics = pd.DataFrame(columns=columns).set_index(columns)

        # write a pymol script when finished
        self.pymol_file = os.path.join(out_dir, f'{out_prefix}.pymol')
        self.dx_prefixes = []
        self.sdf_files = []

        self.verbose = verbose

        # keep sdf files open so that all samples of a given
        #   struct or molecule can be written to one file
        self.open_files = dict()

        # create directories for output files
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)

        if output_latents:
            self.latent_dir = out_dir / 'latents'
            self.latent_dir.mkdir(exist_ok=True)

        if output_grids:
            self.grid_dir = out_dir / 'grids'
            self.grid_dir.mkdir(exist_ok=True)

        if output_structs:
            self.struct_dir = out_dir / 'structs'
            self.struct_dir.mkdir(exist_ok=True)

        if output_mols:
            self.mol_dir = out_dir / 'molecules'
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
            self.print(f'Writing {sdf_file} sample {sample_idx}')
                
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

        self.print(f'Writing {dx_prefix} .dx files')
        grid.to_dx(dx_prefix)
        self.dx_prefixes.append(dx_prefix)

    def write_latent(self, latent_file, latent_vec):

        self.print('Writing ' + str(latent_file))
        write_latent_vec_to_file(latent_file, latent_vec)

    def write(self, example_info, sample_idx, grid_type, grid):
        '''
        Write output files for grid and compute metrics in
        data frame, if all necessary data is present.
        '''
        out_prefix = self.out_prefix
        example_idx, input_rec_name, input_lig_name, cond_rec_name, cond_lig_name = example_info
        grid_prefix = f'{out_prefix}_{example_idx}_{grid_type}'
        i = str(sample_idx)

        assert grid_type in {
            'rec', 'lig', 'cond_rec', 'cond_lig',
            'lig_gen', 'lig_fit', 'lig_gen_fit'
        }
        is_lig_grid = grid_type.startswith('lig')
        is_gen_grid = grid_type.endswith('_gen')
        is_fit_grid = grid_type.endswith('_fit')

        is_real_grid = not (is_gen_grid or is_fit_grid)
        is_first_real_grid = (is_real_grid and sample_idx == 0)
        has_struct = (is_real_grid or is_fit_grid)
        has_conv_grid = not is_fit_grid # and is_lig_grid ?

        # write atomic structs and/or molecules
        if has_struct:

            # get struct that created this grid (via molgrid.GridMaker)
            #   note that depending on the grid_type, this can either be
            #   from atom fitting OR from typing a real molecule
            struct = grid.info['src_struct']

            # the real (source) molecule and atom types don't change
            #   between different samples, so only write them once

            # and we don't apply bond adding to the receptor struct,
            #   so only ligand structs have add_mol and uff_mol

            # write real molecule
            if self.output_mols and is_first_real_grid:

                sdf_file = self.mol_dir / (grid_prefix + '_src.sdf.gz')
                src_mol = struct.info['src_mol']
                self.write_sdf(sdf_file, src_mol, sample_idx, is_real=True)

                if 'pkt_mol' in src_mol.info:
                    sdf_file = self.mol_dir / (grid_prefix+'_src_pkt.sdf.gz')
                    pkt_mol = src_mol.info['pkt_mol']
                    self.write_sdf(sdf_file, pkt_mol, sample_idx, is_real=True)

                if 'uff_mol' in src_mol.info:
                    sdf_file = self.mol_dir / (grid_prefix+'_src_uff.sdf.gz')
                    uff_mol = src_mol.info['uff_mol']
                    self.write_sdf(sdf_file, uff_mol, sample_idx, is_real=True)

                if 'gni_mol' in src_mol.info:
                    sdf_file = self.mol_dir / (grid_prefix+'_src_gna.sdf.gz')
                    gni_mol = src_mol.info['gni_mol']
                    self.write_sdf(sdf_file, gni_mol, sample_idx, is_real=True)

            # write typed atomic structure (real or fit)
            if self.output_structs and (is_first_real_grid or is_fit_grid):

                sdf_file = self.struct_dir / (grid_prefix + '.sdf.gz')
                self.write_sdf(sdf_file, struct, sample_idx, is_real_grid)

                # write atom type channels
                types_base = grid_prefix + '_' + i + '.atom_types'
                types_file = self.struct_dir / types_base
                self.write_atom_types(types_file, struct.atom_types)

            # write bond-added molecule (real or fit ligand)
            if self.output_mols and 'add_mol' in struct.info:
                sdf_file = self.mol_dir / (grid_prefix + '_add.sdf.gz')
                add_mol = struct.info['add_mol']
                self.write_sdf(sdf_file, add_mol, sample_idx, is_real_grid)

                if 'pkt_mol' in add_mol.info:
                    sdf_file = self.mol_dir / (grid_prefix+'_add_pkt.sdf.gz')
                    pkt_mol = add_mol.info['pkt_mol']
                    self.write_sdf(sdf_file, pkt_mol, sample_idx, is_real_grid)

                if 'uff_mol' in add_mol.info:
                    sdf_file = self.mol_dir / (grid_prefix + '_add_uff.sdf.gz')
                    uff_mol = add_mol.info['uff_mol']
                    self.write_sdf(sdf_file, uff_mol, sample_idx, is_real_grid)

                if 'gni_mol' in add_mol.info:
                    sdf_file = self.mol_dir / (grid_prefix+'_add_gna.sdf.gz')
                    gni_mol = add_mol.info['gni_mol']
                    self.write_sdf(sdf_file, gni_mol, sample_idx, is_real_grid)

                if 'cond_pkt_mol' in add_mol.info:
                    sdf_file = self.mol_dir / (grid_prefix+'_add_cond_pkt.sdf.gz')
                    pkt_mol = add_mol.info['pkt_mol']
                    self.write_sdf(sdf_file, pkt_mol, sample_idx, is_real_grid)

                if 'cond_uff_mol' in add_mol.info:
                    sdf_file = self.mol_dir / (grid_prefix + '_add_cond_uff.sdf.gz')
                    uff_mol = add_mol.info['uff_mol']
                    self.write_sdf(sdf_file, uff_mol, sample_idx, is_real_grid)

                if 'cond_gni_mol' in add_mol.info:
                    sdf_file = self.mol_dir / (grid_prefix+'_add_cond_gna.sdf.gz')
                    gni_mol = add_mol.info['gni_mol']
                    self.write_sdf(sdf_file, gni_mol, sample_idx, is_real_grid)

        # write atomic density grids
        if self.output_grids:

            dx_prefix = self.grid_dir / (grid_prefix + '_' + i)
            self.write_dx(dx_prefix, grid)

            # write convolved grid
            if self.output_conv and 'conv_grid' in grid.info:

                dx_prefix = self.grid_dir / (grid_prefix + '_conv_' + i)
                self.write_dx(dx_prefix, grid.info['conv_grid'])

        # write latent vectors
        if self.output_latents and is_gen_grid:

            latent_file = self.latent_dir / (grid_prefix + '_' + i + '.latent')
            self.write_latent(latent_file, grid.info['src_latent'])

        # store grid until ready to compute output metrics
        #   if we're computing batch matrics, need all samples
        #   otherwise, just need all grids for this sample
        self.grids[example_info][sample_idx][grid_type] = grid
        lig_grids = self.grids[example_info]

        if self.batch_metrics:

            # store until grids for all samples are ready
            has_all_samples = (len(lig_grids) == self.n_samples)
            has_all_grids = all(
                set(lig_grids[i]) >= self.grid_types for i in lig_grids
            )

            # compute batch metrics
            if has_all_samples and has_all_grids:
                self.print(
                    f'Computing metrics for all example {example_idx} samples'
                )
                try:
                    self.compute_metrics(example_info)
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
                self.print('Freeing memory')
                del self.grids[example_info] # free memory

        else:
            # only store until grids for this sample are ready
            has_all_grids = (
                set(lig_grids[sample_idx]) >= self.grid_types
            )
            # compute sample metrics
            if has_all_grids:
                self.print(
                    f'Computing metrics for example {example_idx} sample {sample_idx}'
                )
                try:
                    self.compute_metrics(example_info, sample_idx)
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
                self.print('Freeing memory')
                del self.grids[example_info][sample_idx] # free memory

    def compute_metrics(self, example_info, sample_idx=None):
        '''
        Compute metrics for density grids, typed atomic structures,
        and molecules for a given ligand in metrics data frame.
        '''
        has_rec = ('rec' in self.grid_types)
        has_cond_rec = ('cond_rec' in self.grid_types)
        has_lig_gen = ('lig_gen' in self.grid_types)
        has_lig_fit = ('lig_fit' in self.grid_types)
        has_lig_gen_fit = ('lig_gen_fit' in self.grid_types)

        if sample_idx is None:
            sample_idxs = range(self.n_samples)
        else:
            sample_idxs = [sample_idx]

        # TODO don't compute metrics twice w/ diff_cond_transform
        #   the only thing we really need is the lig l2 loss
        
        lig_grids = self.grids[example_info]

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

            if has_cond_rec:
                cond_lig_grid_batch = \
                    [lig_grids[i]['cond_lig'].values for i in sample_idxs]
                cond_lig_grid_mean = sum(cond_lig_grid_batch) / self.n_samples

                cond_lig_struct_batch = [
                    lig_grids[i]['cond_lig'].info['src_struct'] \
                        for i in sample_idxs
                ]
                cond_lig_mean_counts = \
                    get_mean_type_counts(cond_lig_struct_batch)

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
            cond_lig_grid_mean = None
            lig_mean_counts = None
            cond_lig_mean_counts = None
            lig_fit_mean_counts = None
            lig_gen_grid_mean = None
            lig_latent_mean = None
            lig_gen_fit_mean_counts = None

        for sample_idx in sample_idxs:
            idx = example_info + (sample_idx,)

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

            if 'add_mol' in lig_struct.info:

                lig_add_mol = lig_struct.info['add_mol']
                self.compute_mol_metrics(idx,
                    mol_type='lig_add', mol=lig_add_mol, ref_mol=lig_mol
                )

            if has_cond_rec:
                cond_rec_grid = lig_grids[sample_idx]['cond_rec']
                cond_lig_grid = lig_grids[sample_idx]['cond_lig']
                self.compute_grid_metrics(idx,
                    grid_type='cond_lig',
                    grid=cond_lig_grid,
                    mean_grid=cond_lig_grid_mean,
                    cond_grid=cond_rec_grid,
                )

                cond_lig_struct = cond_lig_grid.info['src_struct']
                self.compute_struct_metrics(idx,
                    struct_type='cond_lig',
                    struct=cond_lig_struct,
                    mean_counts=cond_lig_mean_counts,
                )

                cond_lig_mol = cond_lig_struct.info['src_mol']
                self.compute_mol_metrics(idx,
                    mol_type='lig', mol=cond_lig_mol
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

                if has_cond_rec:

                    self.compute_grid_metrics(idx,
                        grid_type='lig_gen_cond',
                        grid=lig_gen_grid,
                        ref_grid=cond_lig_grid,
                        cond_grid=cond_rec_grid,
                        ref_only=True
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
                    ref_struct=lig_fit_struct,
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
                    ref_struct=lig_gen_fit_struct,
                )

                if has_cond_rec:

                    self.compute_struct_metrics(idx,
                        struct_type='lig_gen_fit_cond',
                        struct=lig_gen_fit_struct,
                        ref_struct=cond_lig_struct,
                        ref_only=True,
                    )

                    self.compute_mol_metrics(idx,
                        mol_type='lig_gen_fit_add_cond',
                        mol=lig_gen_fit_add_mol,
                        ref_mol=cond_lig_mol,
                        ref_only=True,
                        use_cond_min=True
                    )

        self.print(self.metrics.loc[example_info].loc[sample_idxs].transpose())

    def compute_grid_metrics(
        self,
        idx,
        grid_type,
        grid,
        ref_grid=None,
        mean_grid=None,
        cond_grid=None,
        ref_only=False,
    ):
        m = self.metrics

        if not ref_only:

            # density magnitude
            m.loc[idx, grid_type+'_grid_norm'] = grid.values.norm().item()
            m.loc[idx, grid_type+'_grid_elem_norm'] = \
                grid.elem_values.norm().item()
            m.loc[idx, grid_type+'_grid_prop_norm'] = \
                grid.prop_values.norm().item()

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
                cond_grid.values.sum(dim=0) * 
                grid.values.sum(dim=0).clamp(0)
            ).sum().item()

            m.loc[idx, grid_type+'_rec_elem_prod'] = (
                cond_grid.elem_values.sum(dim=0) * 
                grid.elem_values.sum(dim=0).clamp(0)
            ).sum().item()

            m.loc[idx, grid_type+'_rec_prop_prod'] = (
                cond_grid.prop_values.sum(dim=0) * 
                grid.prop_values.sum(dim=0).clamp(0)
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
        self,
        idx,
        struct_type,
        struct,
        ref_struct=None,
        mean_counts=None,
        ref_only=False,
    ):
        m = self.metrics

        if not ref_only:

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
            rmsd = liGAN.metrics.compute_struct_rmsd(ref_struct, struct)
            m.loc[idx, struct_type+'_RMSD'] = rmsd

        if not ref_only:
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

    def compute_mol_metrics(
        self,
        idx,
        mol_type,
        mol,
        ref_mol=None,
        ref_only=False,
        use_cond_min=False,
    ):
        m = self.metrics

        # check molecular validity
        valid, reason = mol.validate()

        if not ref_only:
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
        if not ref_only:
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

        if 'uff_mol' not in mol.info:
            return

        # UFF energy minimization
        if use_cond_min and 'cond_uff_mol' in mol.info: # handle diff_cond_transform with no diff_cond_structs
            uff_mol = mol.info['cond_uff_mol']
        else:
            uff_mol = mol.info['uff_mol']
        uff_init = uff_mol.info['E_init']
        uff_min = uff_mol.info['E_min']
        uff_rmsd = uff_mol.info['min_rmsd']

        m.loc[idx, mol_type+'_UFF_init'] = uff_init
        m.loc[idx, mol_type+'_UFF_min'] = uff_min
        m.loc[idx, mol_type+'_UFF_rmsd'] = uff_rmsd
        m.loc[idx, mol_type+'_UFF_error'] = uff_mol.info['min_error']
        m.loc[idx, mol_type+'_UFF_time'] = uff_mol.info['min_time']

        # compare energy to ref mol, before and after minimizing
        if ref_mol:
            ref_uff_mol = ref_mol.info['uff_mol']
            ref_uff_init = ref_uff_mol.info['E_init']
            ref_uff_min = ref_uff_mol.info['E_min']
            ref_uff_rmsd = ref_uff_mol.info['min_rmsd']
            m.loc[idx, mol_type+'_UFF_init_diff'] = uff_init - ref_uff_init
            m.loc[idx, mol_type+'_UFF_min_diff'] = uff_min - ref_uff_min
            m.loc[idx, mol_type+'_UFF_rmsd_diff'] = uff_rmsd - ref_uff_rmsd

        if 'gni_mol' not in mol.info:
            return

        # gnina energy minimization
        if use_cond_min and 'cond_gni_mol' in mol.info:
            gni_mol = mol.info['cond_gni_mol']
        else:
            gni_mol = mol.info['gni_mol']
        vina_aff = gni_mol.info.get('minimizedAffinity', np.nan)
        vina_rmsd = gni_mol.info.get('minimizedRMSD', np.nan)
        cnn_pose = gni_mol.info.get('CNNscore', np.nan)
        cnn_aff = gni_mol.info.get('CNNaffinity', np.nan)

        m.loc[idx, mol_type+'_vina_aff'] = vina_aff
        m.loc[idx, mol_type+'_vina_rmsd'] = vina_rmsd
        m.loc[idx, mol_type+'_cnn_pose'] = cnn_pose
        m.loc[idx, mol_type+'_cnn_aff'] = cnn_aff
        m.loc[idx, mol_type+'_gnina_error'] = gni_mol.info['error']

        # compare gnina metrics to ref mol
        if ref_mol:
            ref_gni_mol = ref_mol.info['gni_mol']
            try:
                ref_vina_aff = ref_gni_mol.info['minimizedAffinity']
            except KeyError:
                print(ref_gni_mol.info)
                raise
            ref_vina_rmsd = ref_gni_mol.info['minimizedRMSD']
            ref_cnn_pose = ref_gni_mol.info['CNNscore']
            ref_cnn_aff = ref_gni_mol.info['CNNaffinity']

            m.loc[idx, mol_type+'_vina_aff_diff'] = vina_aff - ref_vina_aff
            m.loc[idx, mol_type+'_vina_rmsd_diff'] = vina_rmsd - ref_vina_rmsd
            m.loc[idx, mol_type+'_cnn_pose_diff'] = cnn_pose - ref_cnn_pose
            m.loc[idx, mol_type+'_cnn_aff_diff'] = cnn_aff - ref_cnn_aff


def read_rec_from_pdb_file(pdb_file):

    rec_mol = mols.Molecule.from_pdb(pdb_file, sanitize=False)
    try:
        Chem.SanitizeMol(rec_mol)
    except Chem.MolSanitizeException:
        pass
    return rec_mol


def read_lig_from_sdf_file(sdf_file, use_ob=True):
    '''
    Try to find the real molecule in data_root using the
    source path in the data file, without file extension.
    '''
    if use_ob: # read and add Hs with OpenBabel, then convert to RDkit
        lig_mol = mols.read_ob_mols_from_file(sdf_file, 'sdf')[0]
        lig_mol.AddHydrogens()
        lig_mol = mols.Molecule.from_ob_mol(lig_mol)

    else: # read and add Hs with RDKit (need to sanitize before add Hs)
        lig_mol = mols.Molecule.from_sdf(sdf_file, sanitize=False, idx=0)

    try: # need to do this to get ring info, etc.
        lig_mol.sanitize()
    except Chem.MolSanitizeException:
        pass

    if not use_ob: # add Hs with rdkit (after sanitize)
        lig_mol = lig_mol.add_hs()

    return lig_mol


def write_atom_types_to_file(types_file, atom_types):
    with open(types_file, 'w') as f:
        f.write('\n'.join(str(a) for a in atom_types))


def write_latent_vec_to_file(latent_file, latent_vec):

    with open(latent_file, 'w') as f:
        for value in latent_vec:
            f.write(str(value.item()) + '\n')


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
                    r'^.*(molecules|structs)/({}_.*)\.sdf(\.gz)?$'.format(
                        re.escape(out_prefix)
                    ),
                    str(sdf_file)
                )
                obj_name = m.group(2)
            except AttributeError:
                print(sdf_file, file=sys.stderr)
                raise
            f.write('load {}, {}\n'.format(sdf_file, obj_name))

        f.write('util.cbam *rec_src\n')
        f.write('util.cbag *lig_src\n')
        f.write('util.cbac *lig_gen_fit_add\n')
