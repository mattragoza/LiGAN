import sys, os, re, time
import pandas as pd
from openbabel import openbabel as ob
import torch
from torch import nn, utils

import molgrid
from . import atom_types, atom_structs, atom_grids
from .atom_types import AtomTyper
from .interpolation import TransformInterpolation


class MolDataset(utils.data.IterableDataset):

    def __init__(
        self, rec_typer, lig_typer, data_file, data_root, verbose=False
    ):
        super().__init__()

        # what is this unknown column?
        #  it's positive for low_rmsd, negative for ~low_rmsd,
        #  but otherwise same absolute distributions...
        data_cols = [
            'low_rmsd',
            'true_aff',
            'xtal_rmsd',
            'rec_src',
            'lig_src',
            'vina_aff'
        ]
        self.data = pd.read_csv(
            data_file, sep=' ', names=data_cols, index_col=False
        )
        self.root_dir = data_root

        ob_conv = ob.OBConversion()
        ob_conv.SetInFormat('pdb')
        self.read_pdb = ob_conv.ReadFile

        ob_conv = ob.OBConversion()
        ob_conv.SetInFormat('sdf')
        self.read_sdf = ob_conv.ReadFile

        self.mol_cache = dict()
        self.verbose = verbose

        self.rec_typer = rec_typer
        self.lig_typer = lig_typer

    def read_mol(self, mol_src, pdb=False):

        mol_file = os.path.join(self.root_dir, mol_src)
        if self.verbose:
            print('Reading ' + mol_file)

        assert os.path.isfile(mol_file), 'file does not exist'

        mol = ob.OBMol()
        if pdb:
            assert self.read_pdb(mol, mol_file), 'failed to read mol'
        else:
            assert self.read_sdf(mol, mol_file), 'failed to read mol'

        mol.AddHydrogens()
        assert mol.NumAtoms() > 0, 'mol has zero atoms'

        mol.SetTitle(mol_src)
        return mol

    def get_rec_mol(self, mol_src):
        if mol_src not in self.mol_cache:
            self.mol_cache[mol_src] = self.read_mol(mol_src, pdb=True)
        return self.mol_cache[mol_src]

    def get_lig_mol(self, mol_src):
        if mol_src not in self.mol_cache:
            self.mol_cache[mol_src] = self.read_mol(mol_src, pdb=False)
        return self.mol_cache[mol_src]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data.iloc[idx]
        rec_mol = self.get_rec_mol(example.rec_src)
        lig_mol = self.get_lig_mol(example.lig_src)
        return rec_mol, lig_mol

    def __iter__(self):
        for rec_src, lig_src in zip(self.data.rec_src, self.data.lig_src):
            rec_mol = self.get_rec_mol(rec_src)
            lig_mol = self.get_lig_mol(lig_src)
            yield rec_mol, lig_mol


class AtomGridData(object):

    def __init__(
        self,
        data_file,
        data_root,
        batch_size,
        rec_typer,
        lig_typer,
        use_rec_elems=True,
        resolution=0.5,
        dimension=None,
        grid_size=None,
        shuffle=False,
        random_rotation=False,
        random_translation=0.0,
        diff_cond_transform=False,
        diff_cond_structs=False,
        n_samples=1,
        rec_molcache=None,
        lig_molcache=None,
        cache_structs=True,
        device='cuda',
        debug=False,
    ):
        super().__init__()

        assert (dimension or grid_size) and not (dimension and grid_size), \
            'must specify one of either dimension or grid_size'
        if grid_size:
            dimension = atom_grids.size_to_dimension(grid_size, resolution)
        
        # create receptor and ligand atom typers
        self.lig_typer = AtomTyper.get_typer(*lig_typer.split('-'), rec=False)
        self.rec_typer = \
            AtomTyper.get_typer(*rec_typer.split('-'), rec=use_rec_elems)

        atom_typers = [self.rec_typer, self.lig_typer]
        if diff_cond_structs: # duplicate atom typers
            atom_typers *= 2

        # create example provider
        self.ex_provider = molgrid.ExampleProvider(
            *atom_typers,
            data_root=data_root,
            recmolcache=rec_molcache or '',
            ligmolcache=lig_molcache or '',
            cache_structs=cache_structs,
            shuffle=shuffle,
            num_copies=n_samples,
        )

        # create molgrid maker
        self.grid_maker = molgrid.GridMaker(
            resolution=resolution,
            dimension=dimension,
            gaussian_radius_multiple=-1.5,
        )
        self.batch_size = batch_size

        # transformation settings
        self.random_rotation = random_rotation
        self.random_translation = random_translation
        self.diff_cond_transform = diff_cond_transform
        self.diff_cond_structs = diff_cond_structs
        self.debug = debug
        self.device = device

        # transform interpolation state
        self.cond_interp = TransformInterpolation(n_samples=n_samples)

        # load data from file
        self.ex_provider.populate(data_file)

    @classmethod
    def from_param(cls, param):

        return cls(
            data_root=param.root_folder,
            batch_size=param.batch_size,
            rec_typer=param.recmap,
            lig_typer=param.ligmap,
            resolution=param.resolution,
            grid_size=atom_grids.dimension_to_size(
                param.dimension, param.resolution
            ),
            shuffle=param.shuffle,
            random_rotation=param.random_rotation,
            random_translation=param.random_translate,
            rec_molcache=param.recmolcache,
            lig_molcache=param.ligmolcache,
        )

    @property
    def root_dir(self):
        return self.ex_provider.settings().data_root

    @property
    def n_rec_channels(self):
        return self.rec_typer.num_types() if self.rec_typer else 0

    @property
    def n_lig_channels(self):
        return self.lig_typer.num_types() if self.lig_typer else 0
 
    @property
    def n_channels(self):
        return self.n_rec_channels + self.n_lig_channels

    @property
    def resolution(self):
        return self.grid_maker.get_resolution()

    @property
    def dimension(self):
        return self.grid_maker.get_dimension()

    @property
    def grid_size(self):
        return atom_grids.dimension_to_size(self.dimension, self.resolution)

    def __len__(self):
        return self.ex_provider.size()

    def forward(self, interpolate=False, spherical=False):
        assert len(self) > 0, 'data is empty'

        # get next batch of structures
        examples = self.ex_provider.next_batch(self.batch_size)
        labels = torch.zeros(self.batch_size, device=self.device)
        examples.extract_label(0, labels)

        # create lists for examples, structs and transforms
        batch_list = lambda: [None] * self.batch_size

        input_examples = batch_list()
        input_rec_structs = batch_list()
        input_lig_structs = batch_list()
        input_transforms = batch_list()

        cond_examples = batch_list()
        cond_rec_structs = batch_list()
        cond_lig_structs = batch_list()
        cond_transforms = batch_list()

        # create output tensors for atomic density grids
        input_grids = torch.zeros(
            self.batch_size,
            self.n_channels,
            *self.grid_maker.spatial_grid_dimensions(),
            dtype=torch.float32,
            device=self.device,
        )
        cond_grids = torch.zeros(
            self.batch_size,
            self.n_channels,
            *self.grid_maker.spatial_grid_dimensions(),
            dtype=torch.float32,
            device=self.device,
        )

        # split examples, create structs and transforms
        for i, ex in enumerate(examples):

            if self.diff_cond_structs:

                # different input and conditional molecules
                input_rec_coord_set, input_lig_coord_set, \
                    cond_rec_coord_set, cond_lig_coord_set = ex.coord_sets

                # split example into inputs and conditions
                input_ex = molgrid.Example()
                input_ex.coord_sets.append(input_rec_coord_set)
                input_ex.coord_sets.append(input_lig_coord_set)

                cond_ex = molgrid.Example()
                cond_ex.coord_sets.append(cond_rec_coord_set)
                cond_ex.coord_sets.append(cond_lig_coord_set)

            else: # same conditional molecules as input
                input_rec_coord_set, input_lig_coord_set = ex.coord_sets
                cond_rec_coord_set, cond_lig_coord_set = ex.coord_sets
                input_ex = cond_ex = ex

            # store split examples for gridding
            input_examples[i] = input_ex
            cond_examples[i] = cond_ex

            # convert coord sets to atom structs
            input_rec_structs[i] = atom_structs.AtomStruct.from_coord_set(
                input_rec_coord_set,
                typer=self.rec_typer,
                data_root=self.root_dir,
                device=self.device
            )
            input_lig_structs[i] = atom_structs.AtomStruct.from_coord_set(
                input_lig_coord_set,
                typer=self.lig_typer,
                data_root=self.root_dir,
                device=self.device
            )
            if self.diff_cond_structs:
                cond_rec_structs[i] = atom_structs.AtomStruct.from_coord_set(
                    cond_rec_coord_set,
                    typer=self.rec_typer,
                    data_root=self.root_dir,
                    device=self.device
                )
                cond_lig_structs[i] = atom_structs.AtomStruct.from_coord_set(
                    cond_lig_coord_set,
                    typer=self.lig_typer,
                    data_root=self.root_dir,
                    device=self.device
                )
            else: # same structs as input
                cond_rec_structs[i] = input_rec_structs[i]
                cond_lig_structs[i] = input_lig_structs[i]

            # create input transform
            input_transforms[i] = molgrid.Transform(
                center=input_lig_coord_set.center(),
                random_translate=self.random_translation,
                random_rotation=self.random_rotation,
            )
            if self.diff_cond_transform:

                # create conditional transform
                cond_transforms[i] = molgrid.Transform(
                    center=cond_lig_coord_set.center(),
                    random_translate=self.random_translation,
                    random_rotation=self.random_rotation,
                )
            else: # same transform as input
                cond_transforms[i] = input_transforms[i]
        
        if interpolate: # interpolate conditional transforms
            # i.e. location and orientation of conditional grid
            if not self.cond_interp.is_initialized:
                self.cond_interp.initialize(cond_examples[0])
            cond_transforms = self.cond_interp(
                transforms=cond_transforms,
                spherical=spherical,
            )

        # create density grids
        for i in range(self.batch_size):

            # create input density grid
            self.grid_maker.forward(
                input_examples[i],
                input_transforms[i],
                input_grids[i]
            )
            if (
                self.diff_cond_transform or self.diff_cond_structs or interpolate
            ):
                # create conditional density grid
                self.grid_maker.forward(
                    cond_examples[i],
                    cond_transforms[i],
                    cond_grids[i]
                )
            else: # same density grid as input
                cond_grids[i] = input_grids[i]

        input_structs = (input_rec_structs, input_lig_structs)
        cond_structs = (cond_rec_structs, cond_lig_structs)
        transforms = (input_transforms, cond_transforms)
        return (
            input_grids, cond_grids,
            input_structs, cond_structs,
            transforms, labels
        )

    def split_channels(self, grids):
        '''
        Split receptor and ligand grid channels.
        '''
        return torch.split(
            grids, [self.n_rec_channels, self.n_lig_channels], dim=1
        )

    def find_real_mol(self, mol_src, ext):
        return find_real_mol(mol_src, self.root_dir, ext)


def find_real_mol(mol_src, data_root, ext):

    m = re.match(r'(.+)_(\d+)((\..*)+)', mol_src)
    if m:
        mol_name = m.group(1)
        pose_idx = int(m.group(2))
    else:
        m = re.match(r'(.+)((\..*)+)', mol_src)
        mol_name = m.group(1)
        pose_idx = 0

    mol_file = os.path.join(data_root, mol_name + ext)
    return mol_file, mol_name, pose_idx
