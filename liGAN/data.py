import sys, os, re, time
import pandas as pd
from openbabel import openbabel as ob
import torch
from torch import nn, utils

import molgrid
from . import atom_types, atom_structs, atom_grids
from .atom_types import AtomTyper


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


class AtomGridData(nn.Module):

    def __init__(
        self,
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
        self.rec_typer = AtomTyper.get_typer(*rec_typer.split('-'), rec=use_rec_elems)

        # create example provider
        self.ex_provider = molgrid.ExampleProvider(
            self.rec_typer,
            self.lig_typer,
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
        self.labels = torch.zeros(batch_size, device=device)

        # create output tensors for input grids
        self.input_grids = torch.zeros(
            batch_size,
            self.n_rec_channels + self.n_lig_channels,
            *self.grid_maker.spatial_grid_dimensions(),
            dtype=torch.float32,
            device=device,
        )
        self.input_transforms = [None for i in range(batch_size)]

        # transformation settings
        self.random_rotation = random_rotation
        self.random_translation = random_translation
        self.diff_cond_transform = diff_cond_transform
        self.debug = debug

        if diff_cond_transform:

            # create separate output tensors for conditional grids
            self.cond_grids = torch.zeros(
                batch_size,
                self.n_rec_channels + self.n_lig_channels,
                *self.grid_maker.spatial_grid_dimensions(),
                dtype=torch.float32,
                device=device,
            )
            self.cond_transforms = [None for i in range(batch_size)]

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
    def device(self):
        return self.input_grids.device

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

    def populate(self, data_file):
        self.ex_provider.populate(data_file)

    def __len__(self):
        return self.ex_provider.size()

    def forward(self):
        assert len(self) > 0, 'data is empty'

        # get next batch of structures
        t0 = time.time()
        examples = self.ex_provider.next_batch(self.batch_size)
        examples.extract_label(0, self.labels)
        t_mol = time.time() - t0

        t_grid = 0
        t_struct = 0

        rec_structs = []
        lig_structs = []
        for i, example in enumerate(examples):
            t0 = time.time()

            rec_coord_set, lig_coord_set = example.coord_sets

            # store input transforms
            self.input_transforms[i] = molgrid.Transform(
                center=lig_coord_set.center(),
                random_translate=self.random_translation,
                random_rotation=self.random_rotation,
            )
            # create input density grids
            self.grid_maker.forward(
                example, self.input_transforms[i], self.input_grids[i]
            )

            if self.diff_cond_transform:

                # store conditional transforms
                self.cond_transforms[i] = molgrid.Transform(
                    center=lig_coord_set.center(),
                    random_translate=self.random_translation,
                    random_rotation=self.random_rotation,
                )
                # create conditional density grids
                self.grid_maker.forward(
                    example, self.cond_transforms[i], self.cond_grids[i]
                )

            t1 = time.time()

            # convert coord sets to atom structs
            rec_struct = atom_structs.AtomStruct.from_coord_set(
                rec_coord_set,
                typer=self.rec_typer,
                data_root=self.root_dir,
                device=self.device
            )
            lig_struct = atom_structs.AtomStruct.from_coord_set(
                lig_coord_set,
                typer=self.lig_typer,
                data_root=self.root_dir,
                device=self.device
            )
            rec_structs.append(rec_struct)
            lig_structs.append(lig_struct)

            t2 = time.time()
            t_grid += t1 - t0
            t_struct += t2 - t1

        t0 = time.time()
        structs = (rec_structs, lig_structs)

        input_grids = self.input_grids
        cond_grids = self.cond_grids if self.diff_cond_transform else None

        t_split = time.time() - t0
        t_total = t_mol + t_grid + t_struct + t_split

        if self.debug:
            print('{:.4f}s ({:.2f} {:.2f} {:.2f} {:.2f})'.format(
                t_total,
                100*t_mol/t_total,
                100*t_grid/t_total,
                100*t_struct/t_total,
                100*t_split/t_total,
            ))

        return input_grids, cond_grids, structs, self.labels

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
