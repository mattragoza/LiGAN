import sys, os, re, time
import torch
from torch import nn

import molgrid
from . import atom_types, atom_structs, atom_grids
from .atom_types import AtomTyper


class AtomGridData(nn.Module):

    def __init__(
        self,
        data_root,
        batch_size,
        rec_typer,
        lig_typer,
        resolution=0.5,
        dimension=None,
        grid_size=None,
        shuffle=False,
        random_rotation=False,
        random_translation=0.0,
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
        self.lig_typer = AtomTyper.get_typer(*lig_typer.split('-'))
        self.rec_typer = AtomTyper.get_typer(*rec_typer.split('-'))

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

        # create molgrid maker and output tensors
        self.grid_maker = molgrid.GridMaker(
            resolution=resolution,
            dimension=dimension,
            gaussian_radius_multiple=-1.5,
        )
        self.grids = torch.zeros(
            batch_size,
            self.n_rec_channels + self.n_lig_channels,
            *self.grid_maker.spatial_grid_dimensions(),
            dtype=torch.float32,
            device=device,
        )
        self.labels = torch.zeros(
            batch_size, dtype=torch.float32, device=device
        )
        self.transforms = [None for i in range(batch_size)]
        self.batch_size = batch_size

        self.random_rotation = random_rotation
        self.random_translation = random_translation
        self.debug = debug

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
        return self.grids.device

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

    def populate(self, data_file):
        self.ex_provider.populate(data_file)

    def forward(
        self,
        split_rec_lig=False,
        ligand_only=False,
    ):
        assert len(self) > 0, 'data is empty'

        # get next batch of structures and labels
        examples = self.ex_provider.next_batch(self.batch_size)
        examples.extract_label(0, self.labels)
        t1 = time.time()

        rec_structs = []
        lig_structs = []
        for i, example in enumerate(examples):
            t0 = time.time()

            rec_coord_set, lig_coord_set = example.coord_sets
            transform = molgrid.Transform(
                lig_coord_set.center(),
                self.random_translation,
                self.random_rotation,
            )
            self.transforms[i] = transform # store transforms
            transform.forward(example, example)
            self.grid_maker.forward(example, self.grids[i])
            t1 = time.time()

            rec_struct = atom_structs.AtomStruct.from_coord_set(
                rec_coord_set,
                self.rec_typer,
                device=self.grids.device
            )
            lig_struct = atom_structs.AtomStruct.from_coord_set(
                lig_coord_set,
                self.lig_typer,
                device=self.grids.device
            )
            rec_structs.append(rec_struct)
            lig_structs.append(lig_struct)

            t2 = time.time()
            if self.debug:
                print(t1-t0, t2-t1)
        
        if split_rec_lig or ligand_only:

            rec_grids, lig_grids = torch.split(
                self.grids,
                [self.n_rec_channels, self.n_lig_channels],
                dim=1,
            )
            if ligand_only:
                return lig_grids, lig_structs, self.labels
            else:
                return (
                    (rec_grids, lig_grids),
                    (rec_structs, lig_structs),
                    self.labels
                )
        else:
            return self.grids, (rec_structs, lig_structs), self.labels
