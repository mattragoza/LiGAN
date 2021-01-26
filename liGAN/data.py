import torch
import molgrid


class AtomGridData(object):

    def __init__(
        self,
        data_root,
        batch_size,
        rec_map_file,
        lig_map_file,
        resolution,
        dimension,
        shuffle,
        random_rotation=True,
        random_translation=2.0,
        rec_molcache='',
        lig_molcache='',
    ):
        # create receptor and/or ligand atom typers
        atom_typers = []

        if rec_map_file:
            self.rec_typer = molgrid.FileMappedGninaTyper(rec_map_file)
            atom_typers.append(self.rec_typer)
        else:
            self.rec_typer = None

        if lig_map_file:
            self.lig_typer = molgrid.FileMappedGninaTyper(lig_map_file)
            atom_typers.append(self.lig_typer)
        else:
            self.lig_typer = None

        # create example provider
        print(atom_typers)
        self.ex_provider = molgrid.ExampleProvider(
            *atom_typers,
            data_root=data_root,
            recmolcache=rec_molcache,
            ligmolcache=lig_molcache,
            shuffle=shuffle,
        )

        # create molgrid maker and output tensors
        self.grid_maker = molgrid.GridMaker(resolution, dimension)
        self.grids = torch.zeros(
            batch_size,
            self.n_channels,
            *self.grid_maker.spatial_grid_dimensions(),
            dtype=torch.float32,
            device='cuda',
        )
        self.labels = torch.zeros(
            batch_size, dtype=torch.float32, device='cuda'
        )

        self.random_rotation = random_rotation
        self.random_translation = random_translation

    @classmethod
    def from_param(cls, param):

        return cls(
            data_root=param.root_folder,
            batch_size=param.batch_size,
            rec_map_file=param.recmap,
            lig_map_file=param.ligmap,
            resolution=param.resolution,
            dimension=param.dimension,
            shuffle=param.shuffle,
            random_rotation=param.random_rotation,
            random_translation=param.random_translate,
            rec_molcache=param.recmolcache,
            lig_molcache=param.ligmolcache,
        )

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
    def size(self):
        return self.ex_provider.size()

    def populate(self, data_file):
        self.ex_provider.populate(data_file)

    def forward(self):
        assert self.size > 0

        examples = self.ex_provider.next_batch(self.grids.shape[0])
        self.grid_maker.forward(
            examples,
            self.grids,
            random_rotation=self.random_rotation,
            random_translation=self.random_translation,
        )
        examples.extract_label(0, self.labels)

        if self.n_rec_channels > 0 and self.n_lig_channels > 0:
            return torch.split(
                self.grids, [self.n_rec_channels, self.n_lig_channels], dim=1
            ), self.labels
        else:
            return self.grids, self.labels
