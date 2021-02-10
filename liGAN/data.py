import torch
from torch import nn
import molgrid


class AtomGridData(nn.Module):

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
        split_rec_lig=False,
        ligand_only=False,
        rec_molcache='',
        lig_molcache='',
        device='cuda',
    ):
        super().__init__()
        
        # create receptor and ligand atom typers
        self.rec_typer = molgrid.FileMappedGninaTyper(rec_map_file)
        self.lig_typer = molgrid.FileMappedGninaTyper(lig_map_file)

        # create example provider
        self.ex_provider = molgrid.ExampleProvider(
            self.rec_typer,
            self.lig_typer,
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
            device=device,
        )
        self.labels = torch.zeros(
            batch_size, dtype=torch.float32, device=device
        )
        self.batch_size = batch_size

        self.random_rotation = random_rotation
        self.random_translation = random_translation

        self.split_rec_lig = split_rec_lig
        self.ligand_only = ligand_only

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

        # get next batch of structures and labels
        examples = self.ex_provider.next_batch(self.grids.shape[0])
        self.grid_maker.forward(
            examples,
            self.grids,
            random_rotation=self.random_rotation,
            random_translation=self.random_translation,
        )
        examples.extract_label(0, self.labels)

        if self.split_rec_lig or self.ligand_only:

            rec_grids, lig_grids = torch.split(
                self.grids,
                [self.n_rec_channels, self.n_lig_channels],
                dim=1,
            )
            if self.ligand_only:
                return lig_grids, self.labels
            else:
                return (rec_grids, lig_grids), self.labels
        else:
            return self.grids, self.labels
