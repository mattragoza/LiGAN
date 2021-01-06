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
        # create receptor and ligand atom typers
        rec_map = molgrid.FileMappedGninaTyper(rec_map_file)
        lig_map = molgrid.FileMappedGninaTyper(lig_map_file)
        self.rec_lig_split = rec_map.num_types()

        # create example provider
        self.ex_provider = molgrid.ExampleProvider(
            rec_map,
            lig_map,
            data_root=data_root,
            recmolcache=rec_molcache,
            ligmolcache=lig_molcache,
            shuffle=shuffle,
        )

        # create molgrid maker and output tensor
        self.grid_maker = molgrid.GridMaker(resolution, dimension)
        self.grid = torch.zeros(
            batch_size,
            rec_map.num_types() + lig_map.num_types(),
            *self.grid_maker.spatial_grid_dimensions(),
            dtype=torch.float32,
            device='cuda',
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

    def size(self):
        return self.ex_provider.size()

    def populate(self, data_file):
        self.ex_provider.populate(data_file)

    def forward(self):

        examples = self.ex_provider.next_batch(self.grid.shape[0])
        self.grid_maker.forward(
            examples,
            self.grid,
            random_rotation=self.random_rotation,
            random_translation=self.random_translation,
        )
        rec = self.grid[:,:self.rec_lig_split,...]
        lig = self.grid[:,self.rec_lig_split:,...]
        return rec, lig
