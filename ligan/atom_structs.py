import os, struct
import numpy as np
import torch

from . import atom_types, molecules


class AtomStruct(object):
    '''
    A structure of 3D atom coordinates and type
    vectors, stored as torch tensors along with
    a reference to the source atom typer.
    '''
    def __init__(
        self,
        coords,
        types,
        typer,
        dtype=None,
        device=None,
        **info
    ):
        self.check_shapes(coords, types, typer)

        # omit atoms with zero type vectors
        nonzero = (types > 0).any(axis=1)
        coords = coords[nonzero]
        types = types[nonzero]

        self.coords = torch.as_tensor(coords, dtype=dtype, device=device)
        self.types = torch.as_tensor(types, dtype=dtype, device=device)
        self.typer = typer

        self.info = info

        # compute these lazily, since they're expensive and not always needed
        self._atom_types = None
        self._atomic_radii = None

    @staticmethod
    def check_shapes(coords, types, typer):
        assert len(coords.shape) == 2, coords.shape
        assert len(types.shape) == 2, types.shape
        assert coords.shape[0] == types.shape[0], (coords.shape[0], types.shape[0])
        assert coords.shape[1] == 3, coords.shape[1]
        assert types.shape[1] == typer.n_types, (types.shape[1], typer.n_types)
        assert ((0 <= types) & (types <= 1)).all(), set(types)

    @classmethod
    def from_coord_set(
        cls,
        coord_set,
        typer,
        data_root='',
        dtype=None,
        device=None,
        **info
    ):
        if not coord_set.has_vector_types():
            coord_set.make_vector_types()

        # if already on gpu, we shouldn't copy back and forth
        assert not coord_set.coords.ongpu(), 'coords on gpu'
        assert not coord_set.type_vector.ongpu(), 'types on gpu'

        return cls(
            coords=coord_set.coords.tonumpy(),
            types=coord_set.type_vector.tonumpy(), # should be float
            typer=typer,
            dtype=dtype,
            device=typer.device if device is None else device,
            src_file=os.path.join(data_root, coord_set.src) if data_root else coord_set.src,
            **info
        )

    @classmethod
    def from_gninatypes(cls, gtypes_file, typer, **info):
        coords, types = read_gninatypes_file(gtypes_file, typer)
        return AtomStruct(coords, types, typer, **info)

    @classmethod
    def from_rd_mol(cls, rd_mol, types, typer, **info):
        coords = rd_mol.GetConformer(0).GetPositions()
        return cls(coords, types, typer, **info)

    @classmethod
    def from_sdf(cls, sdf_file, typer, **info):
        rd_mol = molecules.read_rd_mols_from_sdf_file(sdf_file)[0]
        channels_file = os.path.splitext(sdf_file)[0] + '.channels'
        types = read_channels_from_file(channels_file, channels)
        return cls.from_rd_mol(rd_mol, types, channels)

    @property
    def atom_types(self):
        if self._atom_types is None:
            self._atom_types = [
                self.typer.get_atom_type(t) for t in self.types
            ]
        return self._atom_types

    @property
    def atomic_radii(self):
        if self._atomic_radii is None:
            self._atomic_radii = torch.as_tensor([
                self.typer.radius_func(a.atomic_num) for a in self.atom_types
            ], dtype=self.dtype, device=self.device)
        return self._atomic_radii

    @property
    def n_atoms(self):
        return self.coords.shape[0]

    @property
    def type_counts(self):
        return self.types.sum(dim=0)

    @property
    def elem_counts(self):
        return self.types[:,:self.typer.n_elem_types].sum(dim=0)

    @property
    def prop_counts(self):
        return self.types[:,self.typer.n_elem_types:].sum(dim=0)

    @property
    def center(self):
        if self.n_atoms > 0:
            return self.coords.mean(dim=0)
        else:
            return np.nan

    @property
    def radius(self):
        if self.n_atoms > 0:
            return (self.coords - self.center).norm(dim=1).max().item()
        else:
            return np.nan

    @property
    def dtype(self):
        return self.coords.dtype

    @property
    def device(self):
        return self.coords.device

    def to(self, dtype, device):
        return AtomStruct(
            coords=self.coords,
            types=self.types,
            typer=self.typer,
            dtype=dtype,
            device=device,
            **self.info
        )
    
    def to_ob_mol(self):
        return molecules.make_ob_mol(
            coords=self.coords.cpu().numpy(),
            types=self.types.cpu().numpy(),
            bonds=None,
            typer=self.typer,
        )

    def to_rd_mol(self):
        return molecules.make_rd_mol(
            coords=self.coords.cpu().numpy().astype(float),
            types=self.types.cpu().numpy(),
            bonds=None,
            typer=self.typer,
        )

    def to_sdf(self, sdf_file):
        if sdf_file.endswith('.gz'):
            outfile = gzip.open(sdf_file, 'wt')
        else:
            outfile = open(sdf_file, 'wt')
        molecules.write_rd_mol_to_sdf_file(outfile, self.to_rd_mol())
        outfile.close()


def read_gninatypes_file(gtypes_file, typer):
    # TODO allow vector typed gninatypes files?
    channel_names = typer.get_type_names()
    channel_name_idx = {n: i for i, n in enumerate(channel_names)}
    xyz, c = [], []
    with open(gtypes_file, 'rb') as f:
        atom_bytes = f.read(16)
        while atom_bytes:
            x, y, z, t = struct.unpack('fffi', atom_bytes)
            smina_type = atom_types.smina_types[t]
            channel_name = 'Ligand' + smina_type.name
            if channel_name in channel_name_idx:
                c_ = channel_names.index(channel_name)
                xyz.append([x, y, z])
                c.append(c_)
            atom_bytes = f.read(16)
    assert xyz and c, lig_file
    return np.array(xyz), np.array(c)


def read_channels_from_file(channels_file):
    # TODO allow vector typed channels files
    with open(channels_file, 'r') as f:
        return np.array([
            int(c) for c in f.read().rstrip().split(' ')
        ])


def count_index_types(types, n_types, dtype=None):
    '''
    Provided a vector of index types, return a
    vector of type counts where type_counts[i] is
    the number of occurences of type index i in c.
    '''
    type_counts = torch.zeros(n_types, dtype=dtype, device=c.device)
    for i in types:
        type_counts[i] += 1
    return type_counts
