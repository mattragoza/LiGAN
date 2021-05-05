import os, struct
import numpy as np
import torch

from . import atom_types, molecules


class AtomStruct(object):
    '''
    A structure of 3D atom coordinates and type
    vectors, stored as torch tensors along with
    a reference to the source atom typer.

    An optional bond matrix can be provided, but
    this is not currently used for anything.
    '''
    def __init__(
        self, coords, types, typer, bonds=None, device=None, **info
    ):
        self.check_shapes(coords, types, typer, bonds)

        # omit atoms with zero type vectors
        nonzero = (types > 0).any(axis=1)
        coords = coords[nonzero]
        types = types[nonzero]

        self.coords = torch.as_tensor(coords, dtype=float, device=device)
        self.types = torch.as_tensor(types, dtype=float, device=device)
        self.typer = typer

        if bonds is not None:
            self.bonds = torch.as_tensor(bonds, device=device)
        else:
            self.bonds = None

        self.info = info

        self.atom_types = [
            self.typer.get_atom_type(t) for t in self.types
        ]

    @staticmethod
    def check_shapes(coords, types, typer, bonds):
        assert len(coords.shape) == 2
        assert len(types.shape) == 2
        assert coords.shape[0] == types.shape[0]
        assert coords.shape[1] == 3
        assert types.shape[1] == typer.n_types
        assert ((types == 0) | (types == 1)).all()
        if bonds is not None:
            assert bonds.shape == (coords.shape[0], coords.shape[0])

    @classmethod
    def from_coord_set(cls, coord_set, typer, device, **info):

        if not coord_set.has_vector_types():
            coord_set.make_vector_types()

        return cls(
            coords=coord_set.coords.tonumpy(),
            types=coord_set.type_vector.tonumpy().astype(float),
            typer=typer,
            device=device,
            src_file=coord_set.src,
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
    def n_atoms(self):
        return self.coords.shape[0]

    @property
    def type_counts(self):
        return self.types.sum(dim=0)

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

    def to(self, device):
        return AtomStruct(
            self.coords, self.types, self.typer, self.bonds, device=device
        )
    
    def to_ob_mol(self):
        return molecules.make_ob_mol(
            self.coords.cpu().numpy(),
            self.types.cpu().numpy(),
            None if self.bonds is None else self.bonds.cpu().numpy(),
            self.typer
        )

    def to_rd_mol(self):
        return molecules.make_rd_mol(
            self.coords.cpu().numpy().astype(float),
            self.types.cpu().numpy(),
            None if self.bonds is None else self.bonds.cpu().numpy(),
            self.typer
        )

    def to_sdf(self, sdf_file):
        if sdf_file.endswith('.gz'):
            outfile = gzip.open(sdf_file, 'wt')
        else:
            outfile = open(sdf_file, 'wt')
        molecules.write_rd_mol_to_sdf_file(outfile, self.to_rd_mol())
        outfile.close()

    def add_bonds(self, tol=0.0):

        # TODO get atomic radii from types and typer
        atomic_radii = torch.tensor(
            [ob.GetCovalentRad(t) for t in self.types],
            device=self.c.device
        )
        atom_dist2 = (
            (self.coords[None,:,:] - self.coords[:,None,:])**2
        ).sum(axis=2)

        max_bond_dist2 = (
            atomic_radii[self.c][None,:] + atomic_radii[self.c][:,None]
        )
        self.bonds = (atom_dist2 < max_bond_dist2 + tol**2)

    def make_mol(self, verbose=False):
        '''
        Attempt to construct a valid molecule from an atomic
        structure by inferring bonds, setting aromaticity
        and connecting fragments, returning a Molecule.
        '''
        from . import dkoes_fitting
        init_mol = self.to_rd_mol()
        add_mol, n_misses, visited_mols = dkoes_fitting.make_rdmol(
            self, verbose
        )
        visited_mols = [init_mol] + visited_mols
        visited_mols = [molecules.Molecule(m) for m in visited_mols]
        return molecules.Molecule(
            add_mol, n_misses=n_misses, visited_mols=visited_mols
        )


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
