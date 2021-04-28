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
        self.coords = torch.as_tensor(coords, device=device)
        self.types = torch.as_tensor(types, device=device)
        self.typer = typer

        if bonds is not None:
            self.bonds = torch.as_tensor(bonds, device=device)
        else:
            self.bonds = None

        self.info = info

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
    def from_gninatypes(cls, gtypes_file, channels, **info):
        xyz, c = read_gninatypes_file(gtypes_file, channels)
        return AtomStruct(xyz, c, channels, **info)

    @classmethod
    def from_rd_mol(cls, rd_mol, c, channels, **info):
        xyz = rd_mol.GetConformer(0).GetPositions()
        return cls(xyz, c, channels, **info)

    @classmethod
    def from_sdf(cls, sdf_file, channels, **info):
        rd_mol = molecules.read_rd_mols_from_sdf_file(sdf_file)[0]
        channels_file = os.path.splitext(sdf_file)[0] + '.channels'
        c = read_channels_from_file(channels_file, channels)
        return cls.from_rd_mol(rd_mol, c, channels)

    @property
    def n_atoms(self):
        return self.xyz.shape[0]

    @property
    def type_counts(self):
        return count_types(self.c, len(self.channels))

    @property
    def center(self):
        if self.n_atoms > 0:
            return self.xyz.mean(dim=0)
        else:
            return np.nan

    @property
    def radius(self):
        if self.n_atoms > 0:
            return (self.xyz - self.center[None,:]).norm(dim=1).max().item()
        else:
            return np.nan

    def to(self, device):
        return AtomStruct(
            self.xyz, self.c, self.channels, self.bonds, device=device
        )
    
    def to_ob_mol(self):
        mol = molecules.make_ob_mol(
            self.xyz.cpu().numpy(),
            self.c.cpu().numpy(),
            self.bonds.cpu().numpy(),
            self.channels
        )
        return mol

    def to_rd_mol(self):
        mol = molecules.make_rd_mol(
            self.xyz.cpu().numpy().astype(float),
            self.c.cpu().numpy(),
            None if self.bonds is None else self.bonds.cpu().numpy(),
            self.channels
        )
        return mol

    def to_sdf(self, sdf_file):
        if sdf_file.endswith('.gz'):
            outfile = gzip.open(sdf_file, 'wt')
        else:
            outfile = open(sdf_file, 'wt')
        molecules.write_rd_mol_to_sdf_file(outfile, self.to_rd_mol())
        outfile.close()

    def add_bonds(self, tol=0.0):

        atomic_radii = torch.tensor(
            [c.atomic_radius for c in self.channels],
            device=self.c.device
        )
        atom_dist2 = (
            (self.xyz[None,:,:] - self.xyz[:,None,:])**2
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


def read_gninatypes_file(gtypes_file, channels):
    channel_names = [c.name for c in channels]
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
    with open(channels_file, 'r') as f:
        return np.array([
            int(c) for c in f.read().rstrip().split(' ')
        ])


def count_types(c, n_types, dtype=None):
    '''
    Provided a vector of type indices c, return a
    vector of type counts where type_counts[i] is
    the number of occurences of type index i in c.
    '''
    count = torch.zeros(n_types, dtype=dtype, device=c.device)
    for i in c:
        count[i] += 1
    return count
