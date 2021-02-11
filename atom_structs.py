import os, struct
import numpy as np

import molecules


class AtomStruct(object):
    '''
    A typed atomic structure.
    '''
    def __init__(self, xyz, c, channels, bonds=None, **info):

        if len(xyz.shape) != 2:
            raise ValueError('AtomStruct xyz must have 2 dims')
        if len(c.shape) != 1:
            raise ValueError('AtomStruct c must have 1 dimension')
        if xyz.shape[0] != c.shape[0]:
            raise ValueError('first dim of AtomStruct xyz and c must be equal')
        if xyz.shape[1] != 3:
            raise ValueError('second dim of AtomStruct xyz must be 3')
        if any(c < 0) or any(c >= len(channels)):
            raise ValueError('invalid channel index in AtomStruct c')

        self.xyz = xyz
        self.c = c
        self.channels = channels

        if bonds is not None:
            if bonds.shape != (self.n_atoms, self.n_atoms):
                raise ValueError('AtomStruct bonds must have shape (n_atoms, n_atoms)')
            self.bonds = bonds
        else:
            self.bonds = np.zeros((self.n_atoms, self.n_atoms))

        if self.n_atoms > 0:
            self.center = self.xyz.mean(0)
            self.radius = max(np.linalg.norm(self.xyz - self.center, axis=1))
        else:
            self.center = np.full(3, np.nan)
            self.radius = np.nan

        self.info = info

    @classmethod
    def from_gninatypes(cls, gtypes_file, channels, **info):
        xyz, c = read_gninatypes_file(gtypes_file, channels)
        return AtomStruct(xyz, c, channels, **info)

    @classmethod
    def from_coord_set(cls, coord_set, channels, **info):
        if not coord_set.has_indexed_types():
            raise ValueError(
                'can only make AtomStruct from CoordinateSet with indexed types'
            )
        xyz = coord_set.coords.tonumpy()
        c = coord_set.type_index.tonumpy().astype(int)
        return cls(xyz, c, channels, **info)

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

    def to_ob_mol(self):
        mol = molecules.make_ob_mol(self.xyz.astype(float), self.c, self.bonds, self.channels)
        return mol

    def to_rd_mol(self):
        mol = molecules.make_rd_mol(self.xyz.astype(float), self.c, self.bonds, self.channels)
        return mol

    def to_sdf(self, sdf_file):
        if sdf_file.endswith('.gz'):
            outfile = gzip.open(sdf_file, 'wt')
        else:
            outfile = open(sdf_file, 'wt')
        molecules.write_rd_mols_to_sdf_file(outfile, [self.to_rd_mol()])
        outfile.close()

    def add_bonds(self, tol=0.0):

        nax = np.newaxis
        channel_radii = np.array([c.atomic_radius for c in self.channels])

        atom_dist2 = ((self.xyz[nax,:,:] - self.xyz[:,nax,:])**2).sum(axis=2)
        max_bond_dist2 = channel_radii[self.c][nax,:] + channel_radii[self.c][:,nax]
        self.bonds = (atom_dist2 < max_bond_dist2 + tol)


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
