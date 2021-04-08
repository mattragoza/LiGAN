import numpy as np
import torch


class AtomGrid(object):
    '''
    A 3D grid representation of a molecular structure.
    
    Each atom is represented as a Gaussian-like density,
    and the grid values are computed by summing across
    these densities in separate channels per atom type.
    '''
    def __init__(
        self, values, channels, center, resolution, device=None, **info
    ):
        self.check_shapes(values, channels, center)
        self.values = torch.as_tensor(values, device=device)
        self.channels = channels
        self.center = torch.as_tensor(center, device=device)
        self.resolution = float(resolution)
        self.info = info

    @staticmethod
    def check_shapes(values, channels, center):
        assert len(values.shape) == 4
        assert values.shape[0] == len(channels)
        assert values.shape[1] == values.shape[2] == values.shape[3]
        assert center.shape == (3,)

    @property
    def n_channels(self):
        return len(self.channels)

    @property
    def size(self):
        return self.values.shape[1]

    @property
    def dimension(self):
        return size_to_dimension(self.size, self.resolution)

    def to(self, device):
        self.values = self.values.to(device)
        self.center = self.center.to(device)

    def to_dx(self, dx_prefix, center=None):
        write_grids_to_dx_files(
            out_prefix=dx_prefix,
            grids=self.values.cpu().numpy(),
            channels=self.channels,
            center=self.center.cpu().numpy() if center is None else center,
            resolution=self.resolution)

    def new_like(self, values, device=None, **info):
        '''
        Return an AtomGrid with the same grid settings but new values.
        '''
        return AtomGrid(
            values,
            self.channels,
            self.center,
            self.resolution,
            self.device if device is None else device,
            **info
        )


def size_to_dimension(size, resolution):
    '''
    Compute the side length of a cubic grid with
    the given size (num. points along each axis)
    and resolution.
    '''
    return (size - 1) * resolution


def dimension_to_size(dimension, resolution):
    '''
    Compute the number of points along each axis
    of a cubic grid spanning the given dimension
    (side length) at the given resolution.
    '''
    return int(np.ceil(dimension / resolution + 1))


def write_grid_to_dx_file(dx_file, grid, center, resolution):
    '''
    Write a grid with a center and resolution to a .dx file.
    '''
    if len(grid.shape) != 3 or len(set(grid.shape)) != 1:
        raise ValueError('grid must have three equal dimensions')
    if len(center) != 3:
        raise ValueError('center must be a vector of length 3')
    dim = grid.shape[0]
    origin = np.array(center) - resolution*(dim-1)/2.
    with open(dx_file, 'w') as f:
        f.write('object 1 class gridpositions counts {:d} {:d} {:d}\n'.format(dim, dim, dim))
        f.write('origin {:.5f} {:.5f} {:.5f}\n'.format(*origin))
        f.write('delta {:.5f} 0 0\n'.format(resolution))
        f.write('delta 0 {:.5f} 0\n'.format(resolution))
        f.write('delta 0 0 {:.5f}\n'.format(resolution))
        f.write('object 2 class gridconnections counts {:d} {:d} {:d}\n'.format(dim, dim, dim))
        f.write('object 3 class array type double rank 0 items [ {:d} ] data follows\n'.format(dim**3))
        total = 0
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    f.write('{:.10f}'.format(grid[i][j][k]))
                    total += 1
                    if total % 3 == 0:
                        f.write('\n')
                    else:
                        f.write(' ')


def write_grids_to_dx_files(out_prefix, grids, channels, center, resolution):
    '''
    Write each of a list of grids a separate .dx file, using the channel names.
    '''
    dx_files = []
    for grid, channel in zip(grids, channels):
        dx_file = '{}_{}.dx'.format(out_prefix, channel.name)
        write_grid_to_dx_file(dx_file, grid, center, resolution)
        dx_files.append(dx_file)
    return dx_files
