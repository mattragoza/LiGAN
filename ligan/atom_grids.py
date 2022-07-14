import numpy as np
import molgrid
import torch


class AtomGridder(molgrid.Coords2Grid):
    '''
    A class for converting atomic structures
    to atomic density grids.
    '''
    def __init__(self, resolution=0.5, dimension=23.5):
        gmaker = molgrid.GridMaker(resolution, dimension, gaussian_radius_multiple=-1.5)
        super().__init__(gmaker)

    @property
    def resolution(self):
        return self.gmaker.get_resolution()

    def forward(self, struct):
        self.center = tuple(float(v) for v in struct.center)
        values = super().forward(
            struct.coords, struct.types, struct.atomic_radii
        )
        return AtomGrid(values, struct.center, self.resolution, struct.typer)


class AtomGrid(object):
    '''
    A 3D grid representation of a molecular structure.

    Each atom is represented as a Gaussian-like density,
    and the grid values are computed by summing across
    these densities in separate channels per atom type.
    '''
    def __init__(
        self,
        values,
        center,
        resolution,
        typer,
        dtype=None,
        device=None,
        **info
    ):
        self.check_shapes(values, center, typer)
        self.values = torch.as_tensor(values, dtype=dtype, device=device)
        self.center = torch.as_tensor(center, dtype=dtype, device=device)
        self.resolution = float(resolution)
        self.typer = typer
        self.info = info

    @staticmethod
    def check_shapes(values, center, typer):
        assert len(values.shape) == 4, values.shape
        assert values.shape[0] == typer.n_types, (values.shape[0], typer.n_types)
        assert values.shape[1] == values.shape[2] == values.shape[3], values.shape[1:]
        assert center.shape == (3,), center.shape

    @classmethod
    def from_dx(cls, dx_prefix, typer, dtype=None, device=None, **info):
        values, center, resolution = read_grid_from_dx_files(dx_prefix, typer)
        return cls(
            values, center, resolution, typer, dtype, device, **info
        )

    @property
    def shape(self):
        return self.values.shape
    
    @property
    def n_channels(self):
        return self.values.shape[0]

    @property
    def size(self):
        return self.values.shape[1]

    @property
    def dimension(self):
        return size_to_dimension(self.size, self.resolution)

    @property
    def origin(self):
        return center_to_origin(self.center, self.size, self.resolution)

    @property
    def n_elem_channels(self):
        return self.typer.n_elem_types

    @property
    def n_prop_channels(self):
        return self.n_channels - self.n_elem_channels

    @property
    def elem_values(self):
        return self.values[:self.n_elem_channels]

    @property
    def prop_values(self):
        return self.values[self.n_elem_channels:]

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def device(self):
        return self.values.device

    def to(self, device, dtype=None):
        return self.new_like(values=self.values, dtype=dtype, device=device)

    def to_dx(self, dx_prefix, center=None):
        return write_grid_to_dx_files(
            dx_prefix=dx_prefix,
            values=self.values.to('cpu', dtype=torch.float64),
            center=self.center.cpu().numpy() if center is None else center,
            resolution=self.resolution,
            typer=self.typer,
        )

    def new_like(self, values, dtype=None, device=None, **info):
        '''
        Return an AtomGrid with the same grid settings but new values.
        '''
        return AtomGrid(
            values=values,
            center=self.center,
            resolution=self.resolution,
            typer=self.typer,
            dtype=self.dtype if dtype is None else dtype,
            device=self.device if device is None else device,
            **info
        )

    def get_coords(self, idx_xyz):
        '''
        Return the 3D coordinates located
        at each of the provided (x,y,z)
        spatial indices of the grid.
        '''
        return spatial_index_to_coords(
            idx_xyz,
            self.center,
            self.size,
            self.resolution
        )


def center_to_origin(center, size, resolution):
    '''
    Compute the coordinates of the origin of
    a grid with the provided center, size,
    and resolution.
    '''
    return center - resolution * (size - 1) / 2.0


def unravel_index(idx, shape):
    ret = []
    for dim in shape:
        ret.append(idx % dim)
        idx = idx // dim
    return torch.stack(ret, dim=1)


def spatial_index_to_coords(idx_xyz, center, size, resolution):
    '''
    Compute 3D coordinates located at each of
    the provided (x,y,z) spatial indices of a
    grid with center, size, and resolution.
    '''
    origin = center_to_origin(center, size, resolution)
    return origin + resolution * idx_xyz.float()


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


def round_dimension(dimension, resolution):
    '''
    Return the actual dimension of a grid
    with the provided resolution that fits
    the provided dimension within it.
    '''
    return size_to_dimension(
        dimension_to_size(dimension, resolution),
        resolution
    )


def write_grid_to_dx_file(dx_file, values, center, resolution):
    '''
    Write a grid with the provided values,
    center, and resolution to a .dx file.
    '''
    assert len(values.shape) == 3
    assert values.shape[0] == values.shape[1] == values.shape[2]
    assert len(center) == 3

    size = values.shape[0]
    origin = np.array(center) - resolution*(size - 1)/2.

    lines = [
        'object 1 class gridpositions counts {:d} {:d} {:d}\n'.format(
            size, size, size
        ),
        'origin {:.5f} {:.5f} {:.5f}\n'.format(*origin),
        'delta {:.5f} 0 0\n'.format(resolution),
        'delta 0 {:.5f} 0\n'.format(resolution),
        'delta 0 0 {:.5f}\n'.format(resolution),
        'object 2 class gridconnections counts {:d} {:d} {:d}\n'.format(
            size, size, size
        ),
        'object 3 class array type double rank 0 items ' \
            + '[ {:d} ] data follows\n'.format(size**3),
    ]
    n_points = 0
    line = ''
    for i in range(size):
        for j in range(size):
            for k in range(size):
                line += '{:.10f}'.format(values[i][j][k])
                n_points += 1
                if n_points % 3 == 0:
                    lines.append(line + '\n')
                    line = ''
                else:
                    line += ' '

    if line: # if n_points is not divisible by 3, need last line
        lines.append(line)

    with open(dx_file, 'w') as f:
        f.write(''.join(lines))


def write_grid_to_dx_files(dx_prefix, values, center, resolution, typer):
    '''
    Write each a multi-channel grid with the provided
    values, center, and resolution to .dx files, using
    the prefix and type names.
    '''
    assert values.shape[0] == typer.n_types
    dx_files = []
    for i, type_name in enumerate(typer.get_type_names()):
        dx_file = '{}_{}.dx'.format(dx_prefix, type_name)
        write_grid_to_dx_file(dx_file, values[i], center, resolution)
        dx_files.append(dx_file)
    return dx_files


def parse_vector(line, dtype, n=3, delim=' '):
    fields = line.rstrip().rsplit(delim, n)[-n:]
    return np.array(fields, dtype=dtype)


def read_grid_from_dx_file(dx_file):
    
    with open(dx_file, 'r') as f:
        lines = f.readlines()

    shape = parse_vector(lines[0], dtype=int)
    assert shape[0] == shape[1] == shape[2]
    size = shape[0]

    origin = parse_vector(lines[1], dtype=float)

    delta0 = parse_vector(lines[2], dtype=float)
    delta1 = parse_vector(lines[3], dtype=float)
    delta2 = parse_vector(lines[4], dtype=float)
    assert delta0[0] == delta1[1] == delta2[2]
    resolution = delta0[0]

    center = origin + resolution*(size - 1)/2

    values = []
    for line in lines[7:]:
        values.extend(parse_vector(line, dtype=float))

    assert len(values) == size**3
    return np.array(values).reshape(*shape), center, resolution


def read_grid_from_dx_files(dx_prefix, typer):
    values = []
    for i, type_name in enumerate(typer.get_type_names()):
        dx_file = '{}_{}.dx'.format(dx_prefix, type_name)
        values_i, center_i, resolution_i = read_grid_from_dx_file(dx_file)
        values.append(values_i)
        if i == 0:
            center, resolution = center_i, resolution_i
        else:
            assert center_i == center
            assert resolution_i == resolution
    return np.stack(values), center, resolution
