import sys
from collections import namedtuple, defaultdict
import numpy as np
import torch
import molgrid

from .common import catch_exception

try:
    from openbabel import openbabel as ob

    try:
        table = ob.OBElementTable()
    except AttributeError:
        table = ob

    get_atomic_num = table.GetAtomicNum
    get_name = table.GetName
    get_symbol = table.GetSymbol
    get_max_bonds = table.GetMaxBonds
    get_rgb = table.GetRGB

except Exception as e:
    print(e, file=sys.stderr)

    def get_rgb(atomic_num):
        return [
            [0.07, 0.5, 0.7],
            [0.75, 0.75, 0.75],
            [0.85, 1.0, 1.0],
            [0.8, 0.5, 1.0],
            [0.76, 1.0, 0.0],
            [1.0, 0.71, 0.71],
            [0.4, 0.4, 0.4],
            [0.05, 0.05, 1.0],
            [1.0, 0.05, 0.05],
            [0.5, 0.7, 1.0],
            [0.7, 0.89, 0.96],
            [0.67, 0.36, 0.95],
            [0.54, 1.0, 0.0],
            [0.75, 0.65, 0.65],
            [0.5, 0.6, 0.6],
            [1.0, 0.5, 0.0],
            [0.7, 0.7, 0.0],
            [0.12, 0.94, 0.12],
            [0.5, 0.82, 0.89],
            [0.56, 0.25, 0.83],
            [0.24, 1.0, 0.0],
            [0.9, 0.9, 0.9],
            [0.75, 0.76, 0.78],
            [0.65, 0.65, 0.67],
            [0.54, 0.6, 0.78],
            [0.61, 0.48, 0.78],
            [0.88, 0.4, 0.2],
            [0.94, 0.56, 0.63],
            [0.31, 0.82, 0.31],
            [0.78, 0.5, 0.2],
            [0.49, 0.5, 0.69],
            [0.76, 0.56, 0.56],
            [0.4, 0.56, 0.56],
            [0.74, 0.5, 0.89],
            [1.0, 0.63, 0.0],
            [0.65, 0.16, 0.16],
            [0.36, 0.72, 0.82],
            [0.44, 0.18, 0.69],
            [0.0, 1.0, 0.0],
            [0.58, 1.0, 1.0],
            [0.58, 0.88, 0.88],
            [0.45, 0.76, 0.79],
            [0.33, 0.71, 0.71],
            [0.23, 0.62, 0.62],
            [0.14, 0.56, 0.56],
            [0.04, 0.49, 0.55],
            [0.0, 0.41, 0.52],
            [0.88, 0.88, 1.0],
            [1.0, 0.85, 0.56],
            [0.65, 0.46, 0.45],
            [0.4, 0.5, 0.5],
            [0.62, 0.39, 0.71],
            [0.83, 0.48, 0.0],
            [0.58, 0.0, 0.58],
            [0.26, 0.62, 0.69],
            [0.34, 0.09, 0.56],
            [0.0, 0.79, 0.0],
            [0.44, 0.83, 1.0],
            [1.0, 1.0, 0.78],
            [0.85, 1.0, 0.78],
            [0.78, 1.0, 0.78],
            [0.64, 1.0, 0.78],
            [0.56, 1.0, 0.78],
            [0.38, 1.0, 0.78],
            [0.27, 1.0, 0.78],
            [0.19, 1.0, 0.78],
            [0.12, 1.0, 0.78],
            [0.0, 1.0, 0.61],
            [0.0, 0.9, 0.46],
            [0.0, 0.83, 0.32],
            [0.0, 0.75, 0.22],
            [0.0, 0.67, 0.14],
            [0.3, 0.76, 1.0],
            [0.3, 0.65, 1.0],
            [0.13, 0.58, 0.84],
            [0.15, 0.49, 0.67],
            [0.15, 0.4, 0.59],
            [0.09, 0.33, 0.53],
            [0.9, 0.85, 0.68],
            [0.8, 0.82, 0.12],
            [0.71, 0.71, 0.76],
            [0.65, 0.33, 0.3],
            [0.34, 0.35, 0.38],
            [0.62, 0.31, 0.71],
            [0.67, 0.36, 0.0],
            [0.46, 0.31, 0.27],
            [0.26, 0.51, 0.59],
            [0.26, 0.0, 0.4],
            [0.0, 0.49, 0.0],
            [0.44, 0.67, 0.98],
            [0.0, 0.73, 1.0],
            [0.0, 0.63, 1.0],
            [0.0, 0.56, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.42, 1.0],
            [0.33, 0.36, 0.95],
            [0.47, 0.36, 0.89],
            [0.54, 0.31, 0.89],
            [0.63, 0.21, 0.83],
            [0.7, 0.12, 0.83]
        ][atomic_num]


atom_type = namedtuple('atom_type', ['name', 'atomic_num', 'symbol', 'covalent_radius', 'xs_radius'])


smina_types = [
    atom_type("Hydrogen",                       1,  "H", 0.37, 0.37),
    atom_type("PolarHydrogen",                  1,  "H", 0.37, 0.37),
    atom_type("AliphaticCarbonXSHydrophobe",    6,  "C", 0.77, 1.90),
    atom_type("AliphaticCarbonXSNonHydrophobe", 6,  "C", 0.77, 1.90),
    atom_type("AromaticCarbonXSHydrophobe",     6,  "C", 0.77, 1.90),
    atom_type("AromaticCarbonXSNonHydrophobe",  6,  "C", 0.77, 1.90),
    atom_type("Nitrogen",                       7,  "N", 0.75, 1.80),
    atom_type("NitrogenXSDonor",                7,  "N", 0.75, 1.80),
    atom_type("NitrogenXSDonorAcceptor",        7,  "N", 0.75, 1.80),
    atom_type("NitrogenXSAcceptor",             7,  "N", 0.75, 1.80),
    atom_type("Oxygen",                         8,  "O", 0.73, 1.70),
    atom_type("OxygenXSDonor",                  8,  "O", 0.73, 1.70),
    atom_type("OxygenXSDonorAcceptor",          8,  "O", 0.73, 1.70),
    atom_type("OxygenXSAcceptor",               8,  "O", 0.73, 1.70),
    atom_type("Sulfur",                        16,  "S", 1.02, 2.00),
    atom_type("SulfurAcceptor",                16,  "S", 1.02, 2.00),
    atom_type("Phosphorus",                    15,  "P", 1.06, 2.10),
    atom_type("Fluorine",                       9,  "F", 0.71, 1.50),
    atom_type("Chlorine",                      17, "Cl", 0.99, 1.80),
    atom_type("Bromine",                       35, "Br", 1.14, 2.00),
    atom_type("Iodine",                        53,  "I", 1.33, 2.20),
    atom_type("Magnesium",                     12, "Mg", 1.30, 1.20),
    atom_type("Manganese",                     25, "Mn", 1.39, 1.20),
    atom_type("Zinc",                          30, "Zn", 1.31, 1.20),
    atom_type("Calcium",                       20, "Ca", 1.74, 1.20),
    atom_type("Iron",                          26, "Fe", 1.25, 1.20),
    atom_type("GenericMetal",                  -1,  "M", 1.75, 1.20),
    atom_type("Boron",                          5,  "B", 0.90, 1.92)
]

smina_types_by_name = dict((t.name, t) for t in smina_types)


channel = namedtuple(
    'channel', ['name', 'atomic_num', 'symbol', 'atomic_radius']
)


class Unknown(object):
    '''
    An object whose only property is that
    it is considered equal to everthing.

    This can be used as the last element
    in a type range to catch any values
    not explicitly specified before it.

    Note that the original value will not
    be recoverable through atom fitting.
    '''
    def __eq__(self, other):
        '''
        Returns True.
        '''
        return True


UNK = Unknown()


class AtomTyper(molgrid.PythonCallbackVectorTyper):
    '''
    An object for converting OBAtoms to atom type vectors.

    An AtomTyper is defined by a list of typing functions
    and ranges of valid output values for each function.

    It converts an atom to a type vector by calling each
    typing function on the atom, using the type ranges to
    convert each output value to a one-hot vector, then
    concatenating the one-hot vectors into a single list.

    A function giving the atomic radius is also needed.
    '''
    def __init__(self, type_funcs, type_ranges, radius_func):
        assert len(type_funcs) == len(type_ranges)
        assert type_funcs[0] == ob.OBAtom.GetAtomicNum
        self.type_funcs = type_funcs
        self.type_ranges = type_ranges
        self.radius_func = radius_func
        super().__init__(
            lambda a: (self.get_type_vector(a), radius_func(a)),
            self.n_types
        )

    @property
    def n_types(self):
        return sum(len(r) for r in self.type_ranges)

    def get_type_vector(self, ob_atom):
        type_vec = []
        for func, range_ in zip(self.type_funcs, self.type_ranges):
            value = func(ob_atom)
            type_vec += make_one_hot(value, range_)
        return type_vec

    @classmethod
    def get_typer(cls, atom_props, radius_type):

        type_funcs = [ob.OBAtom.GetAtomicNum]
        type_ranges = [
            [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, UNK]
        ]
        # TODO when to call AddHydrogens, and what if
        # hydrogens count as UNK when not explicit?
        # won't hydrogen properties like formal charge
        # also contribute density when H is not explicit?

        if 'h' in atom_props: # explicit hydrogens
            type_ranges[0].insert(0, 1)

        if 'o' in atom_props:
            type_funcs += [ob.OBAtom.IsAromatic]
            type_ranges += [1]

        if 'a' in atom_props:
            type_funcs += [ob.OBAtom.IsHbondAcceptor]
            type_ranges += [1]

        if 'd' in atom_props:
            type_funcs += [ob.OBAtom.IsHbondDonor]
            type_ranges += [1]

        if 'c' in atom_props:
            type_funcs += [ob.OBAtom.GetFormalCharge]
            type_ranges += [-1, 0, 1]

        if 'n' in atom_props:
            type_funcs += [get_n_hydrogens]
            type_ranges += [0, 1, 2, UNK]

        if radius_type == 'd': # default
            radius_func = get_default_radius

        elif radius_type == 'c': # covalent
            radius_func = get_covalent_radius

        return cls(type_counts, type_ranges, radius_func)


def make_one_hot(value, range_):
    vec = [0] * len(range_)
    try:
        vec[range_.index(value)] = 1
    except ValueError: # ignore
        pass
    return vec


def get_channel_color(channel):
    if 'LigandAliphatic' in channel.name:
        return [1.00, 1.00, 1.00] #[1.00, 0.50, 1.00]
    elif 'LigandAromatic' in channel.name:
        return [0.83, 0.83, 0.83] #[1.00, 0.00, 1.00]
    elif 'ReceptorAliphatic' in channel.name:
        return [1.00, 1.00, 1.00]
    elif 'ReceptorAromatic' in channel.name:
        return [0.83, 0.83, 0.83]
    else:
        return get_rgb(channel.atomic_num)


def get_channel(t, use_covalent_radius, name_prefix):
    name = name_prefix + t.name
    atomic_radius = t.covalent_radius if use_covalent_radius else t.xs_radius
    return channel(name, t.atomic_num, t.symbol, atomic_radius)


def get_channels_by_index(type_idx, use_covalent_radius=False, name_prefix=''):
    channels = []
    for i in type_idx:
        t = smina_types[i]
        c = get_channel(t, use_covalent_radius, name_prefix)
        channels.append(c)
    return channels


def get_channels_by_name(type_names, use_covalent_radius=False, name_prefix=''):
    channels = []
    for type_name in type_names:
        t = smina_types_by_name[type_name]
        c = get_channel(t, use_covalent_radius, name_prefix)
        channels.append(c)
    return channels


def get_channels_from_map(map_, use_covalent_radius=False, name_prefix=''):
    return get_channels_by_name(map_.get_type_names(), use_covalent_radius, name_prefix)


def get_channels_from_file(map_file, use_covalent_radius=False, name_prefix=''):
    import molgrid
    map_ = molgrid.FileMappedGninaTyper(map_file)
    return get_channels_from_map(map_, use_covalent_radius, name_prefix)

def get_n_unknown_channels(n, radius=1.0):
    channels = []
    for i in range(n):
        channels.append(channel('Unknown', 0, 'X', radius))
    return channels


def get_default_rec_channels(use_covalent_radius=False, name_prefix=''):
    idx = [2, 3, 4, 5, 24, 25, 21, 6, 9, 7, 8, 13, 12, 16, 14, 23]
    return get_channels_by_index(idx, use_covalent_radius, name_prefix)


def get_default_lig_channels(use_covalent_radius=False, name_prefix=''):
    idx = [2, 3, 4, 5, 19, 18, 17, 6, 9, 7, 8, 10, 13, 12, 16, 14, 15, 20, 27]
    return get_channels_by_index(idx, use_covalent_radius, name_prefix)


def get_default_channels(use_covalent_radius=False):
    rec_channels = get_default_rec_channels(use_covalent_radius)
    lig_channels = get_default_lig_channels(use_covalent_radius)
    return rec_channels + lig_channels
