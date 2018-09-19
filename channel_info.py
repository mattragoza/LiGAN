from collections import namedtuple, defaultdict


atom_type = namedtuple('atom_type', ['name', 'element', 'covalent_radius', 'xs_radius'])


smina_types = [
    atom_type("Hydrogen", "H", 0.370000, 0.37),
    atom_type("PolarHydrogen", "H", 0.370000, 0.370000),
    atom_type("AliphaticCarbonXSHydrophobe", "C", 0.770000, 1.900000),
    atom_type("AliphaticCarbonXSNonHydrophobe", "C", 0.770000, 1.900000),
    atom_type("AromaticCarbonXSHydrophobe", "C", 0.770000, 1.900000),
    atom_type("AromaticCarbonXSNonHydrophobe", "C", 0.770000, 1.900000),
    atom_type("Nitrogen", "N", 0.750000, 1.800000),
    atom_type("NitrogenXSDonor", "N", 0.750000, 1.800000),
    atom_type("NitrogenXSDonorAcceptor", "N", 0.750000, 1.800000),
    atom_type("NitrogenXSAcceptor", "N", 0.750000, 1.800000),
    atom_type("Oxygen", "O", 0.730000, 1.700000),
    atom_type("OxygenXSDonor", "O", 0.730000, 1.700000),
    atom_type("OxygenXSDonorAcceptor", "O", 0.730000, 1.700000),
    atom_type("OxygenXSAcceptor", "O", 0.730000, 1.700000),
    atom_type("Sulfur", "S", 1.020000, 2.000000),
    atom_type("SulfurAcceptor", "S", 1.020000, 2.000000),
    atom_type("Phosphorus", "P", 1.060000, 2.100000),
    atom_type("Fluorine", "F", 0.710000, 1.500000),
    atom_type("Chlorine", "Cl", 0.990000, 1.800000),
    atom_type("Bromine", "Br", 1.140000, 2.000000),
    atom_type("Iodine", "I", 1.330000, 2.200000),
    atom_type("Magnesium", "Mg", 1.300000, 1.200000),
    atom_type("Manganese", "Mn", 1.390000, 1.200000),
    atom_type("Zinc", "Zn", 1.310000, 1.200000),
    atom_type("Calcium", "Ca", 1.740000, 1.200000),
    atom_type("Iron", "Fe", 1.250000, 1.200000),
    atom_type("GenericMetal", "M", 1.750000, 1.200000),
    atom_type("Boron", "B", 0.90, 1.920000)
]

# default atom type scheme
rec_idx = [2, 3, 4, 5, 24, 25, 21, 6, 9, 7, 8, 13, 12, 16, 14, 23]
lig_idx = [2, 3, 4, 5, 19, 18, 17, 6, 9, 7, 8, 10, 13, 12, 16, 14, 15, 20, 27]


elem_color_map = defaultdict(
    lambda: 'green',
    C='grey',
    N='blue',
    O='red',
    P='orange',
    S='yellow'
)


max_n_bonds = defaultdict(
    lambda: 8,
    H=1,
    C=4,
    N=3,
    O=2,
    P=6,
    S=6,
    F=1,
    Cl=1,
    Br=1,
    I=1,
    Mg=2,
    Mn=8,
    Zn=6,
    Ca=2,
    Fe=6,
    B=4
)


def get_default_channels(rec, lig, use_covalent_radius=False):
    idx = []
    if rec:
        idx += rec_idx
    if lig:
        idx += lig_idx
    channels = []
    for i in idx:
        channel_name = smina_types[i].name
        element = smina_types[i].element
        if use_covalent_radius:
            atom_radius = smina_types[i].covalent_radius
        else:
            atom_radius = smina_types[i].xs_radius
        channels.append((channel_name, element, atom_radius))
    return channels       


def get_channels_for_grids(grids, use_covalent_radius=False):
    '''
    Infer the atom types for a set of grids by the number of channels.
    '''
    n_channels = grids.shape[0]
    if n_channels == len(rec_idx):
        return get_default_channels(True, False, use_covalent_radius)
    elif n_channels == len(lig_idx):
        return get_default_channels(False, True, use_covalent_radius)
    elif n_channels == len(rec_idx) + len(lig_idx):
        return get_default_channels(True, True, use_covalent_radius)
    else:
        raise Exception('could not infer atom types for grids with {} channels'.format(n_channels))


