from collections import namedtuple, defaultdict
import openbabel as ob


try:
    table = ob.OBElementTable()
except AttributeError:
    table = ob


get_atomic_num = table.GetAtomicNum
get_name = table.GetName
get_symbol = table.GetSymbol
get_max_bonds = table.GetMaxBonds
get_rgb = table.GetRGB


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


channel = namedtuple('channel', ['name', 'atomic_num', 'symbol', 'atomic_radius'])


def get_channel_color(channel):
    if 'LigandAliphatic' in channel.name:
        return [1.00, 0.50, 1.00]
    elif 'LigandAromatic' in channel.name:
        return [1.00, 0.00, 1.00]
    elif 'ReceptorAliphatic' in channel.name:
        return [1.00, 1.00, 1.00]
    elif 'ReceptorAromatic' in channel.name:
        return [0.83, 0.83, 0.83]
    else:
        return get_rgb(channel.atomic_num)


def get_smina_type_channels(idx, use_covalent_radius, name_prefix):
    channels = []
    for i in idx:
        t = smina_types[i]
        c = channel(
            name=name_prefix + t.name,
            atomic_num=t.atomic_num,
            symbol=t.symbol,
            atomic_radius=t.covalent_radius if use_covalent_radius else t.xs_radius,
        )
        channels.append(c)
    return channels


def get_default_rec_channels(use_covalent_radius=False):
    idx = [2, 3, 4, 5, 24, 25, 21, 6, 9, 7, 8, 13, 12, 16, 14, 23]
    return get_smina_type_channels(idx, use_covalent_radius, 'Receptor')


def get_default_lig_channels(use_covalent_radius=False):
    idx = [2, 3, 4, 5, 19, 18, 17, 6, 9, 7, 8, 10, 13, 12, 16, 14, 15, 20, 27]
    return get_smina_type_channels(idx, use_covalent_radius, 'Ligand') 


def get_default_channels(use_covalent_radius=False):
    rec_channels = get_default_rec_channels(use_covalent_radius)
    lig_channels = get_default_lig_channels(use_covalent_radius)
    return rec_channels + lig_channels
