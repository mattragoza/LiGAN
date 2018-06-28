import re
from pymol import cmd, stored


# channel name, element, atom radius
rec_channels = [("rec_AliphaticCarbonXSHydrophobe", "C", 1.90),
                ("rec_AliphaticCarbonXSNonHydrophobe", "C", 1.90),
                ("rec_AromaticCarbonXSHydrophobe", "C", 1.90),
                ("rec_AromaticCarbonXSNonHydrophobe", "C", 1.90),
                ("rec_Calcium", "Ca", 1.20),
                ("rec_Iron", "Fe", 1.20),
                ("rec_Magnesium", "Mg", 1.20),
                ("rec_Nitrogen", "N", 1.80),
                ("rec_NitrogenXSAcceptor", "N", 1.80),
                ("rec_NitrogenXSDonor", "N", 1.80),
                ("rec_NitrogenXSDonorAcceptor", "N", 1.80),
                ("rec_OxygenXSAcceptor", "O", 1.70),
                ("rec_OxygenXSDonorAcceptor", "O", 1.70),
                ("rec_Phosphorus", "P", 2.10),
                ("rec_Sulfur", "S", 2.00),
                ("rec_Zinc", "Zn", 1.20)]

lig_channels = [("lig_AliphaticCarbonXSHydrophobe", "C", 1.90),
                ("lig_AliphaticCarbonXSNonHydrophobe", "C", 1.90),
                ("lig_AromaticCarbonXSHydrophobe", "C", 1.90),
                ("lig_AromaticCarbonXSNonHydrophobe", "C", 1.90),
                ("lig_Bromine", "Br", 2.00),
                ("lig_Chlorine", "Cl", 1.80),
                ("lig_Fluorine", "F", 1.50),
                ("lig_Nitrogen", "N", 1.80),
                ("lig_NitrogenXSAcceptor", "N", 1.80),
                ("lig_NitrogenXSDonor", "N", 1.80),
                ("lig_NitrogenXSDonorAcceptor", "N", 1.80),
                ("lig_Oxygen", "O", 1.70),
                ("lig_OxygenXSAcceptor", "O", 1.70),
                ("lig_OxygenXSDonorAcceptor", "O", 1.70),
                ("lig_Phosphorus", "P", 2.10),
                ("lig_Sulfur", "S", 2.00),
                ("lig_SulfurAcceptor", "S", 2.00),
                ("lig_Iodine", "I", 2.20),
                ("lig_Boron", "B", 1.92)]


def get_color_for_channel(channel_name):
    if 'Carbon' in channel_name:
        return 'grey'
    elif 'Oxygen' in channel_name:
        return 'red'
    elif 'Nitrogen' in channel_name:
        return 'blue'
    elif 'Phosphorus' in channel_name:
        return 'orange'
    elif 'Sulfur' in channel_name:
        return 'yellow'
    else:
        return 'green'


def set_atom_level(level, selection_keyword=''):
    channel_names = [x[0] for x in rec_channels + lig_channels]
    pattern = re.compile('^(.+)_({})_map$'.format('|'.join(channel_names)))
    map_objects = {}
    for object in cmd.get_names('objects'):
        match = pattern.match(object)
        if match:
            prefix = match.group(1)
            if prefix not in map_objects:
                map_objects[prefix] = []
            map_objects[prefix].append(object)
    for prefix in map_objects:
        surface_objects = []
        for map_object in map_objects[prefix]:
            match = pattern.match(map_object)
            channel_name = match.group(2)
            surface_object = '{}_{}_surface'.format(prefix, channel_name)
            if selection_keyword in map_object:
                cmd.isosurface(surface_object, map_object, level=level)
                cmd.color(get_color_for_channel(channel_name), surface_object)
                surface_objects.append(surface_object)
        if surface_objects:
            surface_group = '{}_surfaces'.format(prefix)
            cmd.group(surface_group, ' '.join(surface_objects))


cmd.extend('set_atom_level', set_atom_level)