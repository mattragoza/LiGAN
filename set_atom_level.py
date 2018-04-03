import re
from pymol import cmd, stored
from generate import rec_channels, lig_channels, get_color_for_channel


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
        surface_group = '{}_surfaces'.format(prefix)
        cmd.group(surface_group, ' '.join(surface_objects))


cmd.extend('set_atom_level', set_atom_level)