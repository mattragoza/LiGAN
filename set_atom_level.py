import re
from pymol import cmd, stored

import channel_info as ci


def set_atom_level(level, selection_keyword=''):
    channels = ci.get_default_channels(True, True)
    channel_names = [channel_name for (channel_name, _, _) in channels]
    channel_index = {channel_name: i for i, channel_name in enumerate(channel_names)}
    pattern = re.compile('^(.+)_({})_grid$'.format('|'.join(channel_names)))
    map_objects = {}
    for obj in cmd.get_names('objects'):
        match = pattern.match(obj)
        if match:
            prefix = match.group(1)
            if prefix not in map_objects:
                map_objects[prefix] = []
            map_objects[prefix].append(obj)
    for prefix in map_objects:
        surface_objects = []
        for map_object in map_objects[prefix]:
            match = pattern.match(map_object)
            channel_name = match.group(2)
            channel = channels[channel_index[channel_name]]
            element = channel[1]
            color = ci.elem_color_map[element]
            surface_object = '{}_{}_surface'.format(prefix, channel_name)
            if selection_keyword in map_object:
                cmd.isosurface(surface_object, map_object, level=level)
                cmd.color(color, surface_object)
                surface_objects.append(surface_object)
        if surface_objects:
            surface_group = '{}_surfaces'.format(prefix)
            cmd.group(surface_group, ' '.join(surface_objects))


cmd.extend('set_atom_level', set_atom_level)