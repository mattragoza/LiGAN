import sys, re, glob, fnmatch
from collections import OrderedDict
from pymol import cmd, stored

sys.path.insert(0, '.')
import channel_info as ci


def set_atom_level(level, selection='*'):

    channels = ci.get_default_channels(True, True)
    channel_names = [c[0] for c in channels]
    channel_index = {n: i for i, n in enumerate(channel_names)}

    # first identify .dx atom grid information
    pattern = r'(.*)_({})\.dx'.format('|'.join(channel_names))
    dx_groups = OrderedDict()
    for obj in cmd.get_names('objects'):
        match = re.match(pattern, obj)
        if match:
            dx_prefix = match.group(1)
            if dx_prefix not in dx_groups:
                dx_groups[dx_prefix] = []
            dx_groups[dx_prefix].append(obj)

    surface_groups = OrderedDict()
    for dx_prefix in dx_groups:
        for dx_object in dx_groups[dx_prefix]:
            match = re.match(pattern, dx_object)
            channel_name = match.group(2)
            channel = channels[channel_index[channel_name]]
            element = channel[1]
            color = ci.elem_color_map[element]
            surface_object = dx_object.replace('.dx', '_surface')
            if fnmatch.fnmatch(dx_object, selection):
                match = re.match(r'(ROTATE|2A_lig_gen)_(\d+)', dx_prefix)
                s = int(match.group(2))+1 if match else 0
                cmd.isosurface(surface_object, dx_object, level=level, state=s)
                cmd.color(color, surface_object)
            if dx_prefix not in surface_groups:
                surface_groups[dx_prefix] = []
            surface_groups[dx_prefix].append(surface_object)

    for dx_prefix in surface_groups:
        surface_group = '{}_surface'.format(dx_prefix)
        cmd.group(surface_group, ' '.join(surface_groups[dx_prefix]))


def load_group(pattern, name):
    group_files = []
    for group_file in glob.glob(pattern):
        cmd.load(group_file, group_file)
        group_files.append(group_file)
    print(name, pattern, len(group_files))
    if group_files:
        cmd.group(name, ' '.join(group_files))


def my_rotate(name, axis, angle, states):
    for i in range(int(states)):
        cmd.create(name, name, 1, i+1)
        cmd.rotate(axis, float(angle)*i/int(states), name, state=i+1, camera=0)


cmd.extend('set_atom_level', set_atom_level)
cmd.extend('load_group', load_group)
cmd.extend('my_rotate', my_rotate)
