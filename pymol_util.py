import sys, re, glob, fnmatch
from collections import OrderedDict
from pymol import cmd, stored

sys.path.insert(0, '.')
import isoslider
import atom_types


def set_atom_level(level, selection='*', state=None, rec_map='my_rec_map', lig_map='my_lig_map'):

    rec_channels = atom_types.get_channels_from_file(rec_map, name_prefix='Receptor')
    lig_channels = atom_types.get_channels_from_file(lig_map, name_prefix='Ligand')

    channels = rec_channels + lig_channels
    channels_by_name = dict((c.name, c) for c in channels)

    for c in channels:
        cmd.set_color(c.name+'$', atom_types.get_channel_color(c))

    dx_regex = re.compile(r'(.*)_(\d+)_({})\.dx'.format('|'.join(channels_by_name)))

    surface_groups = OrderedDict()
    for dx_object in sorted(cmd.get_names('objects')):

        if not fnmatch.fnmatch(dx_object, selection):
            continue

        m = dx_regex.match(dx_object)
        if not m:
            continue

        grid_prefix = m.group(1)
        sample_idx = int(m.group(2))
        channel = channels_by_name[m.group(3)]

        if state is None:
            surface_state = sample_idx+1
        else:
            surface_state = state

        surface_object = '{}_{}_surface'.format(grid_prefix, channel.name)
        cmd.isosurface(surface_object, dx_object, level=level, state=surface_state)
        cmd.color(channel.name+'$', surface_object)

        if grid_prefix not in surface_groups:
            surface_groups[grid_prefix] = []

        if surface_object not in surface_groups[grid_prefix]:
            surface_groups[grid_prefix].append(surface_object)

    for grid_prefix, surface_objects in surface_groups.items():
        surface_group = '{}_surfaces'.format(grid_prefix)
        cmd.group(surface_group, ' '.join(surface_objects))


def load_group(pattern, name):
    group_objs = []
    for file in glob.glob(pattern):
        obj = os.path.basename(file)
        cmd.load(file, obj)
        group_objs.append(obj)
    if group_objs:
        cmd.group(name, ' '.join(group_objs))


def my_rotate(name, axis, angle, states, **kwargs):
    for i in range(int(states)):
        cmd.create(name, name, 1, i+1)
        cmd.rotate(axis, float(angle)*i/int(states), name, state=i+1, camera=0, **kwargs)


cmd.extend('set_atom_level', set_atom_level)
cmd.extend('load_group', load_group)
cmd.extend('my_rotate', my_rotate)
