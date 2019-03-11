import sys, re, glob, fnmatch
from collections import OrderedDict
from pymol import cmd, stored

sys.path.insert(0, '.')
import isoslider
import atom_types


def set_atom_level(level, selection='*', state=None):

    channels = atom_types.get_default_channels(True)
    channel_names = [c.name for c in channels]
    channels_by_name = {n: channels[i] for i, n in enumerate(channel_names)}

    for channel in channels:
        if 'Aliphatic' in channel.name:
            color = [1.0, 0.5, 1.0]
        elif 'Aromatic' in channel.name:
            color = [1.0, 0.0, 1.0]
        else:
            color = atom_types.get_rgb(channel.atomic_num)
        cmd.set_color(channel.name+'$', color)

    # first identify .dx atom grid information
    dx_pattern = r'(.*)_({})\.dx'.format('|'.join(channel_names))
    dx_groups = OrderedDict()
    for obj in sorted(cmd.get_names('objects')):

        match = re.match(dx_pattern, obj)
        if match:
            dx_prefix = match.group(1)
            if dx_prefix not in dx_groups:
                dx_groups[dx_prefix] = []
            dx_groups[dx_prefix].append(obj)

    surface_groups = OrderedDict()
    for dx_prefix in dx_groups:

        match = re.match(r'^(.*)_(\d+)$', dx_prefix)
        if match:
            surface_prefix = match.group(1)
            if state is None:
                state_ = int(match.group(2)) + 1
            else:
                state_ = state
        else:
            surface_prefix = dx_prefix
            state_ = state or 0

        for dx_object in dx_groups[dx_prefix]:

            if fnmatch.fnmatch(dx_object, selection):

                match = re.match(dx_pattern, dx_object)
                channel_name = match.group(2)
                channel = channels_by_name[channel_name]
                element = atom_types.get_name(channel.atomic_num)

                surface_object = '{}_{}_surface'.format(surface_prefix, channel_name)
                cmd.isosurface(surface_object, dx_object, level=level, state=state_)

                cmd.color(channel.name+'$', surface_object)

                if surface_prefix not in surface_groups:
                    surface_groups[surface_prefix] = []

                if surface_object not in surface_groups[surface_prefix]:
                    surface_groups[surface_prefix].append(surface_object)

    for surface_prefix in surface_groups:
        surface_group = '{}_surface'.format(surface_prefix)
        cmd.group(surface_group, ' '.join(surface_groups[surface_prefix]))


def load_group(pattern, name):
    group_objs = []
    for file in glob.glob(pattern):
        obj = os.path.basename(file)
        cmd.load(file, obj)
        group_objs.append(obj)
    print(group_objs)
    if group_objs:
        cmd.group(name, ' '.join(group_objs))


def my_rotate(name, axis, angle, states):
    for i in range(int(states)):
        cmd.create(name, name, 1, i+1)
        cmd.rotate(axis, float(angle)*i/int(states), name, state=i+1, camera=0)


cmd.extend('set_atom_level', set_atom_level)
cmd.extend('load_group', load_group)
cmd.extend('my_rotate', my_rotate)
