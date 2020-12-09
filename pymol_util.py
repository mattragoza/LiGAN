import sys, re, glob, fnmatch, shutil
from collections import OrderedDict
import pymol
from pymol import cmd, stored

sys.path.insert(0, '.')
import isoslider
import atom_types


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


def get_channel_color(channel):
    if 'LigandAliphatic' in channel.name:
        return [0.83, 0.83, 0.83] #[1.00, 0.50, 1.00]
    elif 'LigandAromatic' in channel.name:
        return [0.67, 0.67, 0.67] #[1.00, 0.00, 1.00]
    elif 'ReceptorAliphatic' in channel.name:
        return [1.00, 1.00, 1.00]
    elif 'ReceptorAromatic' in channel.name:
        return [0.83, 0.83, 0.83]
    else:
        return get_rgb(channel.atomic_num)


def get_common_prefix(strs):
    for i, chars in enumerate(zip(*strs)):
        if len(set(chars)) > 1:
            return strs[0][:i]


def set_atom_level(
    level,
    selection='*',
    state=None,
    rec_map=None,
    lig_map=None,
    interp=False,
    job_name=None,
    array_job=True,
):

    # get atom type channel info and set custom colors
    try:
        rec_channels = atom_types.get_channels_from_file(rec_map, 'Receptor')
    except:
        rec_channels = atom_types.get_default_rec_channels()
    try:
        lig_channels = atom_types.get_channels_from_file(lig_map, 'Ligand')
    except:
        lig_channels = atom_types.get_default_lig_channels()

    channels = rec_channels + lig_channels
    channels_by_name = {ch.name: ch for ch in channels}

    for ch in channels:
        cmd.set_color(ch.name+'$', get_channel_color(ch))

    # We sort grids based on two different criteria before creating surfaces
    # for each grid. The first criteria determines what STATE to put the
    # surface in, the second determines what GROUP to put it in.

    # We expect grid names to store info in the following basic format:
    #   {job_name}_{lig_name}_{grid_type}_{sample_idx}_{channel_name}

    # By default,
    #   group=(job_name, lig_name, grid_type)
    #   state=(sample_idx,)

    # With interp=True,
    #   group=(job_name, grid_type)
    #   state=(lig_name, sample_idx)

    # get list of selected grid objects
    grids = []

    for obj in cmd.get_names('objects'):
        m = re.match(r'^(.*)(\.dx|_grid)$', obj)
        if m and fnmatch.fnmatch(obj, selection):
            grids.append(obj)

    # try to infer the job_name from working directory
    if job_name is None:
        job_name = os.path.basename(os.getcwd())

    # create a regex for parsing grid names
    grid_re_fields = [r'(?P<job_name>{})'.format(job_name)]

    if array_job:
        grid_re_fields += [r'(?P<array_idx>\d+)']

    grid_re_fields += [
        r'(?P<lig_name>.+)',
        r'(?P<grid_type>lig(_gen)?(_conv|_fit)?)',
        r'(?P<sample_idx>\d+)',
        r'(?P<channel_name>{})'.format('|'.join(channels_by_name)),
    ]

    grid_re = re.compile(
        '^' + '_'.join(grid_re_fields) + r'(\.dx|_grid)$'
    )

    # assign grids to groups and states
    grouped_surfaces = OrderedDict()
    grouped_lig_names = OrderedDict()
    state_counter = OrderedDict()

    for i, grid in enumerate(grids):

        m = grid_re.match(grid)
        try:
            lig_name = m.group('lig_name')
            grid_type = m.group('grid_type')
            sample_idx = int(m.group('sample_idx'))
            channel_name = m.group('channel_name')
        except:
            print(grid_re)
            print(grid)
            raise

        if interp:
            group_criteria = (job_name, grid_type)
            state_criteria = (lig_name, sample_idx)
            surface_format = '{job_name}_{grid_type}_{channel_name}_surface'
        else:
            group_criteria = (job_name, lig_name, grid_type)
            state_criteria = sample_idx
            surface_format = '{job_name}_{lig_name}_{grid_type}_{channel_name}_surface'

        surface = surface_format.format(**m.groupdict())

        if state_criteria not in state_counter:
            state_counter[state_criteria] = len(state_counter) + 1
        s = state_counter[state_criteria]

        if group_criteria not in grouped_surfaces:
            grouped_surfaces[group_criteria] = []
            grouped_lig_names[group_criteria] = []

        if surface not in grouped_surfaces[group_criteria]:
            grouped_surfaces[group_criteria].append(surface)

        if lig_name not in grouped_lig_names[group_criteria]:
            grouped_lig_names[group_criteria].append(lig_name)

        cmd.isosurface(surface, grid, level=level, state=s)
        cmd.color(channel_name+'$', surface)
        print('[{}/{}] {}'.format(i+1, len(grids), surface, group_criteria, s))

    for group_criteria, surfaces in grouped_surfaces.items():
        if interp:
            job_name, grid_type = group_criteria
            lig_names = grouped_lig_names[group_criteria]
            surface_group = '_'.join(
                (job_name, '_to_'.join(lig_names), grid_type)
            ) + '_surfaces'
        else:
            surface_group = '_'.join(group_criteria) + '_surfaces'
        cmd.group(surface_group, ' '.join(surfaces))

    print('Done')


def join_structs(
    selection='*',
    job_name=None,
    array_job=False,
    interp=False,
    delete=False,
):
    '''
    join_structs selection=*, job_name=None, array_job=False, interp=False, delete=False
    '''
    # try to infer the job_name from working directory
    if job_name is None:
        job_name = os.path.basename(os.getcwd())

    # create regex to parse struct names
    struct_re_fields = [r'(?P<job_name>{})'.format(job_name)]

    if array_job:
        struct_re_fields += [r'(?P<array_idx>\d+)']

    struct_re_fields += [
        r'(?P<lig_name>.+)',
        r'(?P<struct_type>lig(_gen)?(_fit)?(_add|_src)?(_uff)?)',
        r'(?P<sample_idx>\d+)',
    ]

    struct_pat = '^' + '_'.join(struct_re_fields) + '$'
    print(struct_pat)
    struct_re = re.compile(struct_pat)

    # keep track of which structs to join and their lig_names
    structs_to_join = OrderedDict()
    lig_names_to_join = OrderedDict()

    for obj in cmd.get_names('objects'):

        if not fnmatch.fnmatch(obj, selection):
            continue

        m = struct_re.match(obj)
        if not m:
            continue
        struct = obj

        lig_name = m.group('lig_name')
        struct_type = m.group('struct_type')
        sample_idx = int(m.group('sample_idx'))

        if interp: # join different lig_names
            join_criteria = (job_name, struct_type)
        else:
            join_criteria = (job_name, lig_name, struct_type)

        if join_criteria not in structs_to_join:
            structs_to_join[join_criteria] = []
            lig_names_to_join[join_criteria] = []

        if lig_name not in lig_names_to_join[join_criteria]:
            lig_names_to_join[join_criteria].append(lig_name)

        if struct not in structs_to_join[join_criteria]:
            lig_idx = lig_names_to_join[join_criteria].index(lig_name)
            structs_to_join[join_criteria].append(
                (lig_idx, sample_idx, struct)
            )

    for join_criteria, structs in structs_to_join.items():
        print(join_criteria)
        structs = [struct for _, _, struct in sorted(structs)]
        lig_names = lig_names_to_join[join_criteria]
        if interp:
            job_name, struct_type = join_criteria
        else:
            job_name, lig_name, struct_type = join_criteria
        joined_struct = '_'.join(
            (job_name, '_to_'.join(lig_names), struct_type)
        )
        cmd.join_states(joined_struct, ' '.join(structs), 0)
        if delete:
            cmd.delete(' '.join(structs))

    print('Done')


def draw_interp(out_dir, selection='*', width=1000, height=1000, antialias=2, dpi=-1):

    structs = []
    interp_prefixes = []
    for obj in cmd.get_names('objects'):
        m = re.match(r'^.*lig(_gen)?(_fit)?(_add)?$', obj)
        if m:
            structs.append(obj)
        m = re.match(r'^(.*_lig_gen)_surfaces$', obj)
        if m and fnmatch.fnmatch(obj, selection):
            interp_prefixes.append(m.group(1))

    # try to infer the job_name from working directory
    job_name = os.path.basename(os.getcwd())

    # once we have a job_name, we can correctly parse the prefixes
    interp_re = re.compile(
        '^' + '_'.join([
            r'(?P<job_name>{})'.format(job_name),
            r'(?P<array_idx>\d+)',
            r'(?P<interp_name>(.+)_to_(.+))',
            r'(?P<grid_type>lig(_gen)?(_fit)?)'
        ]) + '$'
    )

    for interp_prefix in interp_prefixes:
        
        m = interp_re.match(interp_prefix)
        if not m:
            continue

        interp_name = m.group('interp_name')
        interp_dir = os.path.join(out_dir, job_name, interp_name)
        os.makedirs(interp_dir, exist_ok=True)

        density = interp_prefix + '_surfaces'
        struct = interp_prefix + '_fit_add'

        # hide everything except the grouped surface objects,
        # since their visibility is controlled by the group object
        cmd.disable('all')
        cmd.enable('*_surface')

        # density only
        cmd.enable(density)
        cmd.set('transparency', 0.2, density)
        im_prefix = os.path.join(interp_dir, interp_prefix) + '_density_'
        cmd.mpng(im_prefix, first=0, last=0, mode=1, width=width, height=height)

        # density and struct
        cmd.enable(struct)
        cmd.set('transparency', 0.5, density)
        im_prefix = os.path.join(interp_dir, interp_prefix) + '_both_'
        cmd.mpng(im_prefix, first=0, last=0, mode=1, width=width, height=height)

        cmd.disable(density)
        im_prefix = os.path.join(interp_dir, interp_prefix) + '_struct_'
        cmd.mpng(im_prefix, first=0, last=0, mode=1, width=width, height=height)

    print('Done')


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
cmd.extend('join_structs', join_structs)
cmd.extend('draw_interp', draw_interp)
cmd.extend('load_group', load_group)
cmd.extend('my_rotate', my_rotate)
