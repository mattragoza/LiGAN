import sys, re, glob, fnmatch, shutil, time
from collections import OrderedDict
import pymol
from pymol import cmd, stored

sys.path.insert(0, '.')
import isoslider
import atom_types


def get_common_prefix(strs):
    for i, chars in enumerate(zip(*strs)):
        if len(set(chars)) > 1:
            return strs[0][:i]


def get_color(grid_type, atomic_num):
    '''
    Color atomic density isosurfaces by
    element, using different colors for
    carbon depending on the grid_type.
    '''
    if atomic_num == 6:
        if grid_type == 'rec':
            return [0.8, 0.8, 0.8] # gray
        elif grid_type == 'lig_gen':
            return [1.0, 0.0, 1.0] # magenta
        elif grid_type.endswith('fit'):
            return [0.0, 1.0, 0.0] # green
        else:
            return [0.0, 1.0, 1.0] # cyan
    else:
        return atom_types.get_rgb(atomic_num)


def as_bool(s):
    if s in {'True', 'true', 'T', 't', '1'}:
        return True
    elif s in {'False', 'false', 'F', 'f', '0'}:
        return False
    else:
        return bool(s)


def set_atom_level(
    level,
    selection='*',
    state=None,
    rec_map=None,
    lig_map=None,
    interp=False,
    job_name=None,
    array_job=False,
):
    interp = as_bool(interp)
    array_job = as_bool(array_job)

    # get atom type channel info and set custom colors
    try:
        rec_channels = atom_types.get_channels_from_file(rec_map)
    except:
        rec_channels = atom_types.get_default_rec_channels()
    try:
        lig_channels = atom_types.get_channels_from_file(lig_map)
    except:
        lig_channels = atom_types.get_default_lig_channels()

    channels = rec_channels + lig_channels
    channels_by_name = {ch.name: ch for ch in channels}
    channel_names = channels_by_name.keys()

    for ch in channels:
        for grid_type in ['rec', 'lig', 'lig_gen', 'lig_fit', 'lig_gen_fit']:
            color_name = grid_type + '_' + ch.name
            cmd.set_color(color_name+'$', get_color(grid_type, ch.atomic_num))

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
        r'(?P<grid_type>rec|lig(_gen)?(_conv|_fit)?)',
        r'(?P<sample_idx>\d+)',
        r'(?P<channel_name>{})'.format('|'.join(channel_names)),
    ]
    grid_re = re.compile('^' + '_'.join(grid_re_fields) + r'(\.dx|_grid)$')

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

        channel = channels_by_name[channel_name]
        cmd.isosurface(surface, grid, level=level, state=s)
        cmd.color(grid_type + '_' + channel.name+'$', surface)
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


def mutate(protein, resi, mode='ALA', chain='A'):
    cmd.wizard('mutagenesis')
    cmd.do("refresh_wizard")
    cmd.get_wizard().set_mode(mode.upper())
    selection = f'/{protein}//{chain}/{resi}'
    cmd.get_wizard().do_select(selection)
    cmd.frame(str(1))
    cmd.get_wizard().apply()
    cmd.set_wizard('done')
    cmd.refresh()


def my_mutate(rec_name, lig_name, dist):

    # select residues in rec_name within dist of lig_name
    #   and show them as lines, the rest as cartoon
    rec_pocket = f'byres {rec_name} within {dist} of {lig_name}'

    # get the names and indices of the pocket residues
    space = dict(residues=set())
    cmd.iterate(rec_pocket, 'residues.add((resv, resn))', space=space)
    residues = sorted(space['residues'])
    res_idxs = [str(i) for i, n in residues]
    res_idxs = '(resi ' + '+'.join(res_idxs) + ')'

    # create a mapping from charged residues to
    #   oppositely charged residues of similar size
    charge_map = {
        'ARG': 'GLU',
        'HIS': 'ASP',
        'LYS': 'GLU',
        'ASP': 'LYS',
        'GLU': 'LYS',
    }

    # mutate each pocket residue individually
    for res_idx, res_name in residues:

        # make a mutant with the residue as alanine
        mut_name = f'{rec_name}_mut_{res_idx}_{res_name}_ALA'
        cmd.copy(mut_name, rec_name)
        mutate(mut_name, res_idx, 'ALA')
        cmd.save(mut_name + '.pdb', mut_name)

        if res_name in charge_map:

            # make another mutant with the charge flipped
            inv_name = charge_map[res_name]
            mut_name = f'{rec_name}_mut_{res_idx}_{res_name}_{inv_name}'
            cmd.copy(mut_name, rec_name)
            mutate(mut_name, res_idx, inv_name)
            cmd.save(mut_name + '.pdb', mut_name)

    # create one final mutant with ALL charges flipped
    mut_name = f'{rec_name}_mut_all_charges'
    cmd.copy(mut_name, rec_name)
    for res_idx, res_name in residues:
        if res_name in charge_map:
            inv_name = charge_map[res_name]
            mutate(mut_name, res_idx, inv_name)

    cmd.save(mut_name + '.pdb', mut_name)
    cmd.zoom(rec_pocket)
    print(res_idxs)

cmd.extend('set_atom_level', set_atom_level)
cmd.extend('join_structs', join_structs)
cmd.extend('draw_interp', draw_interp)
cmd.extend('load_group', load_group)
cmd.extend('my_rotate', my_rotate)
cmd.extend('mutate', mutate)
cmd.extend('my_mutate', my_mutate)
