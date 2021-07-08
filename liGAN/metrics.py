import torch
import numpy as np
import scipy as sp
import scipy.optimize
from collections import OrderedDict


def compute_scalar_metrics(scalar_type, scalars):
    m = OrderedDict()
    m[scalar_type+'_mean'] = scalars.mean().item()
    m[scalar_type+'_variance'] = scalars.var(unbiased=False).item()
    return m


def compute_mean_grid_norm(grids):
    dim = tuple(range(1, grids.ndim))
    return grids.detach().norm(p=2, dim=dim).mean().item()


def compute_grid_variance(grids):
    mean_grid = grids.detach().mean(dim=0)
    return (((grids.detach() - mean_grid)**2).sum() / grids.shape[0]).item()


def compute_grid_metrics(grid_type, grids):
    m = OrderedDict()
    m[grid_type+'_norm'] = compute_mean_grid_norm(grids)
    m[grid_type+'_variance'] = compute_grid_variance(grids)
    return m


# TODO this is also defined/computed in training, how to consolidate?
def compute_L2_loss(grids, ref_grids):
    return (
        (ref_grids.detach() - grids.detach())**2
    ).sum().item() / 2 / grids.shape[0]


def compute_paired_grid_metrics(grid_type, grids, ref_grid_type, ref_grids):
    m = compute_grid_metrics(ref_grid_type, ref_grids)
    m.update(compute_grid_metrics(grid_type, grids))
    m[grid_type+'_L2_loss'] = compute_L2_loss(grids, ref_grids)
    return m


def compute_mean_n_atoms(structs):
    return np.mean([s.n_atoms for s in structs])


def compute_n_atoms_variance(structs):
    m = np.mean([s.n_atoms for s in structs])
    return np.mean([(s.n_atoms-m)**2 for s in structs])


def compute_mean_radius(structs):
    return np.mean([s.radius for s in structs])


def compute_type_variance(structs, which=None):

    if which is None:
        type_counts = [s.type_counts for s in structs]

    elif which == 'elem':
        type_counts = [s.elem_counts for s in structs]

    elif which == 'prop':
        type_counts = [s.prop_counts for s in structs]

    m = torch.stack(type_counts).mean(dim=0)
    return np.mean([
        (t-m).norm(p=1).item() for t in type_counts
    ])


def compute_struct_metrics(struct_type, structs):
    m = OrderedDict()
    m[struct_type+'_n_atoms'] = compute_mean_n_atoms(structs)
    m[struct_type+'_n_atoms_variance'] = compute_n_atoms_variance(structs)
    m[struct_type+'_radius'] = compute_mean_radius(structs)
    m[struct_type+'_type_variance'] = compute_type_variance(structs)
    m[struct_type+'_elem_variance'] = compute_type_variance(
        structs, which='elem'
    )
    m[struct_type+'_prop_variance'] = compute_type_variance(
        structs, which='prop'
    )
    return m


def compute_mean_type_diff(structs, ref_structs, which=None):
    if which is None:
        type_counts = [s.type_counts for s in structs]
        ref_type_counts = [s.type_counts for s in ref_structs]
    elif which == 'elem':
        type_counts = [s.elem_counts for s in structs]
        ref_type_counts = [s.elem_counts for s in ref_structs]
    elif which == 'prop':
        type_counts = [s.prop_counts for s in structs]
        ref_type_counts = [s.prop_counts for s in ref_structs]
    type_diffs = np.array([
        (t-r).norm(p=1).item() for t,r in zip(type_counts, ref_type_counts)
    ])
    return np.mean(type_diffs), np.mean(type_diffs == 0)


def compute_min_rmsd(coords1, types1, coords2, types2):
    '''
    Compute an RMSD between two sets of positions of the same
    atom types with no prior mapping between particular atom
    positions of a given type. Returns the minimum RMSD across
    all permutations of this mapping.
    '''
    # check that structs are same size
    n1, n2 = len(coords1), len(coords2)
    assert n1 == n2, \
        'structs must have same num atoms ({} vs. {})'.format(n1, n2)
    n_atoms = len(coords1)

    # copy everything into arrays
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    types1 = np.array(types1)
    types2 = np.array(types2)

    # check that atom types are compatible
    # CAUTION this may not be sufficient for vector types
    #   we could have two structs with the same element
    #   counts and property counts, but the properties
    #   could be on different atoms/elements- do we care?
    type_counts1 = types1.sum(axis=0)
    type_counts2 = types2.sum(axis=0)
    type_diff = np.abs(type_counts1 - type_counts2).sum()
    assert (type_counts1 == type_counts2).all(), \
        'structs must have same type counts ({:.2f})'.format(type_diff)

    # find min rmsd by solving linear sum assignment
    # problem on squared dist matrix for each type
    ssd = 0.0
    nax = np.newaxis
    for t in np.unique(types1, axis=0):
        coords1_t = coords1[(types1 == t[nax,:]).all(axis=1)]
        coords2_t = coords2[(types2 == t[nax,:]).all(axis=1)]
        assert len(coords1_t) == len(coords2_t), \
            'structs must have same num atoms of each type'
        dist2_t = ((coords1_t[:,nax,:] - coords2_t[nax,:,:])**2).sum(axis=2)
        idx1, idx2 = sp.optimize.linear_sum_assignment(dist2_t)
        ssd += dist2_t[idx1, idx2].sum()

    return np.sqrt(ssd/n_atoms)


def compute_struct_rmsd(struct1, struct2, catch_exc=True):
    assert struct1.typer == struct2.typer, 'structs have different typers'
    n_elem_types = struct1.typer.n_elem_types
    try:
        return compute_min_rmsd( # ignore property channels
            struct1.coords.cpu(), struct1.types[:,:n_elem_types].cpu(),
            struct2.coords.cpu(), struct2.types[:,:n_elem_types].cpu(),
        )
    except (AssertionError, ZeroDivisionError):
        if catch_exc:
            return np.nan
        raise


def compute_mean_atom_rmsd(structs, ref_structs):
    atom_rmsds = [
        compute_struct_rmsd(s, r) for s, r in zip(
            structs, ref_structs
        )
    ]
    return np.mean(atom_rmsds)


def compute_paired_struct_metrics(
    struct_type,
    structs,
    ref_struct_type,
    ref_structs
):
    m = compute_struct_metrics(struct_type, structs)
    m.update(compute_struct_metrics(ref_struct_type, ref_structs))
    m[struct_type+'_type_diff'], m[struct_type+'_exact_types'] = \
        compute_mean_type_diff(structs, ref_structs)
    m[struct_type+'_elem_diff'], m[struct_type+'_exact_elems'] = \
        compute_mean_type_diff(structs, ref_structs, which='elem')
    m[struct_type+'_prop_diff'], m[struct_type+'_exact_props'] = \
        compute_mean_type_diff(structs, ref_structs, which='prop')
    m[struct_type+'_atom_rmsd'] = compute_mean_atom_rmsd(structs, ref_structs)
    return m
