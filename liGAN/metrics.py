import numpy as np
import scipy as sp
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


def compute_mean_radius(structs):
    return np.mean([s.radius for s in structs])


def compute_struct_metrics(struct_type, structs):
    m = OrderedDict()
    m[struct_type+'_n_atoms'] = compute_mean_n_atoms(structs)
    m[struct_type+'_radius'] = compute_mean_radius(structs)
    return m


def compute_mean_type_diff_and_exact_types(structs, ref_structs):
    type_counts = [s.type_counts for s in structs]
    ref_type_counts = [s.type_counts for s in ref_structs]
    type_diffs = np.array([
        np.linalg.norm(t-r, ord=1) for t,r in zip(
            type_counts, ref_type_counts
        )
    ])
    return np.mean(type_diffs), np.mean(type_diffs == 0)


def compute_min_rmsd(xyz1, c1, xyz2, c2):
    '''
    Compute an RMSD between two sets of positions of the same
    atom types with no prior mapping between particular atom
    positions of a given type. Returns the minimum RMSD across
    all permutations of this mapping.
    '''
    # check that structs are same size
    if len(c1) != len(c2):
        raise ValueError('structs must have same num atoms')
    n_atoms = len(c1)

    # copy everything into arrays
    xyz1 = np.array(xyz1)
    xyz2 = np.array(xyz2)
    c1 = np.array(c1)
    c2 = np.array(c2)

    # check that types are compatible
    idx1 = np.argsort(c1)
    idx2 = np.argsort(c2)
    c1 = c1[idx1]
    c2 = c2[idx2]
    if any(c1 != c2):
        raise ValueError('structs must have same num atoms of each type')
    xyz1 = xyz1[idx1]
    xyz2 = xyz2[idx2]

    # find min rmsd by solving linear sum assignment
    # problem on squared dist matrix for each type
    ssd = 0.0
    nax = np.newaxis
    for c in set(c1): 
        xyz1_c = xyz1[c1 == c]
        xyz2_c = xyz2[c2 == c]
        dist2_c = ((xyz1_c[:,nax,:] - xyz2_c[nax,:,:])**2).sum(axis=2)
        idx1, idx2 = sp.optimize.linear_sum_assignment(dist2_c)
        ssd += dist2_c[idx1, idx2].sum()

    return np.sqrt(ssd/n_atoms)


def compute_atom_rmsd(struct1, struct2):
    try:
        return compute_min_rmsd(
            struct1.xyz, struct1.c, struct2.xyz, struct2.c
        )
    except (ValueError, ZeroDivisionError):
        return np.nan


def compute_mean_atom_rmsd(structs, ref_structs):
    atom_rmsds = [
        compute_atom_rmsd(s, r) for s,r in zip(
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
        compute_mean_type_diff_and_exact_types(structs, ref_structs)
    m[struct_type+'_atom_rmsd'] = compute_mean_atom_rmsd(structs, ref_structs)
    return m
