from __future__ import print_function
import sys
import os
import numpy as np
cimport numpy as np
from scipy.stats import multivariate_normal


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def fit_atoms_to_points_and_density(np.ndarray points, np.ndarray density, np.ndarray atom_mean_init,
                                    DTYPE_t atom_radius, char* noise_type, dict noise_params_init, int max_iter):
    assert points.shape[0] == density.shape[0]
    assert points.shape[1] == 3
    assert atom_mean_init.shape[1] == 3

    cdef int n_points = points.shape[0]
    cdef int n_atoms = atom_mean_init.shape[0]

    # initialize atom component parameters
    cdef np.ndarray[DTYPE_t, ndim=2] atom_mean = np.array(atom_mean_init, dtype=DTYPE)
    cdef DTYPE_t atom_cov = (0.5*atom_radius)**2
    cdef int n_params = atom_mean.size
    cdef int n_comps = n_atoms

    # initialize noise component parameters
    cdef DTYPE_t noise_mean
    cdef DTYPE_t noise_cov
    cdef DTYPE_t noise_prob
    if noise_type[0] == 'd':
        noise_mean = noise_params_init['mean']
        noise_cov = noise_params_init['cov']
        n_params += 2
        n_comps += 1
    elif noise_type[0] == 'p':
        noise_prob = noise_params_init['prob']
        n_params += 1
        n_comps += 1
    elif len(noise_type) > 0:
        raise TypeError("noise_type must be 'd' or 'p', or '', got {}".format(repr(noise_type)))

    # initialize prior over components
    cdef np.ndarray[DTYPE_t, ndim=1] P_comp = np.full(n_comps, 1.0/n_comps, dtype=DTYPE) # P(comp_j)
    n_params += n_comps - 1

    cdef np.ndarray[DTYPE_t, ndim=2] L_point = np.zeros((n_points, n_comps)) # P(point_i|comp_j)
    cdef np.ndarray[DTYPE_t, ndim=2] P_joint = np.zeros((n_points, n_comps)) # P(point_i, comp_j)
    cdef np.ndarray[DTYPE_t, ndim=1] P_point = np.zeros(n_points)            # P(point_i)
    cdef np.ndarray[DTYPE_t, ndim=2] gamma = np.zeros((n_points, n_comps))   # P(point_i|comp_j)

    # maximize expected log likelihood
    cdef DTYPE_t ll = -np.inf
    cdef DTYPE_t ll_prev = -np.inf
    for i in range(max_iter+1):

        for j in range(n_atoms):
            L_point[:,j] = multivariate_normal.pdf(points, mean=atom_mean[j], cov=atom_cov)
        if noise_type[0] == 'd':
            L_point[:,-1] = multivariate_normal.pdf(density, mean=noise_mean, cov=noise_cov)
        elif noise_type[0] == 'p':
            L_point[:,-1] = noise_prob

        P_joint = P_comp * L_point
        P_point = np.sum(P_joint, axis=1)
        gamma = (P_joint.T / P_point).T     # (E-step)

        # compute expected log likelihood
        ll_prev = ll
        ll = np.sum(density * np.log(P_point))
        if ll - ll_prev < 1e-8 or i == max_iter:
            break

        # estimate parameters that maximize expected log likelihood (M-step)
        for j in range(n_atoms):
            g_d = density * gamma[:,j]
            atom_mean[j] = np.sum(g_d * points.T, axis=1) / np.sum(g_d)
        if noise_type[0] == 'd':
            sum_g = np.sum(gamma[:,-1])
            noise_mean = np.sum(gamma[:,-1] * density) / sum_g
            noise_cov = np.sum(gamma[:,-1] * (density - noise_mean)**2) / sum_g
            if noise_cov == 0.0 or np.isnan(noise_cov): # reset noise
                noise_mean = noise_params_init['mean']
                noise_cov = noise_params_init['cov']
        elif noise_type[0] == 'p':
            pass
        if noise_type and n_atoms > 0:
            P_comp[-1] = np.sum(density * gamma[:,-1]) / np.sum(density)
            P_comp[:-1] = (1.0 - P_comp[-1])/n_atoms

    return atom_mean, 2*ll - 2*n_params
