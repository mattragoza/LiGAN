import time
import numpy as np
import torch
import torch.nn.functional as F
import molgrid

from .atom_grids import AtomGrid, spatial_index_to_coords
from .atom_structs import AtomStruct


class AtomFitter(object):
    '''
    An algorithm for fitting atoms to density grids using
    beam search, atom detection, and gradient descent.

    Runs a beam search over structures of atom types and
    coordinates where beam_size current best structures
    are stored and expanded at each step. The objective
    function is L2 loss of the density of the fit struct-
    ure to the reference density.

    Structures are expanded by detecting atoms in the rem-
    aining density after subtracting the density of the
    structure from the reference density.

    Gradient descent is performed after adding atoms to
    structures. If the resulting structure has lower loss
    than any of the current best structures, it is stored,
    otherwise that branch of the search is terminated.
    '''
    def __init__(
        self,
        beam_size=1,
        multi_atom=False,
        n_atoms_detect=1,
        apply_conv=False,
        threshold=0.1,
        peak_value=1.5,
        min_dist=0.0,
        constrain_types=False,
        constrain_frags=False,
        estimate_types=False,
        fit_L1_loss=False,
        interm_gd_iters=10,
        final_gd_iters=100,
        gd_kwargs=dict(
            lr=0.1,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        ),
        dkoes_make_mol=True,
        use_openbabel=False,
        output_kernel=False,
        device='cuda',
        verbose=0,
        debug=False,
    ):
        # number of best structures to store and expand during search
        self.beam_size = beam_size

        # maximum number of atoms to detect in remaining density
        self.n_atoms_detect = n_atoms_detect

        # try placing all detected atoms at once, then try individually
        self.multi_atom = multi_atom

        # other settings for detecting atoms in remaining density
        self.apply_conv = apply_conv
        self.threshold = threshold
        self.peak_value = peak_value
        self.min_dist = min_dist

        # can constrain to find exact atom type counts or single fragment
        self.constrain_types = constrain_types
        self.constrain_frags = constrain_frags
        self.estimate_types = estimate_types

        # can perform gradient descent at each step and/or at final step
        self.fit_L1_loss = fit_L1_loss
        self.interm_gd_iters = interm_gd_iters
        self.final_gd_iters = final_gd_iters
        self.gd_kwargs = gd_kwargs

        self.output_kernel = output_kernel
        self.device = device
        self.verbose = verbose
        self.debug = debug

        self.grid_maker = molgrid.GridMaker()
        self.c2grid = molgrid.Coords2Grid(self.grid_maker)
        self.kernel = None

    def init_kernel(self, resolution, typer, deconv=False):
        '''
        Initialize an atomic density kernel that can
        can be used to detect atoms in density grids.
        The kernel has a different channel for each
        element in the typer's element range.
        '''
        n_channels = typer.n_elem_types

        # kernel is created by computing a molgrid from a
        # struct with one atom of each element at the center
        coords = torch.zeros(
            (n_channels, 3),
            dtype=torch.float32,
            device=self.device
        )
        types = torch.eye(
            n_channels,
            dtype=torch.float32,
            device=self.device
        )
        radii = torch.tensor(
            [typer.radius_func(n) for n in typer.elem_range], 
            dtype=torch.float32,
            device=self.device
        )

        # now set the grid settings
        self.c2grid.center = (0, 0, 0)
        self.grid_maker.set_resolution(resolution)

        # this flag indicates that each atom has its own radius
        #   which may or may not be needed here...
        self.grid_maker.set_radii_type_indexed(False)

        # kernel must be large enough for atom with largest radius
        kernel_radius = 1.5 * max(radii).item()
        self.grid_maker.set_dimension(2 * kernel_radius)

        # kernel must also have odd spatial dimension
        #   so that convolution produces same size output grid
        if self.grid_maker.spatial_grid_dimensions()[0] % 2 == 0:
            self.grid_maker.set_dimension(
                self.grid_maker.get_dimension() + resolution
            )

        # create the kernel
        self.kernel = self.c2grid(coords, types, radii)

        if deconv: # invert the kernel
            self.kernel = torch.tensor(
                weiner_invert_kernel(self.kernel.cpu(), noise_ratio=1),
                dtype=self.kernel.dtype,
                device=self.device,
            )

        if self.output_kernel: # write out the kernel
            dx_prefix = 'deconv_kernel' if deconv else 'conv_kernel'
            if self.verbose:
                kernel_norm = self.kernel.norm().item()
                print(
                    'writing out {} (norm={})'.format(dx_prefix, kernel_norm)
                )
            self.kernel.to_dx(dx_prefix)
            self.output_kernel = False # only write once

    def convolve(self, elem_values, resolution, typer):
        '''
        Compute a convolution between the provided
        density grid and the atomic density kernel.

        The output is normalized by the kernel norm
        so that values above 0.5 indicate grid points
        where placing an atom would decrease L2 loss.
        '''
        if self.kernel is None:
            self.init_kernel(resolution, typer)

        # normalize convolved grid channels by kernel norm
        kernel_norm2 = (self.kernel**2).sum(dim=(1,2,3), keepdim=True)

        return F.conv3d(
            input=elem_values.unsqueeze(0),
            weight=self.kernel.unsqueeze(1),
            padding=self.kernel.shape[-1]//2,
            groups=typer.n_elem_types,
        )[0] / kernel_norm2

    def apply_peak_value(self, elem_values):
        return self.peak_value - (self.peak_value - elem_values).abs()

    def sort_grid_points(self, grid_values):
        n_c, n_x, n_y, n_z  = grid_values.shape

        # get flattened grid values and index, sorted by value
        values, idx = torch.sort(grid_values.flatten(), descending=True)

        # convert flattened grid index to channel and spatial index
        idx_z, idx = idx % n_z, idx // n_z
        idx_y, idx = idx % n_y, idx // n_y
        idx_x, idx = idx % n_x, idx // n_x
        idx_c, idx = idx % n_c, idx // n_c
        idx_xyz = torch.stack((idx_x, idx_y, idx_z), dim=1)

        return values, idx_xyz, idx_c

    def apply_threshold(self, values, idx_xyz, idx_c):
        above_thresh = values > self.threshold
        values = values[above_thresh]
        idx_xyz = idx_xyz[above_thresh]
        idx_c = idx_c[above_thresh]
        return values, idx_xyz, idx_c

    def apply_type_constraint(self, values, idx_xyz, idx_c, type_counts):

        #TODO this does not constrain the atoms types correctly
        # when doing multi_atom fitting, because it only omits
        # atom types that have 0 atoms left- i.e. we could still
        # return 2 atoms of a type that only has 1 atom left.
        # Need to exclude all atoms of type t beyond rank n_t
        # where n_t is the number of atoms left of type t
        has_atoms_left = type_counts[idx_c] > 0
        values = values[has_atoms_left]
        idx_xyz = idx_xyz[has_atoms_left]
        idx_c = idx_c[has_atoms_left]
        return values, idx_xyz, idx_c

    def suppress_non_max(self, values, coords, idx_c, grid, matrix=None):

        r = torch.tensor( # atomic radii
            [grid.typer.radius_func(v) for v in grid.typer.elem_range],
            device=self.device,
        )
        if matrix or (matrix is None and len(coords) < 1000): # use NxN matrices

            same_type = (idx_c.unsqueeze(1) == idx_c.unsqueeze(0))
            bond_radius = r[idx_c].unsqueeze(1) + r[idx_c].unsqueeze(0)
            min_dist2 = (self.min_dist * bond_radius)**2
            dist2 = ((coords.unsqueeze(1)-coords.unsqueeze(0))**2).sum(dim=2)

            # the lower triangular part of a matrix under diagonal -1
            #   gives those indices i,j such that i > j
            # since atoms are sorted by decreasing density value,
            #   i > j implies that atom i has lower value than atom j
            # we use this to check a condition on each atom
            #   only with respect to atoms of higher value 
            too_close = torch.tril(
                (dist2 < min_dist2) & same_type, diagonal=-1
            ).any(dim=1)
            
            return coords[~too_close], idx_c[~too_close]

        else: # use a for-loop

            coords_max = coords[0].unsqueeze(0)
            idx_c_max = idx_c[0].unsqueeze(0)

            for i in range(1, len(idx_c)):
                same_type = (idx_c[i] == idx_c_max)
                bond_radius = r[idx_c[i]] + r[idx_c_max]
                min_dist2 = (self.min_dist * bond_radius)**2
                dist2 = ((coords[i].unsqueeze(0) - coords_max)**2).sum(dim=1)
                if not ((dist2 < min_dist2) & same_type).any():
                    coords_max = torch.cat([coords_max,coords[i].unsqueeze(0)])
                    idx_c_max = torch.cat([idx_c_max, idx_c[i].unsqueeze(0)])

            return coords_max, idx_c_max

    def detect_atoms(self, grid, type_counts=None):
        '''
        Detect a set of typed atoms in an AtomGrid by convolving
        with a kernel, applying a threshold, and then returning
        atom coordinates and type vectors ordered by grid value.
        '''
        not_none = lambda x: x is not None

        # detect atoms in the element channels
        values = grid.elem_values

        # convolve grid with atomic density kernel
        if self.apply_conv:
            values = self.convolve(values, grid.resolution, grid.typer)

        # reflect grid values above peak value
        if not_none(self.peak_value) and self.peak_value < np.inf:
            values = self.apply_peak_value(values)

        # sort grid points by value
        values, idx_xyz, idx_c = self.sort_grid_points(values)

        # apply threshold to grid points and values
        if not_none(self.threshold) and self.threshold > -np.inf:
            values, idx_xyz, idx_c = self.apply_threshold(
                values, idx_xyz, idx_c
            )

        # exclude grid channels with no atoms left
        if self.constrain_types:
            values, idx_xyz, idx_c = self.apply_type_constraint(
                values, idx_xyz, idx_c, type_counts
            )

        # convert spatial index to atomic coordinates
        coords = grid.get_coords(idx_xyz)

        # suppress atoms too close to a higher-value atom of same type
        if (
            not_none(self.min_dist)
            and self.min_dist > 0.0
            and self.n_atoms_detect > 1
        ):
            coords, idx_c = self.suppress_non_max(values, coords, idx_c)

        # limit total number of detected atoms
        if self.n_atoms_detect >= 0:
            coords = coords[:self.n_atoms_detect]
            idx_c = idx_c[:self.n_atoms_detect]

        # convert atom type channel index to one-hot type vector
        types = F.one_hot(idx_c, grid.n_elem_channels).to(
            dtype=grid.dtype, device=self.device
        )
        return coords.detach(), types.detach()

    def fit_gd(self, grid, xyz, c, n_iters):

        r = torch.tensor(
            [ch.atomic_radius for ch in grid.channels],
            device=self.device,
        )
        xyz = xyz.clone().detach().to(self.device)
        c = c.clone().detach().to(self.device)
        xyz.requires_grad = True

        solver = torch.optim.Adam((xyz,), **self.gd_kwargs)

        self.grid_maker.set_radii_type_indexed(True)
        self.grid_maker.set_dimension(grid.dimension)
        self.grid_maker.set_resolution(grid.resolution)
        self.c2grid.center = tuple(grid.center.cpu().numpy().astype(float))

        for i in range(n_iters + 1):
            solver.zero_grad()

            grid_pred = self.c2grid(xyz, c, r)
            grid_diff = grid.values - grid_pred
            if self.fit_L1_loss:
                loss = grid_diff.abs().sum()
            else:
                loss = (grid_diff**2).sum() / 2.0

            if i == n_iters: # or converged
                break

            loss.backward()
            solver.step()

        return (
            xyz.detach(),
            grid_pred.detach(),
            grid_diff.detach(),
            loss.detach()
        )

    def get_types_estimate(self, grid, channels, resolution):
        '''
        Since atom density is additive and non-negative, estimate
        the atom type counts by dividing the total density in each
        grid channel by the total density in each kernel channel.
        '''
        if self.kernel is None:
            self.init_kernel(channels, resolution)

        kernel_sum = self.kernel.values.sum(dim=(1,2,3))
        grid_sum = grid.sum(dim=(1,2,3))
        return grid_sum / kernel_sum

    def fit(self, grid, type_counts=None):
        '''
        Fit atom types and coordinates to atomic density grid.
        '''
        t_start = time.time()
        torch.cuda.reset_max_memory_allocated()

        # get true grid on appropriate device
        grid_true = grid.to(self.device)

        # get true atom type counts on appropriate device
        if type_counts is not None:
            type_counts = type_counts.to(self.device, dtype=torch.float32)

        if self.estimate_types: # estimate atom type counts from grid density
            type_counts_est = self.get_types_estimate(
                grid_true.values, grid_true.channels, grid_true.resolution,
            )
            est_type_loss = (type_counts - type_counts_est).abs().sum().item()
            type_counts = type_counts_est
        else:
            est_type_loss = np.nan

        # initialize empty struct
        if self.verbose:
            print('Initializing empty struct 0')
        n_channels = len(grid.channels)
        xyz = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        c = torch.zeros((0, n_channels), dtype=torch.float32, device=self.device)

        if self.fit_L1_loss:
            fit_loss = grid_true.values.abs().sum()
        else:
            fit_loss = (grid_true.values**2).sum() / 2.0

        # to constrain types, order structs first by type diff, then L2 loss
        if self.constrain_types:
            type_loss = type_counts.abs().sum()
            objective = (type_loss.item(), fit_loss.item())

        else: # otherwise, order structs only by L2 loss
            objective = fit_loss.item()

        # get next atom init locations and channels
        if self.verbose:
            print('Getting next atoms for struct 0')
        xyz_next, c_next = self.detect_atoms(
            grid_true.values,
            grid_true.channels,
            grid_true.center,
            grid_true.resolution,
            type_counts,
        )

        # keep track of best structures so far
        struct_id = 0
        best_structs = [(objective, struct_id, xyz, c, xyz_next, c_next)]
        found_new_best_struct = True

        # keep track of visited and expanded structures
        expanded_ids = set()
        visited_structs = [(objective, struct_id, time.time()-t_start, xyz, c)]
        struct_count = 1

        mi = torch.cuda.max_memory_allocated()
        ms = []

        # search until we can't find a better structure
        while found_new_best_struct:
            torch.cuda.reset_max_memory_allocated()

            new_best_structs = []
            found_new_best_struct = False

            # try to expand each current best structure
            for objective, struct_id, xyz, c, xyz_next, c_next in best_structs:

                if struct_id in expanded_ids:
                    continue

                do_single_atom = True
                #print('expanding struct {} to {} next atoms'.format(struct_id, len(c_next)))

                if self.multi_atom: # evaluate all next atoms simultaneously

                    xyz_new = torch.cat([xyz, xyz_next])
                    c_new = torch.cat([c, c_next])

                    # compute diff and loss after gradient descent
                    xyz_new, grid_pred, grid_diff, fit_loss = self.fit_gd(
                        grid_true, xyz_new, c_new, self.interm_gd_iters
                    )

                    if self.constrain_types:
                        type_diff = types - c_new.sum(dim=0)
                        type_loss = type_diff.abs().sum()
                        objective_new = (type_loss.item(), fit_loss.item())
                    else:
                        type_diff = None
                        objective_new = fit_loss.item()

                    # check if new structure is one of the best yet
                    if any(objective_new < s[0] for s in best_structs):

                        if self.verbose:
                            print('Found new best struct {}'.format(struct_count))

                        xyz_new_next, c_new_next = self.detect_atoms(
                            grid_diff,
                            grid_true.channels,
                            grid_true.center,
                            grid_true.resolution,
                            type_diff,
                        )
                        new_best_structs.append(
                            (
                                objective_new,
                                struct_count,
                                xyz_new,
                                c_new,
                                xyz_new_next,
                                c_new_next,
                            )
                        )
                        found_new_best_struct = True
                        struct_count += 1
                        do_single_atom = False

                    visited_structs.append(
                        (objective_new, struct_id, time.time()-t_start, xyz_new, c_new)
                    )

                if do_single_atom:

                    # evaluate each possible next atom individually
                    for xyz_next_, c_next_ in zip(xyz_next, c_next):

                        # add next atom to structure
                        xyz_new = torch.cat([xyz, xyz_next_.unsqueeze(0)])
                        c_new = torch.cat([c, c_next_.unsqueeze(0)])

                        # compute diff and loss after gradient descent
                        xyz_new, grid_pred, grid_diff, fit_loss = self.fit_gd(
                            grid_true, xyz_new, c_new, self.interm_gd_iters
                        )

                        if self.constrain_types:
                            type_diff = types - c_new.sum(dim=0)
                            type_loss = type_diff.abs().sum()
                            objective_new = (type_loss.item(), fit_loss.item())
                        else:
                            type_diff = None
                            objective_new = fit_loss.item()

                        # check if new structure is one of the best yet
                        if any(objective_new < s[0] for s in best_structs):

                            if self.verbose:
                                print('Found new best struct {}'.format(struct_count))

                            xyz_new_next, c_new_next = self.detect_atoms(
                                grid_diff,
                                grid_true.channels,
                                grid_true.center,
                                grid_true.resolution,
                                type_diff,
                            )
                            new_best_structs.append(
                                (
                                    objective_new,
                                    struct_count,
                                    xyz_new,
                                    c_new,
                                    xyz_new_next,
                                    c_new_next,
                                )
                            )
                            found_new_best_struct = True
                            struct_count += 1

                        visited_structs.append(
                            (objective_new, struct_id, time.time()-t_start, xyz_new, c_new)
                        )

                expanded_ids.add(struct_id)

            if found_new_best_struct: # determine new set of best structures

                best_structs = sorted(best_structs + new_best_structs)[:self.beam_size]
                best_objective = best_structs[0][0]
                best_id = best_structs[0][1]
                best_n_atoms = best_structs[0][2].shape[0]

                if self.verbose:
                    try:
                        gpu_usage = getGPUs()[0].memoryUtil
                    except:
                        gpu_usage = np.nan
                    print(
                        'Best struct {} (objective={}, n_atoms={}, GPU={})'.format(
                            best_id, best_objective, best_n_atoms, gpu_usage
                        )
                    )

                if len(xyz_new) >= 50:
                    found_new_best_struct = False #dkoes: limit molecular size

            ms.append(torch.cuda.max_memory_allocated())

        torch.cuda.reset_max_memory_allocated()

        # done searching for atomic structures
        best_objective, best_id, xyz_best, c_best, _, _ = best_structs[0]

        # perform final gradient descent
        xyz_best, grid_pred, grid_diff, fit_loss = self.fit_gd(
            grid_true, xyz_best, c_best, self.final_gd_iters
        )

        # compute the final L2 and L1 loss
        L2_loss = (grid_diff**2).sum() / 2
        L1_loss = grid_diff.abs().sum()

        if self.constrain_types:
            type_loss = (type_counts - c_best.sum(dim=0)).abs().sum().item()
            best_objective = (type_loss.item(), fit_loss.item())
        else:
            type_loss = np.nan
            best_objective = fit_loss.item()

        # make sure best struct is the last visited struct
        visited_structs.append(
            (best_objective, best_id+1, time.time()-t_start, xyz_best, c_best)
        )

        # finalize the visited atomic structures
        visited_structs_ = iter(visited_structs)
        visited_structs = []
        for objective, struct_id, fit_time, xyz, c in visited_structs_:

            struct = AtomStruct(
                xyz=xyz.detach(),
                c=one_hot_to_index(c).detach(),
                channels=grid.channels,
                L2_loss=L2_loss,
                L1_loss=L1_loss,
                type_diff=type_loss,
                est_type_diff=est_type_loss,
                time=fit_time,
            )
            visited_structs.append(struct)

        # finalize the best fit atomic structure and density grid
        struct_best = AtomStruct(
            xyz=xyz_best.detach(),
            c=one_hot_to_index(c_best).detach(),
            channels=grid.channels,
            L2_loss=L2_loss,
            L1_loss=L1_loss,
            type_diff=type_loss,
            est_type_diff=est_type_loss,
            visited_structs=visited_structs,
            time=time.time()-t_start,
        )

        grid_pred = AtomGrid(
            values=grid_pred.detach(),
            channels=grid.channels,
            center=grid.center,
            resolution=grid.resolution,
        )

        mf = torch.cuda.max_memory_allocated()

        if self.debug:
            MB = int(1024 ** 2)
            print('GPU', mi//MB, [m//MB for m in ms], mf//MB)

        return struct_best, grid_pred

    def fit_batch(self, grids, channels, center, resolution):

        fit_structs, fit_grids = [], []
        for grid in grids:
            grid = AtomGrid(
                grid.detach(), channels, center, resolution
            )
            fit_struct, fit_grid = self.fit(grid)
            fit_structs.append(fit_struct)
            fit_grids.append(fit_grid)

        return fit_structs, fit_grids


class DkoesAtomFitter(AtomFitter):

    def __init__(self, dkoes_make_mol, use_openbabel, iters=25, tol=0.01):
        self.iters = iters
        self.tol = tol
        self.verbose=False
        self.dkoes_make_mol = dkoes_make_mol
        self.use_openbabel = use_openbabel

    def fit(self, grid, types):

        grid = AtomGrid(
            values=grid.values,
            channels=grid.channels,
            center=grid.center,
            resolution=grid.resolution,
        )

        struct, grid_pred = \
            dkoes_fitting.simple_atom_fit(
                mgrid=grid,
                types=types,
                iters=self.iters,
                tol=self.tol,
                grm=1.0
            )

        struct.info['visited_structs'] = [struct]
        self.validify(struct)

        grid_pred = AtomGrid(
            values=grid_pred.cpu().detach().numpy(),
            channels=grid.channels,
            center=grid.center,
            resolution=grid.resolution,
            visited_structs=[struct],
            src_struct=struct,
        )        

        return remove_tensors(grid_pred)


def remove_tensors(obj, visited=None):
    '''
    Recursively traverse an object converting pytorch tensors
    to numpy arrays in-place.
    '''
    visited = visited or set()

    if not isinstance(obj, (AtomGrid, AtomStruct, list, dict)) or id(obj) in visited:
        #avoid traversing everything
        return obj

    visited.add(id(obj))

    dct = None
    if isinstance(obj, dict):
        dct = obj
    elif hasattr(obj, '__dict__'):
        dct = obj.__dict__

    if dct:
        for k, v in dct.items():
            if isinstance(v, torch.Tensor):
                dct[k] = v.cpu().detach().numpy()
            else:
                dct[k] = remove_tensors(v, visited)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = remove_tensors(obj[i], visited)

    return obj


def make_one_hot(x, n, dtype=None, device=None):
    y = torch.zeros(x.shape + (n,), dtype=dtype, device=device)
    for idx, last_idx in np.ndenumerate(x):
        y[idx + (int(last_idx),)] = 1
    return y


def one_hot_to_index(x):
    if len(x) > 0:
        return torch.argmax(x, dim=1)
    else:
        return torch.empty((0,), device=x.device)


def conv_grid(grid, kernel):
    # convolution theorem: g * grid = F-1(F(g)F(grid))
    F_h = np.fft.fftn(kernel)
    F_grid = np.fft.fftn(grid)
    return np.real(np.fft.ifftn(F_grid * F_h))


def weiner_invert_kernel(kernel, noise_ratio=0.0):
    F_h = np.fft.fftn(kernel)
    conj_F_h = np.conj(F_h)
    F_g = conj_F_h / (F_h*conj_F_h + noise_ratio)
    return np.real(np.fft.ifftn(F_g))


def wiener_deconv_grid(grid, kernel, noise_ratio=0.0):
    '''
    Applies a convolution to the input grid that approximates the inverse
    of the operation that converts a set of atom positions to a grid of
    atom density.
    '''
    # we want a convolution g such that g * grid = a, where a is the atom positions
    # we assume that grid = h * a, so g is the inverse of h: g * (h * a) = a
    # take F() to be the Fourier transform, F-1() the inverse Fourier transform
    # convolution theorem: g * grid = F-1(F(g)F(grid))
    # Wiener deconvolution: F(g) = 1/F(h) |F(h)|^2 / (|F(h)|^2 + noise_ratio)
    F_h = np.fft.fftn(kernel)
    F_grid = np.fft.fftn(grid)
    conj_F_h = np.conj(F_h)
    F_g = conj_F_h / (F_h*conj_F_h + noise_ratio)
    return np.real(np.fft.ifftn(F_grid * F_g))


def wiener_deconv_grids(grids, channels, resolution, radius_multiple, noise_ratio=0.0, radius_factor=1.0):

    deconv_grids = np.zeros_like(grids)
    points = get_grid_points(grids.shape[1:], 0, resolution)

    for i, grid in enumerate(grids):

        r = channels[i].atomic_radius*radius_factor
        kernel = get_atom_density(resolution/2, r, points, radius_multiple).reshape(grid.shape)
        kernel = np.roll(kernel, shift=[d//2 for d in grid.shape], axis=range(grid.ndim))
        deconv_grids[i,...] = wiener_deconv_grid(grid, kernel, noise_ratio)

    return np.stack(deconv_grids, axis=0)


def get_grid_points(shape, center, resolution):
    '''
    Return an array of points for a grid with
    the given shape, center, and resolution.
    '''
    shape = np.array(shape)
    center = np.array(center)
    resolution = np.array(resolution)
    origin = center - resolution*(shape - 1)/2.0
    indices = np.array(list(np.ndindex(*shape)))
    return origin + resolution*indices


def grid_to_points_and_values(grid, center, resolution):
    '''
    Convert a grid with a center and resolution to lists
    of grid points and values at each point.
    '''
    points = get_grid_points(grid.shape, center, resolution)
    return points, grid.flatten()
