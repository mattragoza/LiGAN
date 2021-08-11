import torch
import molgrid


class Interpolation(torch.nn.Module):

    def __init__(self, n_samples):
        super().__init__()
        self.endpoints = None
        self.center = None # sphere center
        self.n_samples = n_samples
        self.curr_step = 0

    @property
    def is_initialized(self):
        return self.endpoints is not None

    def initialize(self, init_point, center=None):
        assert not self.is_initialized
        self.endpoints = init_point.unsqueeze(0)
        if center is not None:
            self.center = center.unsqueeze(0)

    def forward(self, inputs, spherical=False):

        assert len(inputs.shape) == 2, 'inputs must be vectors'
        batch_size = inputs.shape[0]

        # whether each input is an interpolation endpoint
        batch_idx = torch.arange(batch_size, device=inputs.device)
        is_endpoint = (self.curr_step + batch_idx) % self.n_samples == 0

        # concat the new endpoints to the list of all endpoints
        self.endpoints = torch.cat([self.endpoints, inputs[is_endpoint]])

        # get start and stop points for each batch idx
        start_idx = (self.curr_step + batch_idx) // self.n_samples
        start_points = self.endpoints[start_idx]
        stop_points = self.endpoints[start_idx+1]

        # get amount to interpolate vectors at each batch_idx
        k_interp = (
            (self.curr_step + batch_idx) % self.n_samples + 1
        ).unsqueeze(1) / self.n_samples

        # do interpolation
        if spherical:
            outputs = slerp(start_points, stop_points, k_interp, self.center)
        else:
            outputs = lerp(start_points, stop_points, k_interp)
        assert not outputs.isnan().any()

        self.curr_step += batch_size


        return outputs


class TransformInterpolation(Interpolation):

    def initialize(self, example):
        rec_coord_set, lig_coord_set = example.coord_sets
        rec_center = tuple(rec_coord_set.center())
        lig_center = tuple(lig_coord_set.center())
        super().initialize(
            init_point=torch.as_tensor(lig_center),
            center=torch.as_tensor(rec_center)
        )

    def forward(self, transforms, **kwargs):

        # just interpolate the centers for now
        centers = torch.tensor(
            [tuple(t.get_rotation_center()) for t in transforms],
            dtype=float
        )
        centers = super().forward(centers, **kwargs)
        return [
            molgrid.Transform(
                t.get_quaternion(),
                tuple(center.numpy()),
                t.get_translation(),
            ) for t, center in zip(transforms, centers)
        ]


def lerp(v0, v1, t):
    '''
    Linear interpolation between
    vectors v0 and v1 at steps t.
    '''
    k0, k1 = (1-t), t
    return k0*v0 + k1*v1


def slerp(v0, v1, t, center=None):
    '''
    Spherical linear interpolation between
    vectors v0 and v1 at steps t.
    '''
    eps = 1e-6
    if center is not None:
        v0 -= center
        v1 -= center
    norm_v0 = v0.norm(dim=1, keepdim=True)
    norm_v1 = v1.norm(dim=1, keepdim=True)
    dot_v0_v1 = (v0*v1).sum(dim=1, keepdim=True)
    cos_theta = dot_v0_v1 / (norm_v0 * norm_v1)
    theta = torch.acos(cos_theta) + eps # angle between the vectors
    sin_theta = torch.sin(theta)
    k0 = torch.sin((1-t)*theta) / sin_theta
    k1 = torch.sin(t*theta) / sin_theta
    if center is not None:
        return k0*v0 + k1*v1 + center
    else:
        return k0*v0 + k1*v1
