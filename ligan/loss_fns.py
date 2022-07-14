from collections import OrderedDict as odict
from math import pi
import torch
from torch import nn
pi = torch.tensor(pi)

has_both = lambda a, b: (
    a is not None and b is not None
)

has_both = lambda a, b: (
    a is not None and b is not None
)


class LossFunction(nn.Module):
    '''
    A multi-task loss function for training
    generative models of atomic density grids.

    The loss function combines KL divergence,
    reconstruction, GAN discrimination, and/or
    receptor-ligand steric clash.

    Each term can have a different loss weight
    associated with it, and different types of
    loss functions are available for each term.

    The different loss terms are computed based
    on the input provided to the forward method.
    '''
    def __init__(self, types, weights, schedules={}, device='cuda'):
        super().__init__()
        self.init_loss_weights(**weights)
        self.init_loss_types(**types)
        self.init_loss_schedules(**schedules)
        self.device = device

    def init_loss_weights(
        self,
        kldiv_loss=0,
        recon_loss=0,
        gan_loss=0,
        steric_loss=0,
        kldiv2_loss=0,
        recon2_loss=0,
    ):
        self.kldiv_loss_wt = float(kldiv_loss)
        self.recon_loss_wt = float(recon_loss)
        self.gan_loss_wt = float(gan_loss)
        self.steric_loss_wt = float(steric_loss)
        self.kldiv2_loss_wt = float(kldiv2_loss)
        self.recon2_loss_wt = float(recon2_loss)

    def init_loss_types(
        self,
        kldiv_loss='k',
        recon_loss='2',
        gan_loss='x',
        steric_loss='p',
        kldiv2_loss='k',
        recon2_loss='2',
    ):
        self.kldiv_loss_fn = get_kldiv_loss_fn(kldiv_loss)
        self.recon_loss_fn = get_recon_loss_fn(recon_loss)
        self.gan_loss_fn = get_gan_loss_fn(gan_loss)
        self.steric_loss_fn = get_steric_loss_fn(steric_loss)
        self.kldiv2_loss_fn = get_kldiv_loss_fn(kldiv2_loss)
        self.recon2_loss_fn = get_recon_loss_fn(recon2_loss)

    def init_loss_schedules(
        self,
        kldiv_loss={},
        recon_loss={},
        gan_loss={},
        steric_loss={},
        kldiv2_loss={},
        recon2_loss={},
    ):
        self.kldiv_loss_schedule, _ = get_loss_schedule(
            start_wt=self.kldiv_loss_wt, **kldiv_loss
        )
        self.recon_loss_schedule, _ = get_loss_schedule(
            start_wt=self.recon_loss_wt, **recon_loss
        )
        self.gan_loss_schedule, self.end_gan_loss_wt = get_loss_schedule(
            start_wt=self.gan_loss_wt, **gan_loss
        )
        self.steric_loss_schedule, self.end_steric_loss_wt = get_loss_schedule(
            start_wt=self.steric_loss_wt, **steric_loss
        )
        self.kldiv2_loss_schedule, _ = get_loss_schedule(
            start_wt=self.kldiv2_loss_wt, **kldiv2_loss
        )
        self.recon2_loss_schedule, _ = get_loss_schedule(
            start_wt=self.recon2_loss_wt, **recon2_loss
        )

    @property
    def has_prior_loss(self):
        '''
        Whether the loss function ever has
        non-zero value on prior samples.
        '''
        return bool(
            self.gan_loss_wt or self.end_gan_loss_wt or 
            self.steric_loss_wt or self.end_steric_loss_wt
        )

    def forward(
        self,
        latent_means=None,
        latent_log_stds=None,
        lig_grids=None,
        lig_gen_grids=None,
        disc_labels=None,
        disc_preds=None,
        rec_grids=None,
        rec_lig_grids=None,
        latent2_means=None,
        latent2_log_stds=None,
        real_latents=None,
        gen_latents=None,
        gen_log_var=torch.zeros(1),
        prior_log_var=torch.zeros(1),
        use_loss_wt=True,
        iteration=0,
    ):
        '''
        Computes the loss as follows:

        = kldiv_loss_wt * 
            kldiv_loss_fn(latent_means, latent_log_stds)
        + recon_loss_wt * 
            recon_loss_fn(lig_gen_grids, lig_grids)
        + gan_loss_wt * 
            gan_loss_fn(disc_preds, disc_labels)
        + steric_loss_wt * 
            steric_loss_fn(rec_grids, rec_lig_grids)
        + ...

        Each term is computed iff both of its inputs are
        provided to the method, and each computed term is
        also returned as values in an OrderedDict.
        '''
        loss = torch.zeros(1, device=self.device)
        losses = odict() # track each loss term

        if has_both(lig_grids, lig_gen_grids):
            recon_loss = self.recon_loss_fn(
                lig_gen_grids, lig_grids, gen_log_var
            )
            recon_loss_wt = self.recon_loss_schedule(iteration, use_loss_wt)
            loss += recon_loss_wt * recon_loss
            losses['recon_loss'] = recon_loss.item()
            losses['recon_loss_wt'] = recon_loss_wt.item()
            losses['recon_log_var'] = gen_log_var.item()

        if has_both(latent_means, latent_log_stds):
            kldiv_loss = self.kldiv_loss_fn(latent_means, latent_log_stds)
            kldiv_loss_wt = self.kldiv_loss_schedule(iteration, use_loss_wt)
            loss += kldiv_loss_wt * kldiv_loss
            losses['kldiv_loss'] = kldiv_loss.item()
            losses['kldiv_loss_wt'] = kldiv_loss_wt.item()

        if has_both(disc_labels, disc_preds):
            gan_loss = self.gan_loss_fn(disc_preds, disc_labels)
            gan_loss_wt = self.gan_loss_schedule(iteration, use_loss_wt)
            loss += gan_loss_wt * gan_loss
            losses['gan_loss'] = gan_loss.item()
            losses['gan_loss_wt'] = gan_loss_wt.item()

        if has_both(rec_grids, rec_lig_grids):
            steric_loss = self.steric_loss_fn(rec_grids, rec_lig_grids)
            steric_loss_wt = self.steric_loss_schedule(iteration, use_loss_wt)
            loss += steric_loss_wt * steric_loss
            losses['steric_loss'] = steric_loss.item()
            losses['steric_loss_wt'] = steric_loss_wt.item()

        if has_both(latent2_means, latent2_log_stds):
            kldiv2_loss = self.kldiv2_loss_fn(latent2_means, latent2_log_stds)
            kldiv2_loss_wt = self.kldiv2_loss_schedule(iteration, use_loss_wt)
            loss += kldiv2_loss_wt * kldiv2_loss
            losses['kldiv2_loss'] = kldiv2_loss.item()
            losses['kldiv2_loss_wt'] = kldiv2_loss_wt.item()

        if has_both(real_latents, gen_latents):
            recon2_loss = self.recon2_loss_fn(
                gen_latents, real_latents, prior_log_var
            )
            recon2_loss_wt = self.recon2_loss_schedule(iteration, use_loss_wt)
            loss += recon2_loss_wt * recon2_loss
            losses['recon2_loss'] = recon2_loss.item()
            losses['recon2_loss_wt'] = recon2_loss_wt.item()
            losses['recon2_log_var'] = prior_log_var.item()

        losses['loss'] = loss.item()
        return loss, losses


### function for getting loss schedule fn from config

def get_loss_schedule(
    start_wt,
    start_iter=500000,
    end_wt=None,
    period=100000,
    type='d',
):
    '''
    Return a function that takes the
    iteration as input and returns a
    modulated loss weight as output.
    '''
    no_schedule = (
        type == 'n' or end_wt is None or end_wt == start_wt
    )
    assert no_schedule or period > 0, period
    assert type in {'n', 'd', 'c', 'r'}, type
    periodic = (type == 'c ' or type == 'r')
    restart = (type == 'r')
    end_iter = start_iter + period

    def loss_schedule(iteration, use_loss_wt):
        if not use_loss_wt:
            return torch.tensor(1)
        if no_schedule or iteration < start_iter:
            return torch.as_tensor(start_wt)
        if iteration >= end_iter and not periodic:
            return torch.as_tensor(end_wt)
        wt_range = (end_wt - start_wt)
        theta = (iteration - start_iter) / period * pi
        if restart: # jump from end_wt to start_wt
            theta %= pi
        return end_wt - wt_range * 0.5 * (1 + torch.cos(theta))

    return loss_schedule, end_wt


### functions for getting loss functions from config


def get_kldiv_loss_fn(type):
    assert type == 'k', type
    return kl_divergence


def get_recon_loss_fn(type):
    assert type in {'1', '2'}, type
    if type == '1':
        return L1_loss
    else: # '2'
        return L2_loss


def get_gan_loss_fn(type):
    assert type in {'x', 'w'}, type
    if type == 'w':
        return wasserstein_loss
    else: # 'x'
        return torch.nn.BCEWithLogitsLoss()


def get_steric_loss_fn(type):
    assert type == 'p', type
    return product_loss


### actual loss function definitions


def kl_divergence(means, log_stds):
    stds = torch.exp(log_stds)
    means2 = means * means
    vars_ = stds * stds
    return (
        -log_stds + means2/2 + vars_/2 - 0.5
    ).sum() / means.shape[0]


def L1_loss(predictions, labels, log_var=0):
    return torch.sum(
        ((labels - predictions) / torch.exp(log_var)).abs() + log_var
    ) / labels.shape[0]


def L2_loss(predictions, labels, log_var=0):
    # https://github.com/daib13/TwoStageVAE/blob/master/network/two_stage_vae_model.py#L39
    return torch.sum(
        ((labels - predictions) / torch.exp(log_var))**2 / 2.0 + log_var
    ) / labels.shape[0]


def wasserstein_loss(predictions, labels):
    labels = (2*labels - 1) # convert {0, 1} to {-1, 1}
    return (labels * predictions).sum() / labels.shape[0]


def product_loss(rec_grids, lig_grids):
    '''
    Minimize receptor-ligand overlap
    by summing the pointwise products
    of total density at each point.
    '''
    return (
        rec_grids.sum(dim=1) * lig_grids.clamp(min=0).sum(dim=1)
    ).sum() / lig_grids.shape[0]
