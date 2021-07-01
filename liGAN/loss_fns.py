from collections import OrderedDict as odict
import torch
from torch import nn


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
    def __init__(self, types, weights, device):
        super().__init__()
        self.init_loss_weights(**weights)
        self.init_loss_types(**types)
        self.device = device

    def init_loss_weights(
        self, kldiv_loss=0, recon_loss=0, gan_loss=0, steric_loss=0
    ):
        self.kldiv_loss_wt = float(kldiv_loss)
        self.recon_loss_wt = float(recon_loss)
        self.gan_loss_wt = float(gan_loss)
        self.steric_loss_wt = float(steric_loss)

    def init_loss_types(
        self, kldiv_loss='k', recon_loss='2', gan_loss='x', steric_loss='p'
    ):
        self.init_kldiv_loss_fn(kldiv_loss)
        self.init_recon_loss_fn(recon_loss)
        self.init_gan_loss_fn(gan_loss)
        self.init_steric_loss_fn(steric_loss)

    def init_kldiv_loss_fn(self, kldiv_loss_type):
        self.kldiv_loss_fn = get_kldiv_loss_fn(kldiv_loss_type)

    def init_recon_loss_fn(self, recon_loss_type):
        self.recon_loss_fn = get_recon_loss_fn(recon_loss_type)

    def init_gan_loss_fn(self, gan_loss_type):
        self.gan_loss_fn = get_gan_loss_fn(gan_loss_type)

    def init_steric_loss_fn(self, steric_loss_type):
        self.steric_loss_fn = get_steric_loss_fn(steric_loss_type)

    @property
    def has_prior_loss(self):
        '''
        Whether the loss function ever has
        non-zero value on prior samples.
        '''
        return self.gan_loss_wt != 0 or self.steric_loss_wt != 0

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

        Each term is computed iff both of its inputs are
        provided to the method, and each computed term is
        also returned as values in an OrderedDict.
        '''
        not_none = lambda x: x is not None
        losses = odict()
        loss = torch.zeros(1, device=self.device)

        if not_none(lig_grids) and not_none(lig_gen_grids):
            recon_loss = self.recon_loss_fn(lig_gen_grids, lig_grids)
            losses['recon_loss'] = recon_loss.item()
            loss += self.recon_loss_wt * recon_loss

        if not_none(latent_means) and not_none(latent_log_stds):
            kldiv_loss = self.kldiv_loss_fn(latent_means, latent_log_stds)
            losses['kldiv_loss'] = kldiv_loss.item()
            loss += self.kldiv_loss_wt * kldiv_loss

        if not_none(disc_labels) and not_none(disc_preds):
            gan_loss = self.gan_loss_fn(disc_preds, disc_labels)
            losses['gan_loss'] = gan_loss.item()
            loss += self.gan_loss_wt * gan_loss

        if not_none(rec_grids) and not_none(rec_lig_grids):
            steric_loss = self.steric_loss_fn(rec_grids, rec_lig_grids)
            losses['steric_loss'] = steric_loss.item()
            loss += self.steric_loss_wt * steric_loss

        losses['loss'] = loss.item()
        return loss, losses


### functions for getting loss functions from config


def get_kldiv_loss_fn(kldiv_loss_type):
    assert kldiv_loss_type == 'k', kldiv_loss_type
    return kl_divergence


def get_recon_loss_fn(recon_loss_type):
    assert recon_loss_type in {'1', '2'}, recon_loss_type
    if recon_loss_type == '1':
        return L1_loss
    else:
        return L2_loss


def get_gan_loss_fn(gan_loss_type):
    assert gan_loss_type in {'x', 'w'}, gan_loss_type
    if gan_loss_type == 'w':
        return wasserstein_loss
    else:
        return torch.nn.BCEWithLogitsLoss()


def get_steric_loss_fn(steric_loss_type):
    assert steric_loss_type == 'p', steric_loss_type
    return product_loss


### actual loss function definitions


def kl_divergence(means, log_stds):
    stds = torch.exp(log_stds)
    means2 = means * means
    vars_ = stds * stds
    return (
        -log_stds + means2/2 + vars_/2 - 0.5
    ).sum() / means.shape[0]


def L1_loss(predictions, labels):
    return (labels - predictions).abs().sum() / labels.shape[0]


def L2_loss(predictions, labels):
    return ((labels - predictions)**2).sum() / 2 / labels.shape[0]


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
