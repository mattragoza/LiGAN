import sys, os, pytest, time, torch
from numpy import isclose, isnan
from torch import optim

sys.path.insert(0, '.')
import liGAN
from liGAN.loss_fns import LossFunction


class TestLossFunction(object):

    @pytest.fixture
    def loss_kws(self):
        return dict(
            types=dict(recon_loss='2', gan_loss='w'),
            weights=dict(recon_loss=1, gan_loss=10, steric_loss=-1)
        )

    @pytest.fixture
    def loss_fn(self, loss_kws):
        return LossFunction(device='cpu', **loss_kws)

    def test_loss_init(self, loss_fn):
        assert loss_fn.kldiv_loss_wt == 0.0
        assert loss_fn.recon_loss_wt == 1.0
        assert loss_fn.gan_loss_wt == 10.0
        assert loss_fn.steric_loss_wt == -1.0

    def test_loss_null(self, loss_fn):
        loss, losses = loss_fn()
        assert loss.item() == 0
        assert losses == dict(loss=0)

    def test_loss_recon0(self, loss_fn):
        lig_grids = torch.zeros(10, 10)
        lig_gen_grids = torch.zeros(10, 10)
        loss, losses = loss_fn(
            lig_grids=lig_grids, lig_gen_grids=lig_gen_grids
        )
        assert loss.item() == 0
        print(losses)
        assert losses == dict(loss=0, recon_loss=0, recon_loss_wt=1, recon_log_var=0)

    def test_loss_recon1(self, loss_fn):
        lig_grids = torch.zeros(10, 10)
        lig_gen_grids = torch.ones(10, 10)
        loss, losses = loss_fn(
            lig_grids=lig_grids, lig_gen_grids=lig_gen_grids
        )
        assert loss.item() == 5
        print(losses)
        assert losses == dict(loss=5, recon_loss=5, recon_loss_wt=1, recon_log_var=0)

    def test_loss_kldiv0(self, loss_fn):
        a = torch.zeros(10, 10)
        b = torch.zeros(10, 10)
        loss, losses = loss_fn(
            latent_means=a, latent_log_stds=b
        )
        assert loss.item() == 0
        assert losses == dict(loss=0, kldiv_loss=0, kldiv_loss_wt=0)

    def test_loss_kldiv1(self, loss_fn):
        a = torch.ones(10, 10)
        b = torch.zeros(10, 10)
        loss, losses = loss_fn(
            latent_means=a, latent_log_stds=b
        )
        assert loss.item() == 0
        assert losses == dict(loss=0, kldiv_loss=5, kldiv_loss_wt=0)

    def test_loss_gan0(self, loss_fn):
        a = torch.zeros(10, 10)
        b = torch.zeros(10, 10)
        loss, losses = loss_fn(
            disc_labels=a, disc_preds=b
        )
        assert loss.item() == 0
        assert losses == dict(loss=0, gan_loss=0, gan_loss_wt=10)

    def test_loss_gan1(self, loss_fn):
        a = torch.ones(10, 10)
        b = torch.ones(10, 10)
        loss, losses = loss_fn(
            disc_labels=a, disc_preds=b
        )
        assert loss.item() == 100
        print(losses)
        assert losses == dict(loss=100, gan_loss=10, gan_loss_wt=10)

    def test_loss_steric0(self, loss_fn):
        a = torch.zeros(10, 10)
        b = torch.zeros(10, 10)
        loss, losses = loss_fn(
            rec_grids=a, rec_lig_grids=b
        )
        assert loss.item() == 0
        assert losses == dict(loss=0, steric_loss=0, steric_loss_wt=-1)

    def test_loss_steric1(self, loss_fn):
        a = torch.ones(10, 10)
        b = torch.ones(10, 10)
        loss, losses = loss_fn(
            rec_grids=a, rec_lig_grids=b
        )
        assert loss.item() == -100
        assert losses == dict(loss=-100, steric_loss=100, steric_loss_wt=-1)


class TestLossSchedule(object):

    @pytest.fixture
    def iters(self):
        return range(1, 1000001, 1000)

    def test_schedule_0(self, iters):

        f = liGAN.loss_fns.get_loss_schedule(
            start_wt=1, end_wt=1
        )
