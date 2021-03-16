import time
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from torch import nn
torch.backends.cudnn.benchmark = True

import molgrid
from . import data, models, atom_types
from .metrics import (
    compute_grid_metrics,
    compute_paired_grid_metrics,
    compute_struct_metrics,
    compute_paired_struct_metrics
)


def kl_divergence(means, log_stds):
    stds = torch.exp(log_stds)
    means2 = means * means
    vars_ = stds * stds
    return (
        -log_stds + means2/2 + vars_/2 - 0.5
    ).sum() / means.shape[0]


def wasserstein_loss(predictions, labels):
    return ((2*labels - 1) * predictions).sum() / labels.shape[0]


def L1_loss(predictions, labels):
    return (labels - predictions).abs() / labels.shape[0]


def L2_loss(predictions, labels):
    return ((labels - predictions)**2).sum() / 2 / labels.shape[0]


def get_recon_loss_fn(loss_type='2'):
    assert loss_type in {'1', '2'}
    if loss_type == '1':
        return L1_loss
    else:
        return L2_loss


def get_gan_loss_fn(loss_type='x'):
    assert loss_type in {'x', 'w'}
    if loss_type == 'w':
        return wasserstein_loss
    else:
        return torch.nn.BCEWithLogitsLoss()


class Solver(nn.Module):
    gen_model_type = None
    index_cols = ['iteration', 'phase', 'batch']

    def __init__(
        self,
        data_root,
        train_file,
        test_file,
        batch_size,
        rec_map_file,
        lig_map_file,
        resolution,
        grid_size,
        shuffle,
        random_rotation,
        random_translation,
        rec_molcache,
        lig_molcache,
        n_filters,
        width_factor,
        n_levels,
        conv_per_level,
        kernel_size,
        relu_leak,
        pool_type,
        unpool_type,
        pool_factor,
        n_latent,
        init_conv_pool,
        skip_connect,
        loss_weights,
        loss_types,
        grad_norm_types,
        optim_type,
        optim_kws,
        atom_fitter_type,
        atom_fitter_kws,
        out_prefix,
        random_seed=None,
        device='cuda',
    ):
        super().__init__()
        self.device = device

        if random_seed is not None:
            self.set_random_seed(random_seed)

        self.train_data, self.test_data = (
            data.AtomGridData(
                data_root=data_root,
                batch_size=batch_size,
                rec_map_file=rec_map_file,
                lig_map_file=lig_map_file,
                resolution=resolution,
                grid_size=grid_size,
                shuffle=shuffle,
                random_rotation=random_rotation,
                random_translation=random_translation,
                rec_molcache=rec_molcache,
                lig_molcache=lig_molcache,
                device=device
        ) for i in range(2))

        self.train_data.populate(train_file)
        self.test_data.populate(test_file)

        if isinstance(self, GenerativeSolver):

            self.gen_model = self.gen_model_type(
                n_channels_in=self.n_channels_in,
                n_channels_cond=self.n_channels_cond,
                n_channels_out=self.n_channels_out,
                grid_size=grid_size,
                n_filters=n_filters,
                width_factor=width_factor,
                n_levels=n_levels,
                conv_per_level=conv_per_level,
                kernel_size=kernel_size,
                relu_leak=relu_leak,
                pool_type=pool_type,
                unpool_type=unpool_type,
                n_latent=n_latent,
                device=device
            )
            self.gen_optimizer = optim_type(
                self.gen_model.parameters(), **optim_kws
            )
            self.gen_iter = 0

        if isinstance(self, (DiscriminativeSolver, GANSolver)):

            self.disc_model = models.Encoder(
                n_channels=self.n_channels_disc,
                grid_size=grid_size,
                n_filters=n_filters,
                width_factor=width_factor,
                n_levels=n_levels,
                conv_per_level=conv_per_level,
                kernel_size=kernel_size,
                relu_leak=relu_leak,
                pool_type=pool_type,
                pool_factor=pool_factor,
                n_output=1
            ).to(device)

            self.disc_optimizer = optim_type(
                self.disc_model.parameters(), **optim_kws
            )
            self.disc_iter = 0

        self.loss_fns = self.initialize_loss(loss_types or {})
        self.loss_weights = loss_weights or {}

        self.grad_norm_types = self.initialize_norm(grad_norm_types or {})

        if isinstance(self, GenerativeSolver):
            self.atom_fitter = atom_fitter_type(**atom_fitter_kws)

        # set up a data frame of training metrics
        self.metrics = pd.DataFrame(
            columns=self.index_cols
        ).set_index(self.index_cols)

        self.out_prefix = out_prefix

    @property
    def n_channels_in(self):
        return None

    @property
    def n_channels_cond(self):
        return None

    @property
    def n_channels_out(self):
        return None

    @property
    def n_channels_disc(self):
        return None

    @property
    def model(self):
        raise NotImplementedError

    @property
    def optimizer(self):
        raise NotImplementedError

    @property
    def curr_iter(self):
        raise NotImplementedError

    @curr_iter.setter
    def curr_iter(self, i):
        raise NotImplementedError
    
    def set_random_seed(random_seed):
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        molgrid.set_random_seed(random_seed)

    @property
    def state_file(self):
        return '{}_iter_{}.checkpoint'.format(
            self.out_prefix, self.curr_iter
        )

    def save_state(self):

        checkpoint = OrderedDict()
        print('Writing model and optimizer state(s) to ' + self.state_file)

        if hasattr(self, 'gen_model'):
            checkpoint['gen_model_state'] = self.gen_model.state_dict()
            checkpoint['gen_optim_state'] = self.gen_optimizer.state_dict()

        if hasattr(self, 'disc_model'):
            checkpoint['disc_model_state'] = self.disc_model.state_dict()
            checkpoint['disc_optim_state'] = self.disc_optimizer.state_dict()

        torch.save(checkpoint, self.state_file)

    def load_state(self):
        checkpoint = torch.load(self.state_file)

        if hasattr(self, 'gen_model'):
            self.gen_model.load_state_dict(
                checkpoint['gen_model_state']
            )
            self.gen_optimizer.load_state_dict(
                checkpoint['gen_optim_state']
            )

        if hasattr(self, 'disc_model'):
            self.disc_model.load_state_dict(
                checkpoint['disc_model_state']
            )
            self.disc_optimizer.load_state_dict(
                checkpoint['disc_optim_state']
            )

    def save_metrics(self):
        csv_file = self.out_prefix + '.metrics'
        print('Writing training metrics to ' + csv_file)
        self.metrics.to_csv(csv_file, sep=' ')

    def load_metrics(self):
        csv_file = self.out_prefix + '.metrics'
        self.metrics = pd.read_csv(
            csv_file, sep=' '
        ).set_index(self.index_cols)

    def print_metrics(self, idx, metrics):
        index_str = ' '.join(
            '{}={}'.format(*kv) for kv in zip(self.index_cols, idx)
        )
        metrics_str = ' '.join(
            '{}={:.4f}'.format(*kv) for kv in metrics.items()
        )
        print('[{}] {}'.format(index_str, metrics_str))

    def insert_metrics(self, idx, metrics):
        for k, v in metrics.items():
            self.metrics.loc[idx, k] = v

    def forward(self, data):
        raise NotImplementedError

    def initialize_norm(self, grad_norm_types):
        self.gen_grad_norm_type = grad_norm_types.get('gen', '0')
        self.disc_grad_norm_type = grad_norm_types.get('disc', '0')
        assert self.gen_grad_norm_type in {'0', '2'}
        assert self.disc_grad_norm_type in {'0', '2'}

    def test(self, n_batches, fit_atoms=False):

        for i in range(n_batches):
            t_start = time.time()
            loss, metrics = self.forward(self.test_data, fit_atoms)
            metrics['forward_time'] = time.time() - t_start
            self.insert_metrics((self.curr_iter, 'test', i), metrics)

        idx = (self.curr_iter, 'test') # mean across batches
        metrics = self.metrics.loc[idx].mean()
        self.print_metrics(idx, metrics)
        self.save_metrics()
        return metrics

    def step(self, update=True):

        idx = (self.curr_iter, 'train', 0)
        t_start = time.time()
        loss, metrics = self.forward(self.train_data)
        torch.cuda.synchronize()
        metrics['forward_time'] = time.time() - t_start

        if update:
            t_start = time.time()
            self.optimizer.zero_grad()
            loss.backward()
            self.normalize_grad()
            self.optimizer.step()
            torch.cuda.synchronize()
            metrics['backward_time'] = time.time() - t_start
            self.curr_iter += 1

        self.insert_metrics(idx, metrics)
        metrics = self.metrics.loc[idx]
        self.print_metrics(idx[:-1], metrics)
        return metrics

    def train(
        self,
        max_iter,
        test_interval,
        n_test_batches,
        fit_interval,
        n_fit_batches,
        save_interval,
    ):
        while self.curr_iter <= max_iter:

            if self.curr_iter % save_interval == 0:
                self.save_state()

            if self.curr_iter % test_interval == 0:
                self.test(n_test_batches)

            if self.curr_iter < max_iter:
                self.step(update=True)
            else:
                self.step(update=False)
                break

        self.save_state()


class DiscriminativeSolver(Solver):

    @property
    def n_channels_disc(self):
        return self.train_data.n_channels

    @property
    def model(self):
        return self.disc_model

    @property
    def optimizer(self):
        return self.disc_optimizer

    @property
    def curr_iter(self):
        return self.disc_iter

    @curr_iter.setter
    def curr_iter(self, i):
        self.disc_iter = i

    def normalize_grad(self):
        if self.disc_grad_norm_type == '2':
            models.normalize_grad(self.disc_model)

    def forward(self, data):
        complex_grids, _, labels = data.forward()
        predictions = self.disc_model(complex_grids)
        loss, metrics = self.compute_loss(predictions, labels)
        return loss, metrics


class GenerativeSolver(Solver):

    @property
    def n_channels_out(self):
        return self.train_data.n_lig_channels

    @property
    def model(self):
        return self.gen_model

    @property
    def optimizer(self):
        return self.gen_optimizer

    @property
    def curr_iter(self):
        return self.gen_iter

    @curr_iter.setter
    def curr_iter(self, i):
        self.gen_iter = i

    def normalize_grad(self):
        if self.gen_grad_norm_type == '2':
            models.normalize_grad(self.gen_model)


class AESolver(GenerativeSolver):
    gen_model_type = models.AE

    @property
    def n_channels_in(self):
        return self.train_data.n_lig_channels

    def initialize_loss(self, loss_types):
        self.recon_loss_fn = get_recon_loss_fn(
            loss_types.get('recon_loss', '2')
        )

    def compute_loss(self, real_lig_grids, gen_lig_grids):
        recon_loss = self.recon_loss_fn(gen_lig_grids, real_lig_grids)
        loss = self.loss_weights.get('recon_loss', 1.0) * recon_loss
        return loss, OrderedDict([
            ('loss', loss.item()),
            ('recon_loss', recon_loss.item())
        ])

    def forward(self, data, fit_atoms=False):

        lig_grids, lig_structs, _ = data.forward(ligand_only=True)
        lig_gen_grids, latent_vecs = self.gen_model(lig_grids)

        loss, metrics = self.compute_loss(lig_grids, lig_gen_grids)
        metrics.update(compute_paired_grid_metrics(
            'lig_gen', lig_gen_grids, 'lig', lig_grids
        ))
        if fit_atoms:
            lig_gen_fit_structs = self.atom_fitter.fit_batch(
                lig_gen_grids,
                data.lig_channels,
                torch.zeros(3),
                data.resolution
            )
            metrics.update(compute_paired_struct_metrics(
                'lig_gen_fit', lig_gen_fit_structs, 'lig', lig_structs
            ))
    
        return loss, metrics


class VAESolver(AESolver):
    gen_model_type = models.VAE

    def initialize_loss(self, loss_types):
        self.kldiv_loss_fn = kl_divergence
        self.recon_loss_fn = get_recon_loss_fn(
            loss_types.get('recon_loss', '2')
        )

    def compute_loss(self, real_lig_grids, gen_lig_grids, means, log_stds):
        recon_loss = self.recon_loss_fn(gen_lig_grids, real_lig_grids)
        kldiv_loss = self.kldiv_loss_fn(means, log_stds)
        loss = (
            self.loss_weights.get('recon_loss', 1.0) * recon_loss
        ) + (
            self.loss_weights.get('kldiv_loss', 1.0) * kldiv_loss
        )
        return loss, OrderedDict([
            ('loss', loss.item()),
            ('recon_loss', recon_loss.item()),
            ('kldiv_loss', kldiv_loss.item())
        ])

    def forward(self, data, fit_atoms=False):

        lig_grids, lig_structs, _ = data.forward(ligand_only=True)
        lig_gen_grids, latent_vecs, latent_means, latent_log_stds = \
            self.gen_model(lig_grids, data.batch_size)

        loss, metrics = self.compute_loss(
            lig_grids, lig_gen_grids, latent_means, latent_log_stds
        )
        metrics.update(compute_paired_grid_metrics(
            'lig_gen', lig_gen_grids, 'lig', lig_grids
        ))
        if fit_atoms:
            lig_gen_fit_structs = self.atom_fitter.fit_batch(
                lig_gen_grids,
                data.lig_channels,
                torch.zeros(3),
                data.resolution
            )
            metrics.update(compute_paired_struct_metrics(
                'lig_gen_fit', lig_gen_fit_structs, 'lig', lig_structs
            ))
    
        return loss, metrics


class CESolver(AESolver):
    gen_model_type = models.CE

    @property
    def n_channels_in(self):
        return 0

    @property
    def n_channels_cond(self):
        return self.train_data.n_rec_channels

    def forward(self, data, fit_atoms=False):

        (rec_grids, lig_grids), lig_structs, _ = \
            data.forward(split_rec_lig=True)
        lig_gen_grids, latent_vecs = self.gen_model(rec_grids)

        loss, metrics = self.compute_loss(lig_grids, lig_gen_grids)
        metrics.update(compute_paired_grid_metrics(
            'lig_gen', lig_gen_grids, 'lig', lig_grids
        ))
        if fit_atoms:
            lig_gen_fit_structs = self.atom_fitter.fit_batch(
                lig_gen_grids,
                data.lig_channels,
                torch.zeros(3),
                data.resolution
            )
            metrics.update(compute_paired_struct_metrics(
                'lig_gen_fit', lig_gen_fit_structs, 'lig', lig_structs
            ))
    
        return loss, metrics


class CVAESolver(VAESolver):
    gen_model_type = models.CVAE

    @property
    def n_channels_in(self):
        return (
            self.train_data.n_rec_channels + self.train_data.n_lig_channels
        )

    @property
    def n_channels_cond(self):
        return self.train_data.n_rec_channels

    def forward(self, data, fit_atoms=False):

        (rec_grids, lig_grids), lig_structs, _ = \
            data.forward(split_rec_lig=True)
        rec_lig_grids = data.grids

        lig_gen_grids, latent_vecs, latent_means, latent_log_stds = \
            self.gen_model(rec_lig_grids, rec_grids, data.batch_size)

        loss, metrics = self.compute_loss(
            lig_grids, lig_gen_grids, latent_means, latent_log_stds
        )
        metrics.update(compute_paired_grid_metrics(
            'lig_gen', lig_gen_grids, 'lig', lig_grids
        ))
        if fit_atoms:
            lig_gen_fit_structs = self.atom_fitter.fit_batch(
                lig_gen_grids,
                data.lig_channels,
                torch.zeros(3),
                data.resolution
            )
            metrics.update(compute_paired_struct_metrics(
                'lig_gen_fit', lig_gen_fit_structs, 'lig', lig_structs
            ))
    
        return loss, metrics


class GANSolver(GenerativeSolver):
    gen_model_type = models.GAN
    index_cols = ['iteration', 'disc_iter', 'phase', 'model', 'batch', 'real']

    @property
    def n_channels_disc(self):
        return self.train_data.n_lig_channels

    def initialize_loss(self, loss_types):
        self.gan_loss_fn = get_gan_loss_fn(
            loss_types.get('gan_loss', 'x')
        )

    def normalize_gen_grad(self):
        if self.gen_grad_norm_type == '2':
            models.normalize_grad(self.gen_model)

    def normalize_disc_grad(self):
        if self.disc_grad_norm_type == '2':
            models.normalize_grad(self.disc_model)

    def compute_loss(self, labels, predictions):
        gan_loss = self.gan_loss_fn(predictions, labels)
        loss = self.loss_weights.get('gan_loss', 1.0) * gan_loss
        return loss, OrderedDict([
            ('loss', loss.item()),
            ('gan_loss', gan_loss.item())
        ])

    def disc_forward(self, data, real):
        '''
        Compute predictions and loss for the discriminator's
        ability to correctly classify real or generated data.
        '''
        with torch.no_grad(): # do not backprop to generator or data

            if real: # get real examples
                lig_grids, lig_structs, _ = data.forward(ligand_only=True)
                labels = torch.ones(data.batch_size, 1, device=self.device)

            else: # get generated examples
                lig_gen_grids, _ = self.gen_model(data.batch_size)
                labels = torch.zeros(data.batch_size, 1, device=self.device)
                lig_grids = lig_gen_grids

        predictions, _ = self.disc_model(lig_grids)
        loss, metrics = self.compute_loss(labels, predictions)
        metrics.update(compute_grid_metrics(
            'lig' if real else 'lig_gen', lig_grids
        ))
        return loss, metrics

    def gen_forward(self, data, fit_atoms=False):
        '''
        Compute predictions and loss for the generator's ability
        to produce data that is misclassified by the discriminator.
        '''
        # get generated examples
        lig_gen_grids, latent_vecs = self.gen_model(data.batch_size)
        labels = torch.ones(data.batch_size, 1, device=self.device)

        predictions, _ = self.disc_model(lig_gen_grids)
        loss, metrics = self.compute_loss(labels, predictions)
        metrics.update(compute_grid_metrics('lig_gen', lig_gen_grids))

        if fit_atoms:
            lig_gen_fit_structs = self.atom_fitter.fit_batch(
                lig_gen_grids,
                data.lig_channels,
                torch.zeros(3),
                data.resolution
            )
            metrics.update(compute_struct_metrics(
                'lig_gen_fit', lig_gen_fit_structs
            ))

        return loss, metrics

    def test(self, n_batches, fit_atoms=False):

        # test discriminator on alternating real/generated batches
        for i in range(n_batches):
            real = (i%2 == 0)
            t_start = time.time()
            loss, metrics = self.disc_forward(self.test_data, real)
            metrics['forward_time'] = time.time() - t_start
            idx = (self.gen_iter, self.disc_iter, 'test', 'disc', i, real)
            self.insert_metrics(idx, metrics)

        idx = (self.gen_iter, self.disc_iter, 'test', 'disc')
        metrics = self.metrics.loc[idx].mean()
        self.print_metrics(idx, metrics)

        # test generator on same number of batches
        for i in range(n_batches):
            t_start = time.time()
            loss, metrics = self.gen_forward(self.test_data, fit_atoms)
            metrics['forward_time'] = time.time() - t_start
            idx = (self.gen_iter, self.disc_iter, 'test', 'gen', i, False)
            self.insert_metrics(idx, metrics)

        idx = (self.gen_iter, self.disc_iter, 'test', 'gen')
        metrics = self.metrics.loc[idx].mean()
        self.print_metrics(idx, metrics)

        self.save_metrics()
        return self.metrics.loc[(self.gen_iter, self.disc_iter, 'test')]

    def disc_step(self, real, update=True, batch_idx=0):

        idx = (
            self.gen_iter, self.disc_iter, 'train', 'disc', batch_idx, real
        )
        t_start = time.time()
        loss, metrics = self.disc_forward(self.train_data, real)
        torch.cuda.synchronize()
        metrics['forward_time'] = time.time() - t_start
        
        if update:
            t_start = time.time()
            self.disc_optimizer.zero_grad()
            loss.backward()
            self.normalize_disc_grad()
            metrics['disc_grad_norm'] = models.compute_grad_norm(self.disc_model)
            self.disc_optimizer.step()
            torch.cuda.synchronize()
            metrics['backward_time'] = time.time() - t_start
            self.disc_iter += 1

        self.insert_metrics(idx, metrics)
        metrics = self.metrics.loc[idx]
        self.print_metrics(idx[:-1], metrics)
        return metrics

    def gen_step(self, update=True, batch_idx=0):

        idx = (
            self.gen_iter, self.disc_iter, 'train', 'gen', batch_idx, False
        )
        t_start = time.time()
        loss, metrics = self.gen_forward(self.train_data)
        torch.cuda.synchronize()
        metrics['forward_time'] = time.time() - t_start

        if update:
            t_start = time.time()
            self.gen_optimizer.zero_grad()
            loss.backward()
            self.normalize_gen_grad()
            metrics['gen_grad_norm'] = models.compute_grad_norm(self.gen_model)
            self.gen_optimizer.step()
            torch.cuda.synchronize()
            metrics['backward_time'] = time.time() - t_start
            self.gen_iter += 1

        self.insert_metrics(idx, metrics)
        metrics = self.metrics.loc[idx]
        self.print_metrics(idx[:-1], metrics)
        return metrics

    def train_disc(self, n_iters, update=True):

        for i in range(n_iters):
            real = (i%2 == 0)
            batch_idx = 0 if update else i
            self.disc_step(real, update, batch_idx)

    def train_gen(self, n_iters, update=True):
        for i in range(n_iters):
            batch_idx = 0 if update else i
            self.gen_step(update, batch_idx)
 
    def train(
        self,
        max_iter,
        n_gen_train_iters,
        n_disc_train_iters,
        test_interval,
        n_test_batches,
        fit_interval,
        n_fit_batches,
        save_interval,
    ):
        while self.curr_iter <= max_iter:

            if self.curr_iter % save_interval == 0:
                self.save_state()

            if self.curr_iter % test_interval == 0:
                self.test(n_test_batches)

            if self.curr_iter < max_iter:
                self.train_disc(n_disc_train_iters, update=True)
                self.train_gen(n_gen_train_iters, update=True)
            else:
                self.train_disc(n_disc_train_iters, update=False)
                self.train_gen(n_gen_train_iters, update=False)
                break

        self.save_state()


class CGANSolver(GANSolver):
    gen_model_type = models.CGAN

    @property
    def n_channels_disc(self):
        return (
            self.train_data.n_rec_channels + self.train_data.n_lig_channels
        )

    @property
    def n_channels_cond(self):
        return self.train_data.n_rec_channels

    def disc_forward(self, data, real):

        with torch.no_grad(): # do not backprop to generator or data

            # get real examples
            (rec_grids, lig_grids), lig_structs, _ = \
                data.forward(split_rec_lig=True)
            if real:
                labels = torch.ones(data.batch_size, 1, device=self.device)

            else: # get generated examples
                lig_gen_grids, _ = self.gen_model(rec_grids, data.batch_size)
                labels = torch.zeros(data.batch_size, 1, device=self.device)
                lig_grids = lig_gen_grids

        rec_lig_grids = torch.cat([rec_grids, lig_grids], dim=1)
        predictions, _ = self.disc_model(rec_lig_grids)
        loss, metrics = self.compute_loss(labels, predictions)
        metrics.update(compute_grid_metrics(
            'lig' if real else 'lig_gen', lig_grids
        ))
        return loss, metrics

    def gen_forward(self, data, fit_atoms=False):

        # get generated examples
        (rec_grids, lig_grids), lig_structs, _ = \
            data.forward(split_rec_lig=True)

        lig_gen_grids, latent_vecs = self.gen_model(
            rec_grids, data.batch_size
        )
        labels = torch.ones(data.batch_size, 1, device=self.device)

        rec_lig_grids = torch.cat([rec_grids, lig_gen_grids], dim=1)
        predictions, _ = self.disc_model(rec_lig_grids)
        loss, metrics = self.compute_loss(labels, predictions)
        metrics.update(compute_grid_metrics('lig_gen', lig_gen_grids))

        if fit_atoms:
            lig_gen_fit_structs = self.atom_fitter.fit_batch(
                lig_gen_grids,
                data.lig_channels,
                torch.zeros(3),
                data.resolution
            )
            metrics.update(compute_struct_metrics(
                'lig_gen_fit', lig_gen_fit_structs
            ))

        return loss, metrics


class VAEGANSolver(GANSolver):
    gen_model_type = models.VAE

    @property
    def n_channels_in(self):
        return self.train_data.n_lig_channels

    def initialize_loss(self, loss_types):
        self.kldiv_loss_fn = kl_divergence
        self.recon_loss_fn = get_recon_loss_fn(
            loss_types.get('recon_loss', '2')
        )
        self.gan_loss_fn = get_gan_loss_fn(
            loss_types.get('gan_loss', 'x')
        )

    def compute_loss(
        self, labels, predictions, real_lig_grids, gen_lig_grids, means, log_stds
    ):
        gan_loss = self.gan_loss_fn(predictions, labels)
        recon_loss = self.recon_loss_fn(gen_lig_grids, real_lig_grids)
        kldiv_loss = self.kldiv_loss_fn(means, log_stds)
        loss = (
            self.loss_weights.get('gan_loss', 1.0) * gan_loss
        ) + (
            self.loss_weights.get('recon_loss', 1.0) * recon_loss
        ) + (
            self.loss_weights.get('kldiv_loss', 1.0) * kldiv_loss
        )
        return loss, OrderedDict([
            ('loss', loss.item()),
            ('gan_loss', gan_loss.item()),
            ('recon_loss', recon_loss.item()),
            ('kldiv_loss', kldiv_loss.item()),
        ])

    def disc_forward(self, data, real):

        with torch.no_grad(): # do not backprop to generator or data

            # get real examples
            lig_grids, lig_structs, _ = data.forward(ligand_only=True)

            if real:
                labels = torch.ones(data.batch_size, 1, device=self.device)

            else: # get generated examples
                lig_gen_grids, latent_vecs, latent_means, latent_log_stds = \
                    self.gen_model(lig_grids, data.batch_size)
                labels = torch.zeros(data.batch_size, 1, device=self.device)

        predictions, _ = self.disc_model(lig_grids if real else lig_gen_grids)

        if real:
            loss, metrics = GANSolver.compute_loss(
                self, labels, predictions
            )
        else:
            loss, metrics = self.compute_loss(
                labels,
                predictions,
                lig_grids,
                lig_gen_grids,
                latent_means,
                latent_log_stds
            )
        metrics.update(compute_grid_metrics(
            'lig' if real else 'lig_gen',
            lig_grids if real else lig_gen_grids
        ))
        return loss, metrics

    def gen_forward(self, data, fit_atoms=False):

        # get generated examples
        lig_grids, lig_structs, _ = data.forward(ligand_only=True)

        lig_gen_grids, latent_vecs, latent_means, latent_log_stds = \
            self.gen_model(lig_grids, data.batch_size)
        labels = torch.ones(data.batch_size, 1, device=self.device)

        predictions, _ = self.disc_model(lig_gen_grids)
        loss, metrics = self.compute_loss(
            labels,
            predictions,
            lig_grids,
            lig_gen_grids,
            latent_means,
            latent_log_stds
        )
        metrics.update(compute_paired_grid_metrics(
            'lig_gen', lig_gen_grids, 'lig', lig_grids
        ))

        if fit_atoms:
            lig_gen_fit_structs = self.atom_fitter.fit_batch(
                lig_gen_grids,
                data.lig_channels,
                torch.zeros(3),
                data.resolution
            )
            metrics.update(compute_paired_struct_metrics(
                'lig_gen_fit', lig_gen_fit_structs, 'lig', lig_structs
            ))

        return loss, metrics


class CVAEGANSolver(VAEGANSolver):
    gen_model_type = models.CVAE

    @property
    def n_channels_in(self):
        return (
            self.train_data.n_rec_channels + self.train_data.n_lig_channels
        )

    @property
    def n_channels_cond(self):
        return self.train_data.n_rec_channels

    @property
    def n_channels_disc(self):
        return (
            self.train_data.n_rec_channels + self.train_data.n_lig_channels
        )

    def disc_forward(self, data, real):

        with torch.no_grad(): # do not backprop to generator or data

            # get real examples
            (rec_grids, lig_grids), lig_structs, _ = \
                data.forward(split_rec_lig=True)
            rec_lig_grids = data.grids

            if real:
                labels = torch.ones(data.batch_size, 1, device=self.device)

            else: # get generated examples
                lig_gen_grids, latent_vecs, latent_means, latent_log_stds = \
                    self.gen_model(rec_lig_grids, rec_grids, data.batch_size)
                labels = torch.zeros(data.batch_size, 1, device=self.device)
                rec_lig_grids = torch.cat([rec_grids, lig_gen_grids], dim=1)

        predictions, _ = self.disc_model(rec_lig_grids)

        if real:
            loss, metrics = GANSolver.compute_loss(
                self, labels, predictions
            )
        else:
            loss, metrics = self.compute_loss(
                labels,
                predictions,
                lig_grids,
                lig_gen_grids,
                latent_means,
                latent_log_stds
            )
        metrics.update(compute_grid_metrics(
            'lig' if real else 'lig_gen',
            lig_grids if real else lig_gen_grids
        ))
        return loss, metrics

    def gen_forward(self, data, fit_atoms=False):

        # get generated examples
        (rec_grids, lig_grids), lig_structs, _ = \
            data.forward(split_rec_lig=True)
        rec_lig_grids = data.grids

        lig_gen_grids, latent_vecs, latent_means, latent_log_stds = \
            self.gen_model(rec_lig_grids, rec_grids, data.batch_size)
        labels = torch.ones(data.batch_size, 1, device=self.device)

        rec_lig_grids = torch.cat([rec_grids, lig_gen_grids], dim=1)
        predictions, _ = self.disc_model(rec_lig_grids)
        loss, metrics = self.compute_loss(
            labels,
            predictions,
            lig_grids,
            lig_gen_grids,
            latent_means,
            latent_log_stds
        )
        metrics.update(compute_paired_grid_metrics(
            'lig_gen', lig_gen_grids, 'lig', lig_grids
        ))

        if fit_atoms:
            lig_gen_fit_structs = self.atom_fitter.fit_batch(
                lig_gen_grids,
                data.lig_channels,
                torch.zeros(3),
                data.resolution
            )
            metrics.update(compute_struct_metrics(
                'lig_gen_fit', lig_gen_fit_structs
            ))

        return loss, metrics
