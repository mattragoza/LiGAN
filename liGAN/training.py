import time
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from torch import nn
torch.backends.cudnn.benchmark = True

import molgrid
from . import data, models


def kl_divergence(means, log_stds):
    stds = torch.exp(log_stds)
    means2 = means * means
    vars = stds * stds
    return (-log_stds + means2/2 + vars/2 - 1/2).sum() / means.shape[0]


def wasserstein_loss(predicted, labels):
    return (2*labels - 1) * predicted


def get_recon_loss_fn(loss_type='2'):
    assert loss_type in {'1', '2'}
    if loss_type == '1':
        return torch.nn.L1Loss()
    else:
        return torch.nn.MSELoss()


def get_gan_loss_fn(loss_type='x'):
    assert loss_type in {'x', 'w'}
    if loss_type == 'w':
        return wasserstein_loss
    else:
        return torch.nn.BCEWithLogitsLoss()


class Solver(nn.Module):

    split_rec_lig = False
    ligand_only = False
    generative = False
    variational = False
    adversarial = False
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
        loss_weights,
        loss_types,
        grad_norms,
        optim_type,
        optim_kws,
        save_prefix,
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
                split_rec_lig=self.split_rec_lig,
                ligand_only=self.ligand_only,
                rec_molcache=rec_molcache,
                lig_molcache=lig_molcache,
                device=device
        ) for i in range(2))

        self.train_data.populate(train_file)
        self.test_data.populate(test_file)

        if self.generative:

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
                variational=self.variational,
                device=device
            )
            self.gen_optimizer = optim_type(
                self.gen_model.parameters(), **optim_kws
            )
            self.gen_iter = 0

        if not self.generative or self.adversarial:

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

        self.grad_norms = self.initialize_norm(grad_norms or {})

        # set up a data frame of training metrics
        self.metrics = pd.DataFrame(
            columns=self.index_cols
        ).set_index(self.index_cols)

        self.save_prefix = save_prefix

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
            self.save_prefix, self.curr_iter
        )

    def save_state(self):
        checkpoint = OrderedDict()

        if self.generative:
            checkpoint['gen_model_state'] = self.gen_model.state_dict()
            checkpoint['gen_optim_state'] = self.gen_optimizer.state_dict()

        if not self.generative or self.adversarial:
            checkpoint['disc_model_state'] = self.disc_model.state_dict()
            checkpoint['disc_optim_state'] = self.disc_optimizer.state_dict()

        torch.save(checkpoint, self.state_file)

    def load_state(self):
        checkpoint = torch.load(self.state_file)

        if self.generative:
            self.gen_model.load_state_dict(
                checkpoint['gen_model_state']
            )
            self.gen_optimizer.load_state_dict(
                checkpoint['gen_optim_state']
            )

        if not self.generative or self.adversarial:
            self.disc_model.load_state_dict(
                checkpoint['disc_model_state']
            )
            self.disc_optimizer.load_state_dict(
                checkpoint['disc_optim_state']
            )

    def save_metrics(self):
        csv_file = self.save_prefix + '.metrics'
        self.metrics.to_csv(csv_file, sep=' ')

    def load_metrics(self):
        csv_file = self.save_prefix + '.metrics'
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

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError

    def compute_metrics(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, data):
        raise NotImplementedError

    def initialize_norm(self, grad_norms):
        self.gen_grad_norm = grad_norms.get('gen', '0')
        self.disc_grad_norm = grad_norms.get('disc', '0')
        assert self.gen_grad_norm in {'0', '2', 's'}
        assert self.disc_grad_norm in {'0', '2', 's'}

    def test(self, n_batches):

        for i in range(n_batches):
            t_start = time.time()
            predictions, loss, metrics = self.forward(self.test_data)
            metrics['forward_time'] = time.time() - t_start
            self.insert_metrics((self.curr_iter, 'test', i), metrics)

        idx = (self.curr_iter, 'test') # batch mean
        metrics = self.metrics.loc[idx].mean()
        self.print_metrics(idx, metrics)
        self.save_metrics()
        return metrics

    def step(self, update=True):

        idx = (self.curr_iter, 'train', 0)
        t_start = time.time()
        predictions, loss, metrics = self.forward(self.train_data)
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
        if self.disc_grad_norm == '2':
            models.normalize_grad(self.disc_model)

    def compute_metrics(self, labels, predictions):
        metrics = OrderedDict()
        metrics['true_norm'] = labels.detach().norm().item()
        metrics['pred_norm'] = predictions.detach().norm().item()
        metrics['grad_norm'] = models.compute_grad_norm(self.model)
        return metrics

    def forward(self, data):
        inputs, labels = data.forward()
        predictions = self.disc_model(inputs)
        loss, metrics = self.compute_loss(labels, predictions)
        metrics.update(self.compute_metrics(labels, predictions))
        return predictions, loss, metrics


class GenerativeSolver(Solver):
    generative = True
    split_rec_lig = True

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
        if self.gen_grad_norm == '2':
            models.normalize_grad(self.gen_model)

    def compute_metrics(self, inputs, generated, latents):
        metrics = OrderedDict()
        metrics['lig_norm'] = inputs.detach().norm().item()
        metrics['lig_gen_norm'] = generated.detach().norm().item()
        metrics['latent_norm'] = latents.detach().norm().item()
        metrics['gen_grad_norm'] = models.compute_grad_norm(self.model)
        return metrics


class AESolver(GenerativeSolver):
    ligand_only = True
    gen_model_type = models.AE

    @property
    def n_channels_in(self):
        return self.train_data.n_lig_channels

    def initialize_loss(self, loss_types):
        self.recon_loss_fn = get_recon_loss_fn(
            loss_types.get('recon_loss', '2')
        )

    def compute_loss(self, inputs, generated):
        recon_loss = self.recon_loss_fn(inputs, generated)
        return recon_loss, OrderedDict([
            ('loss', recon_loss.item()),
            ('recon_loss', recon_loss.item())
        ])

    def forward(self, data):
        inputs, _ = data.forward()
        generated, latents = self.gen_model(inputs)
        loss, metrics = self.compute_loss(inputs, generated)
        metrics.update(self.compute_metrics(inputs, generated, latents))
        return generated, loss, metrics


class VAESolver(AESolver):
    variational = True
    gen_model_type = models.VAE

    def initialize_loss(self, loss_types):
        self.kldiv_loss_fn = kl_divergence
        self.recon_loss_fn = get_recon_loss_fn(
            loss_types.get('recon_loss', '2')
        )

    def compute_loss(self, inputs, generated, means, log_stds):
        recon_loss = self.recon_loss_fn(generated, inputs)
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

    def forward(self, data):
        inputs, _ = data.forward()
        generated, latents, means, log_stds = self.gen_model(
            inputs, data.batch_size
        )
        loss, metrics = self.compute_loss(inputs, generated, means, log_stds)
        metrics.update(self.compute_metrics(inputs, generated, latents))
        return generated, loss, metrics


class CESolver(GenerativeSolver):
    ligand_only = False
    gen_model_type = models.CE

    @property
    def n_channels_cond(self):
        return self.train_data.n_rec_channels

    def initialize_loss(self, loss_types):
        self.recon_loss_fn = get_recon_loss_fn(
            loss_types.get('recon_loss', '2')
        )

    def compute_loss(self, inputs, generated):
        recon_loss = self.recon_loss_fn(generated, inputs)
        return recon_loss, OrderedDict([
            ('loss', recon_loss.item()),
            ('recon_loss', recon_loss.item())
        ])

    def forward(self, data):
        (context, missing), _ = data.forward()
        generated, latents = self.gen_model(context)
        loss, metrics = self.compute_loss(missing, generated)
        metrics.update(self.compute_metrics(missing, generated, latents))
        return generated, loss, metrics


class CVAESolver(VAESolver):
    ligand_only = False
    gen_model_type = models.CVAE

    @property
    def n_channels_in(self):
        return (
            self.train_data.n_rec_channels + self.train_data.n_lig_channels
        )

    @property
    def n_channels_cond(self):
        return self.train_data.n_rec_channels

    def forward(self, data):
        (conditions, real), _ = data.forward()
        inputs = data.grids
        generated, latents, means, log_stds = self.gen_model(
            inputs, conditions, data.batch_size
        )
        loss, metrics = self.compute_loss(real, generated, means, log_stds)
        metrics.update(self.compute_metrics(real, generated, latents))
        return generated, loss, metrics


class GANSolver(GenerativeSolver):
    ligand_only = True
    variational = True
    adversarial = True
    gen_model_type = models.GAN
    index_cols = ['gen_iter', 'disc_iter', 'phase', 'model', 'batch', 'real']

    @property
    def n_channels_disc(self):
        return self.train_data.n_lig_channels

    def initialize_loss(self, loss_types):
        self.gan_loss_fn = get_gan_loss_fn(
            loss_types.get('gan_loss', 'x')
        )

    def normalize_grad(self):
        if self.gen_grad_norm == '2':
            models.normalize_grad(self.gen_model)
        if self.disc_grad_norm == '2':
            models.normalize_grad(self.disc_model)

    def compute_loss(self, inputs, generated):
        gan_loss = self.gan_loss_fn(generated, inputs)
        return gan_loss, OrderedDict([
            ('loss', gan_loss.item()),
            ('gan_loss', gan_loss.item())
        ])

    def compute_metrics(self, inputs):
        metrics = OrderedDict()
        metrics['lig_norm'] = inputs.detach().norm().item()
        metrics['disc_grad_norm'] = models.compute_grad_norm(self.disc_model)
        metrics['gen_grad_norm'] = models.compute_grad_norm(self.gen_model)
        return metrics

    def disc_forward(self, data, real):
        '''
        Compute predictions and loss for the discriminator's
        ability to correctly classify real or generated data.
        '''
        with torch.no_grad(): # do not backprop to generator or data

            if real: # get real examples
                inputs, _ = data.forward()
                labels = torch.ones(data.batch_size, 1, device=self.device)

            else: # get generated examples
                inputs, latents = self.gen_model(data.batch_size)
                labels = torch.zeros(data.batch_size, 1, device=self.device)

        predictions = self.disc_model(inputs)
        loss, metrics = self.compute_loss(labels, predictions)
        metrics.update(self.compute_metrics(inputs))
        return predictions, loss, metrics

    def gen_forward(self, data):
        '''
        Compute predictions and loss for the generator's ability
        to produce data that is misclassified by the discriminator.
        '''
        # get generated examples
        inputs, latents = self.gen_model(data.batch_size)
        labels = torch.ones(data.batch_size, 1, device=self.device)

        predictions = self.disc_model(inputs)
        loss, metrics = self.compute_loss(labels, predictions)
        metrics.update(self.compute_metrics(inputs))
        return predictions, loss, metrics

    def test(self, n_batches):

        # test discriminator on alternating real/generated batches
        for i in range(n_batches):
            real = (i%2 == 0)
            t_start = time.time()
            predictions, loss, metrics = self.disc_forward(
                self.test_data, real
            )
            metrics['forward_time'] = time.time() - t_start
            idx = (self.gen_iter, self.disc_iter, 'test', 'disc', i, real)
            self.insert_metrics(idx, metrics)

        idx = (self.gen_iter, self.disc_iter, 'test', 'disc') # batch mean
        metrics = self.metrics.loc[idx].mean()
        self.print_metrics(idx, metrics)

        # test generator on same number of batches
        for i in range(n_batches):
            t_start = time.time()
            predictions, loss, metrics = self.gen_forward(
                self.test_data
            )
            metrics['forward_time'] = time.time() - t_start
            idx = (self.gen_iter, self.disc_iter, 'test', 'gen', i, False)
            self.insert_metrics(idx, metrics)

        idx = (self.gen_iter, self.disc_iter, 'test', 'gen') # batch mean
        metrics = self.metrics.loc[idx].mean()
        self.print_metrics(idx, metrics)

        self.save_metrics()
        return self.metrics.loc[(self.gen_iter, self.disc_iter, 'test')]

    def disc_step(self, real, update=True, batch_idx=0):

        idx = (
            self.gen_iter, self.disc_iter, 'train', 'disc', batch_idx, real
        )
        t_start = time.time()
        predictions, loss, metrics = self.disc_forward(self.train_data, real)
        torch.cuda.synchronize()
        metrics['forward_time'] = time.time() - t_start
        
        if update:
            t_start = time.time()
            self.disc_optimizer.zero_grad()
            loss.backward()
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
        predictions, loss, metrics = self.gen_forward(self.train_data)
        torch.cuda.synchronize()
        metrics['forward_time'] = time.time() - t_start

        if update:
            t_start = time.time()
            self.gen_optimizer.zero_grad()
            loss.backward()
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
    ligand_only = False
    split_rec_lig = True
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
        '''
        Compute predictions and loss for the discriminator's
        ability to correctly classify real or generated data.
        '''
        with torch.no_grad(): # do not backprop to generator or data

            # get real examples
            (conditions, inputs), _ = data.forward()

            if real:
                labels = torch.ones(data.batch_size, 1, device=self.device)

            else: # get generated examples
                inputs, latents = self.gen_model(conditions, data.batch_size)
                labels = torch.zeros(data.batch_size, 1, device=self.device)

        predictions = self.disc_model(torch.cat([conditions, inputs], dim=1))
        loss, metrics = self.compute_loss(labels, predictions)
        metrics.update(self.compute_metrics(inputs))
        return predictions, loss, metrics

    def gen_forward(self, data):
        '''
        Compute predictions and loss for the generator's ability
        to produce data that is misclassified by the discriminator.
        '''
        # get generated examples
        (conditions, inputs), _ = data.forward()
        inputs, latents = self.gen_model(conditions, data.batch_size)
        labels = torch.ones(data.batch_size, 1, device=self.device)

        predictions = self.disc_model(torch.cat([conditions, inputs], dim=1))
        loss, metrics = self.compute_loss(labels, predictions)
        metrics.update(self.compute_metrics(inputs))
        return predictions, loss, metrics


class VAEGANSolver(GANSolver):
    pass


class CVAEGANSolver(GANSolver):
    pass

