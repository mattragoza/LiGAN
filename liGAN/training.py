import os, time, psutil, pynvml, re, glob
from collections import OrderedDict
from functools import partial
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
torch.backends.cudnn.benchmark = True

import molgrid
from . import data, models, atom_types, atom_fitting, molecules
from .metrics import (
    compute_scalar_metrics,
    compute_grid_metrics,
    compute_paired_grid_metrics,
    compute_struct_metrics,
    compute_paired_struct_metrics
)


MB = 1024 ** 2

def get_memory_used():
    return psutil.Process(os.getpid()).memory_info().rss


def kl_divergence(means, log_stds):
    stds = torch.exp(log_stds)
    means2 = means * means
    vars_ = stds * stds
    return (
        -log_stds + means2/2 + vars_/2 - 0.5
    ).sum() / means.shape[0]


def wasserstein_loss(predictions, labels):
    labels = (2*labels - 1) # convert {0, 1} to {-1, 1}
    return (labels * predictions).sum() / labels.shape[0]


def L1_loss(predictions, labels):
    return (labels - predictions).abs().sum() / labels.shape[0]


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


def save_on_exception(method):
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except:
            self.save_state()
            raise
    return wrapper


def find_last_iter(out_prefix, min_iter=-1):
    last_iter = min_iter
    state_file_re = re.compile(out_prefix + r'_iter_(\d+)')
    for state_file in glob.glob(out_prefix + '_iter_*'):
        m = state_file_re.match(state_file)
        last_iter = max(last_iter, int(m.group(1)))
    if last_iter > min_iter:
        return last_iter
    else:
        raise FileNotFoundError('could not find state files')


class Solver(nn.Module):
    gen_model_type = None
    index_cols = ['iteration', 'phase', 'batch']

    def __init__(
        self,
        train_file,
        test_file,
        data_kws,
        gen_model_kws,
        disc_model_kws,
        loss_fn_kws,
        gen_optim_kws,
        disc_optim_kws,
        atom_fitting_kws,
        out_prefix,
        random_seed=None,
        caffe_init=False,
        device='cuda',
        debug=False,
    ):
        super().__init__()
        self.device = device

        if random_seed is not None:
            self.set_random_seed(random_seed)

        print('loading data')
        self.train_data, self.test_data = (
            data.AtomGridData(device=device, **data_kws) for i in range(2)
        )
        self.train_data.populate(train_file)
        self.test_data.populate(test_file)

        if isinstance(self, GenerativeSolver):

            print('creating generative model and optimizer')
            self.gen_model = self.gen_model_type(
                n_channels_in=self.n_channels_in,
                n_channels_cond=self.n_channels_cond,
                n_channels_out=self.n_channels_out,
                grid_size=self.train_data.grid_size,
                device=device,
                **gen_model_kws
            )
            gen_optim_type = getattr(optim, gen_optim_kws.pop('type'))
            self.n_gen_train_iters = gen_optim_kws.pop('n_train_iters', 2)
            self.gen_clip_grad = gen_optim_kws.pop('clip_gradient', 0)
            self.gen_optimizer = gen_optim_type(
                self.gen_model.parameters(), **gen_optim_kws
            )
            self.gen_iter = 0

        if isinstance(self, (DiscriminativeSolver, GANSolver)):

            print('creating discriminative model and optimizer')
            self.disc_model = models.Encoder(
                n_channels=self.n_channels_disc,
                grid_size=self.train_data.grid_size,
                **disc_model_kws
            ).to(device)

            disc_optim_type = getattr(optim, disc_optim_kws.pop('type'))
            self.n_disc_train_iters = disc_optim_kws.pop('n_train_iters', 2)
            self.disc_clip_grad = disc_optim_kws.pop('clip_gradient', 0)
            self.disc_optimizer = disc_optim_type(
                self.disc_model.parameters(), **disc_optim_kws
            )
            self.disc_iter = 0

        self.loss_fns = self.initialize_loss(loss_fn_kws.get('types', {}))
        self.loss_weights = loss_fn_kws.get('weights', {})

        self.initialize_weights(caffe=caffe_init)

        if isinstance(self, GenerativeSolver):
            self.atom_fitter = atom_fitting.AtomFitter(
                device=device, debug=debug, **atom_fitting_kws
            )

        # set up a data frame of training metrics
        self.metrics = pd.DataFrame(
            columns=self.index_cols
        ).set_index(self.index_cols)

        self.out_prefix = out_prefix
        self.debug = debug

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

    @property
    def state_prefix(self):
        return '{}_iter_{}'.format(self.out_prefix, self.curr_iter)
    
    def set_random_seed(self, random_seed):
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        molgrid.set_random_seed(random_seed)

    def save_state(self):

        if hasattr(self, 'gen_model'):

            state_file = self.state_prefix + '.gen_model_state'
            print('Saving generative model state to ' + state_file)
            torch.save(self.gen_model.state_dict(), state_file)

            state_file = self.state_prefix + '.gen_solver_state'
            print('Saving generative solver state to ' + state_file)
            state_dict = OrderedDict()
            state_dict['optim_state'] = self.gen_optimizer.state_dict() 
            state_dict['iter'] = self.gen_iter
            torch.save(state_dict, state_file)

        if hasattr(self, 'disc_model'):

            state_file = self.state_prefix + '.disc_model_state'
            print('Saving discriminative model state to ' + state_file)
            torch.save(self.disc_model.state_dict(), state_file)

            state_file = self.state_prefix + '.disc_solver_state'
            print('Saving discriminative solver state to ' + state_file)
            state_dict = OrderedDict()
            state_dict['optim_state'] = self.disc_optimizer.state_dict() 
            state_dict['iter'] = self.disc_iter
            torch.save(state_dict, state_file)

    def load_state(self, cont_iter=None):
        self.curr_iter = cont_iter if cont_iter else self.find_last_iter()

        if hasattr(self, 'gen_model'):

            state_file = self.state_prefix + '.gen_model_state'
            print('Loading generative model state from ' + state_file)
            self.gen_model.load_state_dict(torch.load(state_file))

            state_file = self.state_prefix + '.gen_solver_state'
            print('Loading generative solver state from ' + state_file)
            state_dict = torch.load(state_file)
            self.gen_optimizer.load_state_dict(state_dict['optim_state'])
            self.gen_iter = state_dict['iter']

        if hasattr(self, 'disc_model'):

            state_file = self.state_prefix + '.disc_model_state'
            print('Loading discriminative model state from ' + state_file)
            self.disc_model.load_state_dict(torch.load(state_file))

            state_file = self.state_prefix + '.disc_solver_state'
            print('Loading discriminative solver state from ' + state_file)
            state_dict = torch.load(state_file)
            self.disc_optimizer.load_state_dict(state_dict['optim_state'])
            self.disc_iter = state_dict['iter']

    def find_last_iter(self):
        return find_last_iter(self.out_prefix)

    def save_metrics(self):
        csv_file = self.out_prefix + '.train_metrics'
        print('Writing training metrics to ' + csv_file)
        self.metrics.to_csv(csv_file, sep=' ')

    def load_metrics(self):
        csv_file = self.out_prefix + '.train_metrics'
        print('Reading training metrics from ' + csv_file)
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
        print('[{}] {}'.format(index_str, metrics_str), flush=True)

    def insert_metrics(self, idx, metrics):
        for k, v in metrics.items():
            self.metrics.loc[idx, k] = v

    def save_structs(self, structs):
        sdf_file = '{}_iter_{}.sdf'.format(
            self.out_prefix, self.curr_iter
        )
        print('Writing generated molecules to ' + sdf_file)
        molecules.write_rd_mols_to_sdf_file(
            sdf_file, (s.info['add_mol'] for s in structs), kekulize=False
        )

    def initialize_weights(self, caffe):
        if hasattr(self, 'gen_model'):
            self.gen_model.apply(
                partial(models.initialize_weights, caffe=caffe)
            )
        if hasattr(self, 'disc_model'):
            self.disc_model.apply(
                partial(models.initialize_weights, caffe=caffe)
            )

    def forward(self, data):
        raise NotImplementedError

    @save_on_exception
    def test(self, n_batches, fit_atoms=False):

        for i in range(n_batches):
            torch.cuda.reset_max_memory_allocated()
            t0 = time.time()
            loss, metrics = self.forward(self.test_data, fit_atoms)
            metrics['forward_time'] = time.time() - t0
            metrics['forward_gpu'] = torch.cuda.max_memory_allocated() / MB
            metrics['memory'] = get_memory_used() / MB
            self.insert_metrics((self.curr_iter, 'test', i), metrics)

        idx = (self.curr_iter, 'test') # mean across batches
        metrics = self.metrics.loc[idx].mean()
        self.print_metrics(idx, metrics)
        self.save_metrics()
        return metrics

    @save_on_exception
    def step(self, update=True, sync=False):
        torch.cuda.reset_max_memory_allocated()
        t0 = time.time()

        idx = (self.curr_iter, 'train', 0)
        loss, metrics = self.forward(self.train_data)
        if sync:
            torch.cuda.synchronize()

        m1 = torch.cuda.max_memory_allocated()
        torch.cuda.reset_max_memory_allocated()
        t1 = time.time()

        if update:
            self.optimizer.zero_grad()
            loss.backward()
            t2 = time.time()

            self.clip_gradient()
            grad_norm = models.compute_grad_norm(self.model)
            t3 = time.time()

            self.optimizer.step()
            if sync:
                torch.cuda.synchronize()

            self.curr_iter += 1

        m2 = torch.cuda.max_memory_allocated()
        t4 = time.time()

        metrics['forward_time'] = t1 - t0
        metrics['forward_gpu'] = m1 / MB
        metrics['memory'] = get_memory_used() / MB
        if update:
            if isinstance(self, DiscriminativeSolver):
                metrics['disc_grad_norm'] = grad_norm
            elif isinstance(self, GenerativeSolver):
                metrics['gen_grad_norm'] = grad_norm
            metrics['backward_time'] = t4 - t1
            metrics['backward_grad_time'] = t2 - t1
            metrics['backward_norm_time'] = t3 - t2
            metrics['backward_update_time'] = t4 - t3
            metrics['backward_gpu'] = m2 / MB

        self.insert_metrics(idx, metrics)
        metrics = self.metrics.loc[idx]
        self.print_metrics(idx[:-1], metrics)
        assert not loss.isnan(), 'loss is nan'
        if update:
            assert not np.isnan(grad_norm), 'gradient is nan'
            assert not np.isclose(0, grad_norm), 'gradient is zero'
        return metrics

    def train(
        self,
        max_iter,
        test_interval,
        n_test_batches,
        fit_interval,
        save_interval,
    ):
        while self.curr_iter <= max_iter:

            if self.curr_iter % save_interval == 0:
                self.save_state()

            if self.curr_iter % test_interval == 0:
                fit_atoms = (
                    fit_interval > 0 and self.curr_iter % fit_interval == 0
                )
                self.test(n_test_batches, fit_atoms)

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

    def clip_gradient(self):
        if self.disc_clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                self.disc_model.parameters(), self.disc_clip_grad
            )

    def forward(self, data):
        complex_grids, _, labels = data.forward()
        predictions = self.disc_model(complex_grids)
        loss, metrics = self.compute_loss(predictions, labels)
        metrics.update(compute_scalar_metrics('pred', predictions))
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

    def clip_gradient(self):
        if self.gen_clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                self.gen_model.parameters(), self.gen_clip_grad
            )


class AESolver(GenerativeSolver):
    gen_model_type = models.AE

    @property
    def n_channels_in(self):
        return self.train_data.n_lig_channels

    def initialize_loss(self, loss_types):
        self.recon_loss_fn = get_recon_loss_fn(loss_types['recon_loss'])

    def compute_loss(self, real_lig_grids, gen_lig_grids):
        recon_loss = self.recon_loss_fn(gen_lig_grids, real_lig_grids)
        loss = self.loss_weights['recon_loss'] * recon_loss
        return loss, OrderedDict([
            ('loss', loss.item()),
            ('recon_loss', recon_loss.item())
        ])

    def forward(self, data, fit_atoms=False):
        t0 = time.time()

        lig_grids, lig_structs, _ = data.forward(ligand_only=True)
        t1 = time.time()

        lig_gen_grids, latent_vecs = self.gen_model(lig_grids)
        loss, metrics = self.compute_loss(lig_grids, lig_gen_grids)
        t2 = time.time()
        
        if fit_atoms:
            lig_gen_fit_structs, _ = self.atom_fitter.fit_batch(
                lig_gen_grids,
                data.lig_channels,
                torch.zeros(3),
                data.resolution
            )
            self.save_structs(lig_gen_fit_structs)
        t3 = time.time()

        metrics.update(compute_paired_grid_metrics(
            'lig_gen', lig_gen_grids, 'lig', lig_grids
        ))
        if fit_atoms:
            metrics.update(compute_paired_struct_metrics(
                'lig_gen_fit', lig_gen_fit_structs, 'lig', lig_structs
            ))
        t4 = time.time()

        metrics['forward_data_time'] = t1 - t0
        metrics['forward_gen_time'] = t2 - t1
        metrics['forward_fit_time'] = t3 - t2
        metrics['forward_metrics_time'] = t4 - t3
        return loss, metrics


class VAESolver(AESolver):
    gen_model_type = models.VAE

    def initialize_loss(self, loss_types):
        self.kldiv_loss_fn = kl_divergence
        self.recon_loss_fn = get_recon_loss_fn(loss_types['recon_loss'])

    def compute_loss(self, real_lig_grids, gen_lig_grids, means, log_stds):
        recon_loss = self.recon_loss_fn(gen_lig_grids, real_lig_grids)
        kldiv_loss = self.kldiv_loss_fn(means, log_stds)
        loss = (
            self.loss_weights['recon_loss'] * recon_loss
        ) + (
            self.loss_weights['kldiv_loss'] * kldiv_loss
        )
        return loss, OrderedDict([
            ('loss', loss.item()),
            ('recon_loss', recon_loss.item()),
            ('kldiv_loss', kldiv_loss.item())
        ])

    def forward(self, data, fit_atoms=False):
        t0 = time.time()

        lig_grids, lig_structs, _ = data.forward(ligand_only=True)
        t1 = time.time()

        lig_gen_grids, latent_vecs, latent_means, latent_log_stds = \
            self.gen_model(lig_grids, data.batch_size)
        loss, metrics = self.compute_loss(
            lig_grids, lig_gen_grids, latent_means, latent_log_stds
        )
        t2 = time.time()
        
        if fit_atoms:
            lig_gen_fit_structs, _ = self.atom_fitter.fit_batch(
                lig_gen_grids,
                data.lig_channels,
                torch.zeros(3),
                data.resolution
            )
            self.save_structs(lig_gen_fit_structs)
        t3 = time.time()
        
        metrics.update(compute_paired_grid_metrics(
            'lig_gen', lig_gen_grids, 'lig', lig_grids
        ))
        if fit_atoms:
            metrics.update(compute_paired_struct_metrics(
                'lig_gen_fit', lig_gen_fit_structs, 'lig', lig_structs
            ))
        t4 = time.time()
        
        metrics['forward_data_time'] = t1 - t0
        metrics['forward_gen_time'] = t2 - t1
        metrics['forward_fit_time'] = t3 - t2
        metrics['forward_metrics_time'] = t4 - t3
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
        t0 = time.time()

        (rec_grids, lig_grids), lig_structs, _ = \
            data.forward(split_rec_lig=True)
        t1 = time.time()

        lig_gen_grids, latent_vecs = self.gen_model(rec_grids)
        loss, metrics = self.compute_loss(lig_grids, lig_gen_grids)
        t2 = time.time()

        if fit_atoms:
            lig_gen_fit_structs, _ = self.atom_fitter.fit_batch(
                lig_gen_grids,
                data.lig_channels,
                torch.zeros(3),
                data.resolution
            )
            self.save_structs(lig_gen_fit_structs)
        t3 = time.time()

        metrics.update(compute_paired_grid_metrics(
            'lig_gen', lig_gen_grids, 'lig', lig_grids
        ))
        if fit_atoms:
            metrics.update(compute_paired_struct_metrics(
                'lig_gen_fit', lig_gen_fit_structs, 'lig', lig_structs
            ))
        t4 = time.time()
        
        metrics['forward_data_time'] = t1 - t0
        metrics['forward_gen_time'] = t2 - t1
        metrics['forward_fit_time'] = t3 - t2
        metrics['forward_metrics_time'] = t4 - t3
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
        t0 = time.time()

        (rec_grids, lig_grids), lig_structs, _ = \
            data.forward(split_rec_lig=True)
        rec_lig_grids = data.grids
        t1 = time.time()

        lig_gen_grids, latent_vecs, latent_means, latent_log_stds = \
            self.gen_model(rec_lig_grids, rec_grids, data.batch_size)
        
        loss, metrics = self.compute_loss(
            lig_grids, lig_gen_grids, latent_means, latent_log_stds
        )
        t2 = time.time()

        if fit_atoms:
            lig_gen_fit_structs, _ = self.atom_fitter.fit_batch(
                lig_gen_grids,
                data.lig_channels,
                torch.zeros(3),
                data.resolution
            )
            self.save_structs(lig_gen_fit_structs)
        t3 = time.time()

        metrics.update(compute_paired_grid_metrics(
            'lig_gen', lig_gen_grids, 'lig', lig_grids
        ))
        if fit_atoms:
            metrics.update(compute_paired_struct_metrics(
                'lig_gen_fit', lig_gen_fit_structs, 'lig', lig_structs
            ))
        t4 = time.time()

        metrics['forward_data_time'] = t1 - t0
        metrics['forward_gen_time'] = t2 - t1
        metrics['forward_fit_time'] = t3 - t2
        metrics['forward_metrics_time'] = t4 - t3
        return loss, metrics


class GANSolver(GenerativeSolver):
    gen_model_type = models.GAN
    index_cols = ['iteration', 'disc_iter', 'phase', 'model', 'batch', 'real']

    @property
    def n_channels_disc(self):
        return self.train_data.n_lig_channels

    def initialize_loss(self, loss_types):
        self.gan_loss_fn = get_gan_loss_fn(loss_types['gan_loss'])

    def clip_gradient(self, gen=False, disc=False):

        if gen and self.gen_clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                self.gen_model.parameters(), self.gen_clip_grad
            )
        if disc and self.disc_clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                self.disc_model.parameters(), self.disc_clip_grad
            )

    def compute_loss(self, labels, predictions):
        gan_loss = self.gan_loss_fn(predictions, labels)
        loss = self.loss_weights['gan_loss'] * gan_loss
        return loss, OrderedDict([
            ('loss', loss.item()),
            ('gan_loss', gan_loss.item())
        ])

    def disc_forward(self, data, real):
        '''
        Compute predictions and loss for the discriminator's
        ability to correctly classify real or generated data.
        '''
        t0 = time.time()
        with torch.no_grad(): # do not backprop to generator or data

            if real: # get real examples
                lig_grids, lig_structs, _ = data.forward(ligand_only=True)
                labels = torch.ones(data.batch_size, 1, device=self.device)

            else: # get generated examples
                lig_grids, _ = self.gen_model(data.batch_size)
                labels = torch.zeros(data.batch_size, 1, device=self.device)
            
        t1 = time.time()

        predictions, _ = self.disc_model(lig_grids)
        loss, metrics = self.compute_loss(labels, predictions)
        t2 = time.time()

        metrics.update(compute_grid_metrics(
            'lig' if real else 'lig_gen', lig_grids
        ))
        metrics.update(compute_scalar_metrics('pred', predictions))
        t3 = time.time()

        if real:
            metrics['forward_data_time'] = t1 - t0
        else:
            metrics['forward_gen_time'] = t1 - t0
        metrics['forward_disc_time'] = t2 - t1
        metrics['forward_metrics_time'] = t3 - t2
        return loss, metrics

    def gen_forward(self, data, fit_atoms=False):
        '''
        Compute predictions and loss for the generator's ability
        to produce data that is misclassified by the discriminator.
        '''
        t0 = time.time()

        # get generated examples
        lig_gen_grids, latent_vecs = self.gen_model(data.batch_size)
        labels = torch.ones(data.batch_size, 1, device=self.device)
        t1 = time.time()

        predictions, _ = self.disc_model(lig_gen_grids)
        loss, metrics = self.compute_loss(labels, predictions)
        t2 = time.time()

        if fit_atoms:
            lig_gen_fit_structs, _ = self.atom_fitter.fit_batch(
                lig_gen_grids,
                data.lig_channels,
                torch.zeros(3),
                data.resolution
            )
            self.save_structs(lig_gen_fit_structs)
        t3 = time.time()

        metrics.update(compute_grid_metrics('lig_gen', lig_gen_grids))
        metrics.update(compute_scalar_metrics('pred', predictions))
        if fit_atoms:
            metrics.update(compute_struct_metrics(
                'lig_gen_fit', lig_gen_fit_structs
            ))
        t4 = time.time()

        metrics['forward_gen_time'] = t1 - t0
        metrics['forward_disc_time'] = t2 - t1
        metrics['forward_fit_time'] = t3 - t2
        metrics['forward_metrics_time'] = t4 - t3
        return loss, metrics

    @save_on_exception
    def test(self, n_batches, fit_atoms=False):

        # test discriminator on alternating real/generated batches
        for i in range(n_batches):
            torch.cuda.reset_max_memory_allocated()
            t0 = time.time()
            real = (i%2 == 0)
            loss, metrics = self.disc_forward(self.test_data, real)
            metrics['forward_time'] = time.time() - t0
            metrics['forward_gpu'] = torch.cuda.max_memory_allocated() / MB
            metrics['memory'] = get_memory_used() / MB
            idx = (self.gen_iter, self.disc_iter, 'test', 'disc', i, real)
            self.insert_metrics(idx, metrics)

        idx = (self.gen_iter, self.disc_iter, 'test', 'disc')
        metrics = self.metrics.loc[idx].mean()
        self.print_metrics(idx, metrics)

        # test generator on same number of batches
        for i in range(n_batches):
            torch.cuda.reset_max_memory_allocated()
            t0 = time.time()
            loss, metrics = self.gen_forward(self.test_data, fit_atoms)
            metrics['forward_time'] = time.time() - t0
            metrics['forward_gpu'] = torch.cuda.max_memory_allocated() / MB
            metrics['memory'] = get_memory_used() / MB
            idx = (self.gen_iter, self.disc_iter, 'test', 'gen', i, False)
            self.insert_metrics(idx, metrics)

        idx = (self.gen_iter, self.disc_iter, 'test', 'gen')
        metrics = self.metrics.loc[idx].mean()
        self.print_metrics(idx, metrics)

        self.save_metrics()
        return self.metrics.loc[(self.gen_iter, self.disc_iter, 'test')]

    @save_on_exception
    def disc_step(self, real, update=True, batch_idx=0, sync=False):
        torch.cuda.reset_max_memory_allocated()
        t0 = time.time()

        idx = (
            self.gen_iter, self.disc_iter, 'train', 'disc', batch_idx, real
        )
        loss, metrics = self.disc_forward(self.train_data, real)
        if sync:
            torch.cuda.synchronize()

        m1 = torch.cuda.max_memory_allocated()
        torch.cuda.reset_max_memory_allocated()
        t1 = time.time()
        
        if update:
            self.disc_optimizer.zero_grad()
            loss.backward()
            t2 = time.time()

            self.clip_gradient(disc=True)
            grad_norm = models.compute_grad_norm(self.disc_model)
            t3 = time.time()

            self.disc_optimizer.step()
            if sync:
                torch.cuda.synchronize()

            self.disc_iter += 1

        m2 = torch.cuda.max_memory_allocated()
        t4 = time.time()

        metrics['forward_time'] = t1 - t0
        metrics['forward_gpu'] = m1 / MB
        metrics['memory'] = get_memory_used() / MB
        if update:
            metrics['disc_grad_norm'] = grad_norm
            metrics['backward_time'] = t4 - t1
            metrics['backward_grad_time'] = t2 - t1
            metrics['backward_norm_time'] = t3 - t2
            metrics['backward_update_time'] = t4 - t3
            metrics['backward_gpu'] = m2 / MB

        self.insert_metrics(idx, metrics)
        metrics = self.metrics.loc[idx]
        self.print_metrics(idx[:-1], metrics)
        assert not loss.isnan(), 'discriminator loss is nan'
        if update:
            assert not np.isnan(grad_norm), 'discriminator gradient is nan'
            #assert not np.isclose(0, grad_norm), 'discriminator gradient is zero'
        return metrics

    @save_on_exception
    def gen_step(self, update=True, batch_idx=0, sync=False):
        torch.cuda.reset_max_memory_allocated()
        t0 = time.time()

        idx = (
            self.gen_iter, self.disc_iter, 'train', 'gen', batch_idx, False
        )
        loss, metrics = self.gen_forward(self.train_data)
        if sync:
            torch.cuda.synchronize()

        m1 = torch.cuda.max_memory_allocated()
        torch.cuda.reset_max_memory_allocated()
        t1 = time.time()

        if update:
            self.gen_optimizer.zero_grad()
            loss.backward()
            t2 = time.time()

            self.clip_gradient(gen=True)
            grad_norm = models.compute_grad_norm(self.gen_model)
            t3 = time.time()

            self.gen_optimizer.step()
            if sync:
                torch.cuda.synchronize()

            self.gen_iter += 1

        m2 = torch.cuda.max_memory_allocated()
        t4 = time.time()

        metrics['forward_time'] = t1 - t0
        metrics['forward_gpu'] = m1 / MB
        metrics['memory'] = get_memory_used() / MB
        if update:
            metrics['gen_grad_norm'] = grad_norm
            metrics['backward_time'] = t4 - t1
            metrics['backward_grad_time'] = t2 - t1
            metrics['backward_norm_time'] = t3 - t2
            metrics['backward_update_time'] = t4 - t3
            metrics['backward_gpu'] = m2 / MB

        self.insert_metrics(idx, metrics)
        metrics = self.metrics.loc[idx]
        self.print_metrics(idx[:-1], metrics)
        assert not loss.isnan(), 'generator loss is nan'
        if update:
            assert not np.isnan(grad_norm), 'generator gradient is nan'
            assert not np.isclose(0, grad_norm), 'generator gradient is zero'
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
        test_interval,
        n_test_batches,
        fit_interval,
        save_interval,
    ):
        while self.curr_iter <= max_iter:

            if self.curr_iter % save_interval == 0:
                self.save_state()

            if self.curr_iter % test_interval == 0:
                fit_atoms = (
                    fit_interval > 0 and self.curr_iter % fit_interval == 0
                )
                self.test(n_test_batches, fit_atoms)

            if self.curr_iter < max_iter:
                self.train_disc(self.n_disc_train_iters, update=True)
                self.train_gen(self.n_gen_train_iters, update=True)
            else:
                self.train_disc(self.n_disc_train_iters, update=False)
                self.train_gen(self.n_gen_train_iters, update=False)
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

        t0 = time.time()
        with torch.no_grad(): # do not backprop to generator or data

            # get real examples
            (rec_grids, lig_grids), lig_structs, _ = \
                data.forward(split_rec_lig=True)
            t1 = time.time()
            if real:
                labels = torch.ones(data.batch_size, 1, device=self.device)

            else: # get generated examples
                lig_gen_grids, _ = self.gen_model(rec_grids, data.batch_size)
                labels = torch.zeros(data.batch_size, 1, device=self.device)
                lig_grids = lig_gen_grids

        t2 = time.time()

        rec_lig_grids = torch.cat([rec_grids, lig_grids], dim=1)
        predictions, _ = self.disc_model(rec_lig_grids)
        loss, metrics = self.compute_loss(labels, predictions)
        t3 = time.time()

        metrics.update(compute_grid_metrics(
            'lig' if real else 'lig_gen', lig_grids
        ))
        metrics.update(compute_scalar_metrics('pred', predictions))
        t4 = time.time()

        metrics['forward_data_time'] = t1 - t0
        if not real:
            metrics['forward_gen_time'] = t2 - t1
        metrics['forward_disc_time'] = t3 - t2
        metrics['forward_metrics_time'] = t4 - t3
        return loss, metrics

    def gen_forward(self, data, fit_atoms=False):
        t0 = time.time()

        # get generated examples
        (rec_grids, lig_grids), lig_structs, _ = \
            data.forward(split_rec_lig=True)
        t1 = time.time()

        lig_gen_grids, latent_vecs = self.gen_model(
            rec_grids, data.batch_size
        )
        labels = torch.ones(data.batch_size, 1, device=self.device)
        t2 = time.time()

        rec_lig_grids = torch.cat([rec_grids, lig_gen_grids], dim=1)
        predictions, _ = self.disc_model(rec_lig_grids)
        loss, metrics = self.compute_loss(labels, predictions)
        t3 = time.time()

        if fit_atoms:
            lig_gen_fit_structs, _ = self.atom_fitter.fit_batch(
                lig_gen_grids,
                data.lig_channels,
                torch.zeros(3),
                data.resolution
            )
            self.save_structs(lig_gen_fit_structs)
        t4 = time.time()

        metrics.update(compute_grid_metrics('lig_gen', lig_gen_grids))
        metrics.update(compute_scalar_metrics('pred', predictions))
        if fit_atoms:
            metrics.update(compute_struct_metrics(
                'lig_gen_fit', lig_gen_fit_structs
            ))
        t5 = time.time()

        metrics['forward_data_time'] = t1 - t0
        metrics['forward_gen_time'] = t2 - t1
        metrics['forward_disc_time'] = t3 - t2
        metrics['forward_fit_time'] = t4 - t3
        metrics['forward_metrics_time'] = t5 - t4
        return loss, metrics


class VAEGANSolver(GANSolver):
    gen_model_type = models.VAE

    @property
    def n_channels_in(self):
        return self.train_data.n_lig_channels

    def initialize_loss(self, loss_types):
        self.kldiv_loss_fn = kl_divergence
        self.recon_loss_fn = get_recon_loss_fn(loss_types['recon_loss'])
        self.gan_loss_fn = get_gan_loss_fn(loss_types['gan_loss'])

    def compute_loss(
        self, labels, predictions, real_lig_grids, gen_lig_grids, means, log_stds
    ):
        gan_loss = self.gan_loss_fn(predictions, labels)
        recon_loss = self.recon_loss_fn(gen_lig_grids, real_lig_grids)
        kldiv_loss = self.kldiv_loss_fn(means, log_stds)
        loss = (
            self.loss_weights['gan_loss'] * gan_loss
        ) + (
            self.loss_weights['recon_loss'] * recon_loss
        ) + (
            self.loss_weights['kldiv_loss'] * kldiv_loss
        )
        return loss, OrderedDict([
            ('loss', loss.item()),
            ('gan_loss', gan_loss.item()),
            ('recon_loss', recon_loss.item()),
            ('kldiv_loss', kldiv_loss.item()),
        ])

    def disc_forward(self, data, real):
        t0 = time.time()

        with torch.no_grad(): # do not backprop to generator or data

            # get real examples
            lig_grids, lig_structs, _ = data.forward(ligand_only=True)
            t1 = time.time()

            if real:
                labels = torch.ones(data.batch_size, 1, device=self.device)

            else: # get generated examples
                lig_gen_grids, latent_vecs, latent_means, latent_log_stds = \
                    self.gen_model(lig_grids, data.batch_size)
                labels = torch.zeros(data.batch_size, 1, device=self.device)

        t2 = time.time()
        
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
        t3 = time.time()

        metrics.update(compute_grid_metrics(
            'lig' if real else 'lig_gen',
            lig_grids if real else lig_gen_grids
        ))
        metrics.update(compute_scalar_metrics('pred', predictions))
        t4 = time.time()

        metrics['forward_data_time'] = t1 - t0
        metrics['forward_gen_time'] = t2 - t1
        metrics['forward_disc_time'] = t3 - t2
        metrics['forward_metrics_time'] = t4 - t3
        return loss, metrics

    def gen_forward(self, data, fit_atoms=False):
        t0 = time.time()

        # get generated examples
        lig_grids, lig_structs, _ = data.forward(ligand_only=True)
        t1 = time.time()

        lig_gen_grids, latent_vecs, latent_means, latent_log_stds = \
            self.gen_model(lig_grids, data.batch_size)
        labels = torch.ones(data.batch_size, 1, device=self.device)
        t2 = time.time()

        predictions, _ = self.disc_model(lig_gen_grids)
        loss, metrics = self.compute_loss(
            labels,
            predictions,
            lig_grids,
            lig_gen_grids,
            latent_means,
            latent_log_stds
        )
        t3 = time.time()

        if fit_atoms:
            lig_gen_fit_structs, _ = self.atom_fitter.fit_batch(
                lig_gen_grids,
                data.lig_channels,
                torch.zeros(3),
                data.resolution
            )
            self.save_structs(lig_gen_fit_structs)
        t4 = time.time()

        metrics.update(compute_paired_grid_metrics(
            'lig_gen', lig_gen_grids, 'lig', lig_grids
        ))
        metrics.update(compute_scalar_metrics('pred', predictions))
        if fit_atoms:
            metrics.update(compute_paired_struct_metrics(
                'lig_gen_fit', lig_gen_fit_structs, 'lig', lig_structs
            ))
        t5 = time.time()

        metrics['forward_data_time'] = t1 - t0
        metrics['forward_gen_time'] = t2 - t1
        metrics['forward_disc_time'] = t3 - t2
        metrics['forward_fit_time'] = t4 - t3
        metrics['forward_metrics_time'] = t5 - t4
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

        t0 = time.time()
        with torch.no_grad(): # do not backprop to generator or data

            # get real examples
            (rec_grids, lig_grids), lig_structs, _ = \
                data.forward(split_rec_lig=True)
            rec_lig_grids = data.grids
            t1 = time.time()

            if real:
                labels = torch.ones(data.batch_size, 1, device=self.device)

            else: # get generated examples
                lig_gen_grids, latent_vecs, latent_means, latent_log_stds = \
                    self.gen_model(rec_lig_grids, rec_grids, data.batch_size)
                labels = torch.zeros(data.batch_size, 1, device=self.device)
                rec_lig_grids = torch.cat([rec_grids, lig_gen_grids], dim=1)

        t2 = time.time()

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
        t3 = time.time()

        metrics.update(compute_grid_metrics(
            'lig' if real else 'lig_gen',
            lig_grids if real else lig_gen_grids
        ))
        metrics.update(compute_scalar_metrics('pred', predictions))
        t4 = time.time()

        metrics['forward_data_time'] = t1 - t0
        metrics['forward_gen_time'] = t2 - t1
        metrics['forward_disc_time'] = t3 - t2
        metrics['forward_metrics_time'] = t4 - t3
        return loss, metrics

    def gen_forward(self, data, fit_atoms=False):
        t0 = time.time()

        # get generated examples
        (rec_grids, lig_grids), lig_structs, _ = \
            data.forward(split_rec_lig=True)
        rec_lig_grids = data.grids
        t1 = time.time()

        lig_gen_grids, latent_vecs, latent_means, latent_log_stds = \
            self.gen_model(rec_lig_grids, rec_grids, data.batch_size)
        labels = torch.ones(data.batch_size, 1, device=self.device)
        t2 = time.time()

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
        t3 = time.time()

        if fit_atoms:
            lig_gen_fit_structs, _ = self.atom_fitter.fit_batch(
                lig_gen_grids,
                data.lig_channels,
                torch.zeros(3),
                data.resolution
            )
            self.save_structs(lig_gen_fit_structs)
        t4 = time.time()

        metrics.update(compute_paired_grid_metrics(
            'lig_gen', lig_gen_grids, 'lig', lig_grids
        ))
        metrics.update(compute_scalar_metrics('pred', predictions))
        if fit_atoms:
            metrics.update(compute_struct_metrics(
                'lig_gen_fit', lig_gen_fit_structs
            ))
        t5 = time.time()

        metrics['forward_data_time'] = t1 - t0
        metrics['forward_gen_time'] = t2 - t1
        metrics['forward_disc_time'] = t3 - t2
        metrics['forward_fit_time'] = t4 - t3
        metrics['forward_metrics_time'] = t5 - t4
        return loss, metrics
