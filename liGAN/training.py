import sys, os, time, psutil, pynvml, re, glob
from collections import OrderedDict
from functools import partial
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
torch.backends.cudnn.benchmark = True

import molgrid
from . import data, models, atom_types, atom_fitting, bond_adding, molecules
from . import loss_fns
from .common import set_random_seed
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


def save_on_exception(method):
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except:
            self.save_state_and_metrics()
            raise
    return wrapper


def get_state_prefix(out_prefix, iter_):
    return '{}_iter_{}'.format(out_prefix, iter_)


def find_last_iter(out_prefix, min_iter=-1):
    last_iter = min_iter
    state_file_re = re.compile(re.escape(out_prefix) + r'_iter_(\d+).*state')
    for state_file in glob.glob(out_prefix + '_iter_*state'):
        m = state_file_re.match(state_file)
        try:
            last_iter = max(last_iter, int(m.group(1)))
        except AttributeError:
            print(state_file_re.pattern)
            print(state_file)
            raise
    if last_iter > min_iter:
        return last_iter
    else:
        raise FileNotFoundError('could not find state files ({})'.format(last_iter))


class GenerativeSolver(nn.Module):
    '''
    Base class for training models that
    generate ligand atomic density grids.
    '''
    gen_model_type = None
    has_disc_model = False
    has_complex_input = False

    def __init__(
        self,
        out_prefix,
        data_kws={},
        gen_model_kws={},
        disc_model_kws={},
        loss_fn_kws={},
        gen_optim_kws={},
        disc_optim_kws={},
        atom_fitting_kws={},
        bond_adding_kws={},
        device='cuda',
        debug=False,
        sync_cuda=False,
    ):
        super().__init__()
        self.device = device

        print('Loading data')
        self.init_data(device=device, **data_kws)

        print('Initializing generative model and optimizer')
        self.init_gen_model(device=device, **gen_model_kws)
        self.init_gen_optimizer(**gen_optim_kws)

        if self.has_disc_model:

            print('Initializing discriminative model and optimizer')
            self.init_disc_model(device=device, **disc_model_kws)
            self.init_disc_optimizer(**disc_optim_kws)

        else: # needed for df index
            self.disc_iter = 0

        self.init_loss_fn(device=device, **loss_fn_kws)

        print('Initializing atom fitter and bond adder')
        self.atom_fitter = atom_fitting.AtomFitter(
            device=device, **atom_fitting_kws
        )
        self.bond_adder = bond_adding.BondAdder(
            debug=debug, **bond_adding_kws
        )

        # set up a data frame of training metrics
        self.index_cols = [
            'iteration',   # gen model iteration
            'disc_iter',   # disc model iteration
            'data_phase',  # train/test data
            'model_phase', # gen/disc model
            'grid_phase',  # real, prior, poster
            'batch'        # batch in current iteration
        ]
        self.metrics = pd.DataFrame(columns=self.index_cols)
        self.metrics.set_index(self.index_cols, inplace=True)

        self.out_prefix = out_prefix
        self.debug = debug
        self.sync_cuda = sync_cuda

    def init_gen_model(
        self,
        device,
        caffe_init=False,
        gen_model_state=None,
        **gen_model_kws
    ):
        self.gen_model = self.gen_model_type(
            n_channels_in=self.n_channels_in,
            n_channels_cond=self.n_channels_cond,
            n_channels_out=self.n_channels_out,
            grid_size=self.train_data.grid_size,
            device=device,
            **gen_model_kws
        )
        if caffe_init:
            self.gen_model.apply(models.caffe_init_weights)

        if gen_model_state:
            self.gen_model.load_state_dict(torch.load(gen_model_state))

    def init_data(self, device, train_file, test_file, **data_kws):
        self.train_data, self.test_data = (
            data.AtomGridData(device=device, **data_kws) for i in range(2)
        )
        self.train_data.populate(train_file)
        self.test_data.populate(test_file)

    def init_disc_model(
        self,
        device,
        caffe_init=False,
        disc_model_state=None,
        **disc_model_kws
    ):
        self.disc_model = models.Discriminator(
            n_channels=self.n_channels_disc,
            grid_size=self.train_data.grid_size,
            **disc_model_kws
        ).to(device)

        if caffe_init:
            self.disc_model.apply(models.caffe_init_weights)

        if disc_model_state:
            self.disc_model.load_state_dict(torch.load(disc_model_state))

    def init_gen_optimizer(
        self, type, n_train_iters=1, clip_gradient=0, **gen_optim_kws
    ):
        self.n_gen_train_iters = n_train_iters
        self.gen_clip_grad = clip_gradient
        self.gen_optimizer = getattr(optim, type)(
            self.gen_model.parameters(), **gen_optim_kws
        )
        self.gen_iter = 0

    def init_disc_optimizer(
        self, type, n_train_iters=2, clip_gradient=0, **disc_optim_kws
    ):
        self.n_disc_train_iters = n_train_iters
        self.disc_clip_grad = clip_gradient
        self.disc_optimizer = getattr(optim, type)(
            self.disc_model.parameters(), **disc_optim_kws
        )
        self.disc_iter = 0

    def init_loss_fn(
        self, device, balance=False, **loss_fn_kws,
    ):
        self.loss_fn = loss_fns.LossFunction(device=device, **loss_fn_kws)

        if self.has_disc_model:
            assert self.loss_fn.gan_loss_wt != 0, 'GAN loss weight is zero'

            if balance:
                self.disc_gan_loss = -1
                self.gen_gan_loss = 0
        else:
            assert self.loss_fn.gan_loss_wt == 0, \
                'non-zero GAN loss but no disc'
            assert balance == False, 'can only balance GAN loss'

        if not self.gen_model_type.has_conditional_encoder:
            assert self.loss_fn.steric_loss_wt == 0, \
                'non-zero steric loss but no rec'

        self.balance = balance

    @property
    def n_channels_in(self):
        if self.gen_model_type.has_input_encoder:
            data = self.train_data
            if self.has_complex_input:
                return data.n_rec_channels + data.n_lig_channels
            else:
                return data.n_lig_channels

    @property
    def n_channels_cond(self):
        if self.gen_model_type.has_conditional_encoder:
            return self.train_data.n_rec_channels

    @property
    def n_channels_out(self):
        return self.train_data.n_lig_channels

    @property
    def n_channels_disc(self):
        if self.has_disc_model:
            data = self.train_data
            if self.gen_model_type.has_conditional_encoder:
                return data.n_rec_channels + data.n_lig_channels
            else:
                return data.n_lig_channels

    @property
    def state_prefix(self):
        return get_state_prefix(self.out_prefix, self.gen_iter)

    @property
    def gen_model_state_file(self):
        return self.state_prefix + '.gen_model_state'

    @property
    def gen_solver_state_file(self):
        return self.state_prefix + '.gen_solver_state'

    @property
    def disc_model_state_file(self):
        return self.state_prefix + '.disc_model_state'

    @property
    def disc_solver_state_file(self):
        return self.state_prefix + '.disc_solver_state'

    @property
    def metrics_file(self):
        return self.out_prefix + '.train_metrics'

    def save_state(self):

        self.gen_model.cpu()

        state_file = self.gen_model_state_file
        print('Saving generative model state to ' + state_file)
        torch.save(self.gen_model.state_dict(), state_file)

        state_file = self.gen_solver_state_file
        print('Saving generative solver state to ' + state_file)
        state_dict = OrderedDict()
        state_dict['optim_state'] = self.gen_optimizer.state_dict() 
        state_dict['iter'] = self.gen_iter
        torch.save(state_dict, state_file)

        self.gen_model.to(self.device)

        if self.has_disc_model:
            self.disc_model.cpu()

            state_file = self.disc_model_state_file
            print('Saving discriminative model state to ' + state_file)
            torch.save(self.disc_model.state_dict(), state_file)

            state_file = self.disc_solver_state_file
            print('Saving discriminative solver state to ' + state_file)
            state_dict = OrderedDict()
            state_dict['optim_state'] = self.disc_optimizer.state_dict() 
            state_dict['iter'] = self.disc_iter
            torch.save(state_dict, state_file)

            self.disc_model.to(self.device)

    def load_state(self, cont_iter=None):

        if cont_iter is None:
            self.gen_iter = self.find_last_iter()
        else:
            self.gen_iter = cont_iter

        state_file = self.state_prefix + '.gen_model_state'
        print('Loading generative model state from ' + state_file)
        self.gen_model.load_state_dict(torch.load(state_file))

        state_file = self.state_prefix + '.gen_solver_state'
        print('Loading generative solver state from ' + state_file)
        state_dict = torch.load(state_file)
        self.gen_optimizer.load_state_dict(state_dict['optim_state'])
        self.gen_iter = state_dict['iter']

        if self.has_disc_model:

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
        csv_file = self.metrics_file
        print('Writing training metrics to ' + csv_file)
        self.metrics.to_csv(csv_file, sep=' ')

    def load_metrics(self):
        csv_file = self.metrics_file
        print('Reading training metrics from ' + csv_file)
        self.metrics = pd.read_csv(
            csv_file, sep=' '
        ).set_index(self.index_cols)

    def load_state_and_metrics(self):
        self.load_state()
        try:
            self.load_metrics()
        except FileNotFoundError:
            if self.gen_iter > 0:
                raise

    def save_state_and_metrics(self):
        self.save_metrics()
        self.save_state()

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

    def save_mols(self, mols, grid_type):
        sdf_file = '{}_iter_{}_{}.sdf'.format(
            self.out_prefix, self.gen_iter, grid_type
        )
        print('Writing generated molecules to ' + sdf_file)
        molecules.write_rd_mols_to_sdf_file(sdf_file, mols, kekulize=False)

    @property
    def has_prior_phase(self):
        return (
            self.gen_model_type.is_variational and
            self.loss_fn.has_prior_loss
        )

    @property
    def has_posterior_phase(self):
        return (
            self.gen_model_type.has_input_encoder or
            not self.has_prior_phase
        )

    def get_gen_grid_phase(self, batch_idx):
        '''
        Determine whether to sample prior or
        posterior grids in the next gen batch.
        '''
        has_prior_phase = self.has_prior_phase
        has_posterior_phase = self.has_posterior_phase
        assert has_prior_phase or has_posterior_phase, 'no gen grid phases'

        grid_phases = []
        if has_posterior_phase:
            grid_phases.append('poster')

        if has_prior_phase:
            grid_phases.append('prior')

        phase_idx = (self.gen_iter + batch_idx)
        return grid_phases[phase_idx % len(grid_phases)]

    def get_disc_grid_phase(self, batch_idx):
        '''
        Determine whether to sample real, prior,
        or posterior grids in the next disc batch.
        '''
        has_prior_phase = self.has_prior_phase
        has_posterior_phase = self.has_posterior_phase
        assert has_prior_phase or has_posterior_phase, 'no disc grid phases'

        grid_phases = []
        if has_posterior_phase:
            grid_phases += ['real', 'poster']

        if has_prior_phase:
            grid_phases += ['real', 'prior']

        phase_idx = (self.disc_iter + batch_idx)
        return grid_phases[phase_idx % len(grid_phases)]

    def gen_forward(self, data, grid_type, fit_atoms=False):
        '''
        Compute loss and other metrics for the
        generative model's ability to produce
        realistic atomic density grids.
        '''
        is_varial = self.gen_model.is_variational
        has_input = self.gen_model.has_input_encoder
        has_cond = self.gen_model_type.has_conditional_encoder
        has_disc = self.has_disc_model

        valid_grid_types = set()
        if self.has_prior_phase:
            valid_grid_types.add('prior')
        if self.has_posterior_phase:
            valid_grid_types.add('poster')

        assert grid_type in valid_grid_types, \
            'invalid grid type ' + repr(grid_type)
        prior = (grid_type == 'prior')
        posterior = (grid_type == 'poster')

        t0 = time.time()
        if posterior or has_cond: # get real examples
            grids, structs, _ = data.forward(split_rec_lig=True)
            rec_grids, lig_grids = grids
            rec_lig_grids = data.grids
            rec_structs, lig_structs = structs

        if self.sync_cuda:
            torch.cuda.synchronize()
        t1 = time.time()

        # get generated ligand grids
        if posterior:
            if self.has_complex_input:
                gen_input_grids = rec_lig_grids
            else:
                gen_input_grids = lig_grids

        lig_gen_grids, latent_vecs, latent_means, latent_log_stds = \
            self.gen_model(
                inputs=gen_input_grids if posterior else None,
                conditions=rec_grids if has_cond else None,
                batch_size=data.batch_size
            )

        if self.sync_cuda:
            torch.cuda.synchronize()
        t2 = time.time()

        if has_disc: # get discriminator predictions
            if has_cond:
                disc_input_grids = torch.cat([rec_grids,lig_gen_grids], dim=1)
            else:
                disc_input_grids = lig_gen_grids

            disc_labels = torch.ones(data.batch_size, 1, device=self.device)
            disc_preds, _ = self.disc_model(inputs=disc_input_grids)

        loss, metrics = self.loss_fn(
            lig_grids=lig_grids if posterior else None,
            lig_gen_grids=lig_gen_grids if posterior else None,
            disc_labels=disc_labels if has_disc else None,
            disc_preds=disc_preds if has_disc else None,
            latent_means=latent_means if posterior else None,
            latent_log_stds=latent_log_stds if posterior else None,
            rec_grids=rec_grids if has_cond else None,
            rec_lig_grids=lig_gen_grids if has_cond else None,
            iteration=self.gen_iter,
        )

        if self.sync_cuda:
            torch.cuda.synchronize()
        t3 = time.time()

        if fit_atoms:
            lig_gen_fit_structs, _ = self.atom_fitter.fit_batch(
                batch_values=lig_gen_grids,
                center=torch.zeros(3),
                resolution=data.resolution,
                typer=data.lig_typer,
            )
            lig_gen_fit_mols, _ = self.bond_adder.make_batch(
                structs=lig_gen_fit_structs
            )
            self.save_mols(lig_gen_fit_mols, grid_type)

        if self.sync_cuda:
            torch.cuda.synchronize()
        t4 = time.time()

        if posterior:
            metrics.update(compute_paired_grid_metrics(
                'lig_gen', lig_gen_grids, 'lig', lig_grids
            ))
        else:
            metrics.update(compute_grid_metrics('lig_gen', lig_gen_grids))

        if has_disc:
            metrics.update(compute_scalar_metrics('pred', disc_preds))

        if fit_atoms:
            if posterior:
                metrics.update(compute_paired_struct_metrics(
                    'lig_gen_fit', lig_gen_fit_structs, 'lig', lig_structs
                ))
            else:
                metrics.update(compute_struct_metrics(
                    'lig_gen_fit', lig_gen_fit_structs
                ))

        if self.sync_cuda:
            torch.cuda.synchronize()
        t5 = time.time()

        metrics['forward_data_time'] = t1 - t0
        metrics['forward_gen_time'] = t2 - t1
        metrics['forward_disc_time'] = t3 - t2
        metrics['forward_fit_time'] = t4 - t3
        metrics['forward_metrics_time'] = t5 - t4
        return loss, metrics

    def disc_forward(self, data, grid_type):
        '''
        Compute loss and other metrics for the
        discriminative model's ability to tell
        apart real and generated data.
        '''
        is_varial = self.gen_model.is_variational
        has_input = self.gen_model.has_input_encoder
        has_cond = self.gen_model_type.has_conditional_encoder

        valid_grid_types = {'real'}
        if is_varial:
            valid_grid_types.add('prior')
        if has_input:
            valid_grid_types.add('poster')

        assert grid_type in valid_grid_types, 'invalid grid type'
        real = (grid_type == 'real')
        prior = (grid_type == 'prior')
        posterior = (grid_type == 'poster')

        t0 = time.time()
        with torch.no_grad(): # do not backprop to generator or data

            if real or posterior or has_cond: # get real examples
                grids, structs, _ = data.forward(split_rec_lig=True)
                rec_grids, lig_grids = grids
                rec_lig_grids = data.grids
                rec_structs, lig_structs = structs

            t1 = time.time()

            if not real: # get generated ligand grids

                if posterior:
                    if self.has_complex_input:
                        gen_input_grids = rec_lig_grids
                    else:
                        gen_input_grids = lig_grids

                lig_gen_grids, latent_vecs, latent_means, latent_log_stds = \
                    self.gen_model(
                        inputs=gen_input_grids if posterior else None,
                        conditions=rec_grids if has_cond else None,
                        batch_size=data.batch_size
                    )
            t2 = time.time()

        # get discriminator predictions
        if real:
            disc_grids = rec_lig_grids if has_cond else lig_grids
        elif has_cond:
            disc_grids = torch.cat([rec_grids, lig_gen_grids], dim=1)
        else:
            disc_grids = lig_gen_grids

        disc_labels = torch.full(
            (data.batch_size, 1), real, device=self.device
        )
        disc_preds, _ = self.disc_model(inputs=disc_grids)
        loss, metrics = self.loss_fn(
            disc_labels=disc_labels, disc_preds=disc_preds, use_loss_wt=False
        )
        t3 = time.time()

        metrics.update(compute_grid_metrics(
            'lig' if real else 'lig_gen',
            lig_grids if real else lig_gen_grids
        ))
        metrics.update(compute_scalar_metrics('disc_pred', disc_preds))
        t4 = time.time()

        metrics['forward_data_time'] = t1 - t0
        metrics['forward_gen_time'] = t2 - t1
        metrics['forward_disc_time'] = t3 - t2
        metrics['forward_metrics_time'] = t4 - t3
        return loss, metrics

    def gen_backward(self, loss, update=False, compute_norm=False):
        '''
        Backpropagate loss gradient onto
        generative model parameters, op-
        tionally computing the gradient
        norm and/or updating parameters.
        '''
        metrics = OrderedDict()
        t0 = time.time()

        # compute gradient of loss wrt parameters
        self.gen_optimizer.zero_grad()
        loss.backward()

        t1 = time.time()

        if self.gen_clip_grad: # clip norm of parameter gradient
            models.clip_grad_norm(self.gen_model, self.gen_clip_grad)

        if compute_norm: # compute parameter gradient norm
            grad_norm = models.compute_grad_norm(self.gen_model)

        t2 = time.time()

        if update: # descend gradient on parameters
            self.gen_optimizer.step()
            self.gen_iter += 1
        
        t3 = time.time()
        if compute_norm:
            metrics['gen_grad_norm'] = grad_norm
        metrics['backward_grad_time'] = t1 - t0
        metrics['backward_norm_time'] = t2 - t1
        metrics['backward_update_time'] = t3 - t2
        return metrics

    def disc_backward(self, loss, update=False, compute_norm=False):
        '''
        Backpropagate loss gradient onto
        discriminative model parameters,
        optionally computing the gradient
        norm and/or updating parameters.
        '''
        metrics = OrderedDict()
        t0 = time.time()

        # compute gradient of loss wrt parameters
        self.disc_optimizer.zero_grad()
        loss.backward()

        t1 = time.time()

        if self.disc_clip_grad: # clip norm of parameter gradient
            models.clip_grad_norm(self.disc_model, self.disc_clip_grad)

        if compute_norm: # compute parameter gradient norm
            grad_norm = models.compute_grad_norm(self.disc_model)

        t2 = time.time()

        if update: # descend gradient on parameters
            self.disc_optimizer.step()
            self.disc_iter += 1
        
        t3 = time.time()
        if compute_norm:
            metrics['disc_grad_norm'] = grad_norm
        metrics['backward_grad_time'] = t1 - t0
        metrics['backward_norm_time'] = t2 - t1
        metrics['backward_update_time'] = t3 - t2
        return metrics

    def gen_step(
        self, grid_type, update=True, compute_norm=True, batch_idx=0
    ):
        '''
        Perform a single forward-backward pass
        on the generative model, optionally
        updating model parameters and/or comp-
        uting the parameter gradient norm.
        '''
        idx = (
            self.gen_iter, self.disc_iter, 'train',
            'gen', grid_type, batch_idx,
        )
        need_gradient = (update or compute_norm)
        torch.cuda.reset_max_memory_allocated()
        t0 = time.time()

        # forward pass
        loss, metrics = self.gen_forward(self.train_data, grid_type)

        if self.sync_cuda:
            torch.cuda.synchronize()
        m1 = torch.cuda.max_memory_allocated()
        torch.cuda.reset_max_memory_allocated()
        t1 = time.time()

        if need_gradient: # backward pass
            metrics.update(self.gen_backward(loss, update, compute_norm))           

        if self.sync_cuda:
            torch.cuda.synchronize()
        m2 = torch.cuda.max_memory_allocated()
        t2 = time.time()

        metrics['memory'] = get_memory_used() / MB
        metrics['forward_time'] = t1 - t0
        metrics['forward_gpu'] = m1 / MB
        metrics['backward_time'] = t2 - t1
        metrics['backward_gpu'] = m2 / MB
        self.insert_metrics(idx, metrics)

        metrics = self.metrics.loc[idx]
        self.print_metrics(idx[:-1], metrics)

        assert not loss.isnan(), 'generator loss is nan'
        if compute_norm:
            grad_norm = metrics['gen_grad_norm']
            assert not np.isnan(grad_norm), 'generator gradient is nan'
            assert not np.isclose(0, grad_norm), 'generator gradient is zero'

        return metrics

    def disc_step(
        self, grid_type, update=True, compute_norm=True, batch_idx=0
    ):
        '''
        Perform a single forward-backward pass
        on the discriminative model, optionally
        updating model parameters and/or comp-
        uting the parameter gradient norm.
        '''
        idx = (
            self.gen_iter, self.disc_iter, 'train',
            'disc', grid_type, batch_idx,
        )
        need_gradient = (update or compute_norm)
        torch.cuda.reset_max_memory_allocated()
        t0 = time.time()

        # forward pass
        loss, metrics = self.disc_forward(self.train_data, grid_type)

        if self.sync_cuda:
            torch.cuda.synchronize()
        m1 = torch.cuda.max_memory_allocated()
        torch.cuda.reset_max_memory_allocated()
        t1 = time.time()
        
        if need_gradient: # backward pass
            metrics.update(self.disc_backward(loss, update, compute_norm))

        if self.sync_cuda:
            torch.cuda.synchronize()
        m2 = torch.cuda.max_memory_allocated()
        t2 = time.time()

        metrics['memory'] = get_memory_used() / MB
        metrics['forward_time'] = t1 - t0
        metrics['forward_gpu'] = m1 / MB
        metrics['backward_time'] = t2 - t1
        metrics['backward_gpu'] = m2 / MB
        self.insert_metrics(idx, metrics)

        metrics = self.metrics.loc[idx]
        self.print_metrics(idx[:-1], metrics)

        assert not loss.isnan(), 'discriminator loss is nan'
        if compute_norm:
            grad_norm = metrics['disc_grad_norm']
            assert not np.isnan(grad_norm), 'discriminator gradient is nan'
            #assert not np.isclose(0, grad_norm), 'discriminator gradient is zero'

        return metrics

    def test_model(self, n_batches, model_type, fit_atoms=False):
        '''
        Evaluate a model's performance on
        n_batches of test data, optionally
        performing atom fitting.
        '''
        valid_model_types = {'gen'}
        if self.has_disc_model:
            valid_model_types.add('disc')
        test_disc = (model_type == 'disc')

        for i in range(n_batches):
            torch.cuda.reset_max_memory_allocated()
            t0 = time.time()

            if test_disc: # test discriminative model
                grid_type = self.get_disc_grid_phase(i)
                loss, metrics = self.disc_forward(
                    data=self.test_data, grid_type=grid_type
                )
            else: # test generative model
                grid_type = self.get_gen_grid_phase(i)
                loss, metrics = self.gen_forward(
                    data=self.test_data,
                    grid_type=grid_type,
                    fit_atoms=fit_atoms
                )

            metrics['memory'] = get_memory_used() / MB
            metrics['forward_time'] = time.time() - t0
            metrics['forward_gpu'] = torch.cuda.max_memory_allocated() / MB
            idx = (
                self.gen_iter, self.disc_iter,
                'test', model_type, grid_type, i
            )
            self.insert_metrics(idx, metrics)

        idx = idx[:-1]
        metrics = self.metrics.loc[idx].mean()
        self.print_metrics(idx, metrics)

    def test_models(self, n_batches, fit_atoms):
        '''
        Evaluate each model on n_batches of test
        data, optionally performing atom fitting.
        '''
        if self.has_disc_model:
            self.test_model(n_batches=n_batches, model_type='disc')

        self.test_model(
            n_batches=n_batches, model_type='gen', fit_atoms=fit_atoms
        )
        self.save_metrics()

    def train_model(
        self, n_iters, model_type, update=True, compute_norm=True
    ):
        '''
        Perform n_iters forward-backward passes
        on one of the models, optionally updating
        its parameters.
        '''
        valid_model_types = {'gen'}
        if self.has_disc_model:
            valid_model_types.add('disc')
        train_disc = (model_type == 'disc')

        for i in range(n_iters):
            batch_idx = 0 if update else i

            if train_disc: # train discriminative model
                grid_type = self.get_disc_grid_phase(batch_idx)

                metrics = self.disc_step(
                    grid_type=grid_type,
                    update=update,
                    compute_norm=compute_norm,
                    batch_idx=batch_idx
                )

                if grid_type == 'real':
                    disc_gan_loss = metrics.get('gan_loss', -1)
                    self.disc_gan_loss = disc_gan_loss

            else: # train generative model
                grid_type = self.get_gen_grid_phase(batch_idx)

                metrics = self.gen_step(
                    grid_type=grid_type,
                    update=update,
                    compute_norm=compute_norm,
                    batch_idx=batch_idx
                )

                gen_gan_loss = metrics.get('gan_loss', 0)
                self.gen_gan_loss = gen_gan_loss

    def train_models(self, update=True, compute_norm=False):
        '''
        Train each model on training data for
        a pre-determined number of iterations.
        '''
        if update: # determine which models to update

            if self.balance: # only update gen if disc is better
                update_disc = True
                update_gen = (self.disc_gan_loss < self.gen_gan_loss)

            else: # update both models
                update_disc = update_gen = True

        else: # don't update, just evaluate
            update_disc = update_gen = False

        if self.has_disc_model:

            self.train_model(
                n_iters=self.n_disc_train_iters,
                model_type='disc',
                update=update_disc,
                compute_norm=compute_norm
            )

        self.train_model(
            n_iters=self.n_gen_train_iters,
            model_type='gen',
            update=update_gen,
            compute_norm=compute_norm
        )

    @save_on_exception
    def train_and_test(
        self,
        max_iter,
        test_interval,
        n_test_batches,
        fit_interval,
        norm_interval,
        save_interval,
    ):
        init_iter = self.gen_iter
        last_save = None
        last_test = None
        divides = lambda d, n: (n % d == 0)

        while self.gen_iter <= max_iter:
            i = self.gen_iter
 
            # save model and optimizer states
            if last_save != i and divides(save_interval, i):
                self.save_state()
                last_save = i

            # test models on test data
            if last_test != i and divides(test_interval, i):
                fit_atoms = (fit_interval > 0 and divides(fit_interval, i))
                self.test_models(n_batches=n_test_batches, fit_atoms=fit_atoms)
                last_test = i

            # train models on training data
            update = (i < max_iter)
            compute_norm = (norm_interval > 0 and divides(norm_interval, i))
            self.train_models(update=update, compute_norm=compute_norm)

            if i == max_iter:
                break

        self.save_state_and_metrics()


class AESolver(GenerativeSolver):
    gen_model_type = models.AE


class VAESolver(GenerativeSolver):
    gen_model_type = models.VAE


class CESolver(GenerativeSolver):
    gen_model_type = models.CE


class CVAESolver(GenerativeSolver):
    gen_model_type = models.CVAE
    has_complex_input = True


class GANSolver(GenerativeSolver):
    gen_model_type = models.GAN
    has_disc_model = True


class CGANSolver(GenerativeSolver):
    gen_model_type = models.CGAN
    has_disc_model = True


class VAEGANSolver(GenerativeSolver):
    gen_model_type = models.VAE
    has_disc_model = True


class CVAEGANSolver(GenerativeSolver):
    gen_model_type = models.CVAE
    has_complex_input = True
    has_disc_model = True
