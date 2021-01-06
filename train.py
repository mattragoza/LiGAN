from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import sys, os, argparse, time
import collections
import itertools
import datetime as dt
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('whitegrid')

import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

import molgrid
import generate
import caffe_util as cu
from results import plot_lines


def CaffeGAN(object):

    def __init__(
        self,
        batch_size,
        gen_net_param,
        disc_net_param,
        solver_param,
        random_seed,
        out_prefix,
    ):
        # create generative net solver
        self.gen_prefix = out_prefix + '_gen'
        self.gen = cu.CaffeSolver(
            param=solver_param,
            net_param=gen_net_param,
            random_seed=random_seed,
            snapshot_prefix=self.gen_prefix,
            scaffold=False,
        )

        # create discriminative net solver
        disc_net_param.force_backward = True
        self.disc_prefix = out_prefix + '_disc'
        self.disc = cu.CaffeSolver(
            param=solver_param,
            net_param=disc_net_param,
            random_seed=random_seed,
            snapshot_prefix=self.disc_prefix,
            scaffold=False,
        )

        # write and plot out train and test metrics
        self.metrics_file = out_prefix + '.train_metrics'
        self.plot_file = out_prefix + '.png'

    def scaffold(
        self,
        cont_iter=None,
        gen_weights=None,
        disc_weights=None,
    ):
        if cont_iter:
            state_suffix = '_iter_{}.solverstate'.format(cont_iter)
            gen_state = self.gen_prefix + state_suffix
            disc_state = self.disc_prefix + state_suffix
        else:
            gen_state = disc_state = None

        self.gen.scaffold(gen_state, gen_weights)
        self.disc.scaffold(disc_state, disc_weights)

        if cont_iter:
            self.metrics = pd.read_csv(
                self.metrics_file, sep=' ', header=0, index_col=[0,1]
            )[:cont_iter+1]
            self.curr_iter = cont_iter

        else:
            cols = ['iteration', 'phase']
            self.metrics = pd.DataFrame(columns=cols).set_index(cols)
            self.curr_iter = 0

        self.gen_net = generate.MolGridGenerator(self.gen.net)

    def snapshot(self):
        self.gen.snapshot()
        self.disc.snapshot()

    def test(self, data, n_iter):
        
        for i in range(n_iter):
            self.disc_step(data, n_iter, train=False)

        for i in range(n_iter):
            self.gen_step(data, n_iter, train=False)

    def balance(self):
        return True, True # TODO get from loss df

    def disc_step(self, data, real, train):
        '''
        Perform one forward-backward pass on discriminator
        and optionally update its weights. Can use real or
        generated data.
        '''
        # get real densities
        rec, lig = data.forward()

        if not real: # generate densities
            lig = self.gen_net.forward(rec=rec, lig=lig)
        
        self.disc.net.forward(rec=rec, lig=lig, label=real)
        self.disc.net.clear_param_diffs()
        self.disc.net.backward()

        # TODO record metrics

        if train:
            self.disc.apply_update()
            self.disc.increment_iter()

    def gen_step(self, data, train):
        '''
        Perform one forward-backward pass on generator
        and optionally update its weights.
        '''
        # get real densities
        rec, lig = data.forward()

        # generate densities
        lig_gen = self.gen_net.forward(rec=rec, lig=lig)

        # compute adversarial loss
        self.disc.net.forward(rec=rec, lig=lig_gen, label=True)
        self.disc.net.clear_param_diffs()
        self.disc.net.backward()

        # copy disc input gradient to gen output
        self.gen.net.blobs['lig_gen'].diff[...] = \
            self.disc.net.blobs['lig'].diff

        self.gen.net.clear_param_diffs()
        self.gen.net.backward()

        # TODO record metrics

        if train:
            self.gen.apply_update()
            self.gen.increment_iter()

    def train_disc(self, data, n_iter):

        for i in range(n_iter):
            real = (i%2 == 0)
            self.disc_step(data, real, train=True)

    def train_gen(self, data, n_iter):

        for i in range(n_iter):
            self.gen_step(data, train=True)

    def train(
        self,
        train_data,
        test_data,
        n_iter,
        gen_train_iter=2,
        disc_train_iter=2,
        test_interval=10,
        test_iter=10,
        snapshot=10000,
        alternate=False,
        balance=False,
    ):
        train_gen = True
        train_disc = True

        for i in range(self.curr_iter, n_iter+1):

            if i % snapshot == 0:
                self.snapshot()

            if i % test_interval == 0:
                self.test(train_data, test_iter)
                self.test(test_data, test_iter)

            if i == n_iter:
                return

            if train_disc:
                self.train_disc(train_data, disc_train_iter)

            if train_gen:
                self.train_gen(train_data, gen_train_iter)

            if balance: # dynamic G/D loss balancing
                train_gen, train_disc = self.balance()

            self.curr_iter += 1


def get_gradient_norm(net, ord=2):
    '''
    Compute the overall norm of blob diffs in a net.
    '''
    grad_norm = 0.0
    for blob_vec in net.params.values():
        for blob in blob_vec:
            grad_norm += (abs(blob.diff)**ord).sum()
    return grad_norm**(1/ord)


def gradient_normalize(net, ord=2):
    '''
    Divide all blob diffs by the gradient norm.
    '''
    grad_norm = get_gradient_norm(net, ord)
    if grad_norm > 1.0:
        for blob_vec in net.params.values():
            for blob in blob_vec:
                blob.diff[...] /= grad_norm


def normalize(x, ord=2):
    '''
    Divide input by its norm.
    '''
    return x / np.linalg.norm(x, ord)


def spectral_power_iterate(W, u, n_iter):
    '''
    Estimate the singular vectors and spectral norm
    (largest singular value) of a matrix W, starting
    from initial vector u, by n_iter power iterations.
    '''
    W = W.reshape(W.shape[0], -1) # treat as matrix

    for i in range(n_iter):
        v = normalize(np.matmul(W.T, u))
        u = normalize(np.matmul(W, v))

    sigma = np.matmul(u.T, np.matmul(W, v))
    return u, v, sigma


def spectral_norm_setup(net):
    '''
    Initialize a param dict for a net that maps names of
    layers with weight params to tuples (u, v, sigma) of
    params to be used for spectral normalization.
    '''
    params = collections.OrderedDict()
    for layer in net.params:
        W = net.params[layer][0]
        u = np.random.normal(0, 1, W.shape[0])
        u, v, sigma = spectral_power_iterate(W.data, u, 10)
        params[layer] = (u, v, sigma)

    return params


def spectral_norm_forward(net, params):
    '''
    Perform one power iteration on spectral norm params
    and then divide net weights by their spectral norm.
    Update spectral norm params dict with new estimates.
    '''
    for layer in net.params:
        W = net.params[layer][0]
        u, v, sigma = params[layer]
        u, v, sigma = spectral_power_iterate(W.data, u, 1)
        W.data[...] /= sigma
        params[layer] = (u, v, sigma)


def spectral_norm_backward(net, params):
    '''
    Replace diffs of net weights with diffs
    of spectral-normalized weights.
    '''
    for layer in net.params:
        W = net.params[layer][0]
        y = net.blobs[layer]
        u, v, sigma = params[layer]
        lambda_ = np.sum(y.diff * y.data) / y.shape[0]
        W.diff[...] -= (lambda_ * np.outer(u, v)).reshape(W.shape)
        W.diff[...] /= sigma


def disc_step(data, gen, disc, n_iter, args, train, compute_metrics):
    '''
    Train or test GAN discriminator for n_iter iterations.
    '''
    disc_loss_names = [b for b in disc.net.blobs if b.endswith('loss')]

    if args.alternate: # find latent variable blob names for prior sampling
        latent_mean = generate.find_blobs_in_net(gen.net, r'.+_latent_mean')[0]
        latent_std = generate.find_blobs_in_net(gen.net, r'.+_latent_std')[0]
        latent_noise = generate.find_blobs_in_net(gen.net, r'.+_latent_noise')[0]

        # train on prior samples every 4th iteration (real, posterior, real, prior)
        n_iter *= 2

    metrics = collections.defaultdict(lambda: np.full(n_iter, np.nan))

    for i in range(n_iter):

        real = i%2 == 0
        prior = args.alternate and i%4 == 3

        if real: # get real receptors and ligands

            data.forward()
            rec = data.blobs['rec'].data
            lig = data.blobs['lig'].data

            disc.net.blobs['rec'].data[...] = rec
            disc.net.blobs['lig'].data[...] = lig
            disc.net.blobs['label'].data[...] = 1.0

        else: # generate fake ligands

            # reuse rec and lig from last real forward pass
            gen.net.blobs['rec'].data[...] = rec
            gen.net.blobs['lig'].data[...] = lig

            if args.gen_spectral_norm:
                spectral_norm_forward(gen.net, args.gen_spectral_norm)

            if not prior: # posterior

                gen.net.forward()

            else: # prior

                gen.net.blobs[latent_mean].data[...] = 0.0
                gen.net.blobs[latent_std].data[...] = 1.0
                gen.net.forward(start=latent_noise) # assumes cond branch is after latent space

            lig_gen = gen.net.blobs['lig_gen'].data

            disc.net.blobs['rec'].data[...] = rec
            disc.net.blobs['lig'].data[...] = lig_gen
            disc.net.blobs['label'].data[...] = 0.0

        if args.instance_noise:
            noise = np.random.normal(0, args.instance_noise, lig.shape)
            disc.net.blobs['lig'].data[...] += noise

        if args.disc_spectral_norm:
            spectral_norm_forward(disc.net, args.disc_spectral_norm)

        disc.net.forward()

        # record discriminator loss
        for l in disc_loss_names:

            loss = float(disc.net.blobs[l].data)
            metrics['disc_' + l][i] = loss

            if args.alternate: # also record separate prior and posterior GAN losses
                if prior:
                    metrics['disc_prior_' + l][i] = loss
                else:
                    metrics['disc_post_' + l][i] = loss
        
        metrics['disc_iter'][i] = disc.iter

        if train or compute_metrics: # compute gradient

            disc.net.clear_param_diffs()
            disc.net.backward()

            if args.disc_spectral_norm:
                spectral_norm_backward(disc.net, args.disc_spectral_norm)

            if args.disc_grad_norm:
                gradient_normalize(disc.net)

            if compute_metrics:
                metrics['disc_grad_norm'][i] = get_gradient_norm(disc.net)

            if train:
                disc.apply_update()

    return {m: np.nanmean(metrics[m]) for m in metrics}


def gen_step(data, gen, disc, n_iter, args, train, compute_metrics):
    '''
    Train or test the GAN generator for n_iter iterations.
    '''

    # find loss blob names for recording loss output
    gen_loss_names  = [b for b in gen.net.blobs if b.endswith('loss')]
    disc_loss_names = [b for b in disc.net.blobs if b.endswith('loss')]

    # keep track of generator loss weights
    loss_weights = dict((l,float(gen.net.blobs[l].diff[...])) for l in gen_loss_names)

    if args.alternate: # find latent variable blob names for prior sampling
        latent_mean = generate.find_blobs_in_net(gen.net, r'.+_latent_mean')[0]
        latent_std = generate.find_blobs_in_net(gen.net, r'.+_latent_std')[0]
        latent_noise = generate.find_blobs_in_net(gen.net, r'.+_latent_noise')[0]

        # train on prior samples every other iteration
        n_iter *= 2

    metrics = collections.defaultdict(lambda: np.full(n_iter, np.nan))

    for i in range(n_iter):

        prior = args.alternate and i%2 == 1

        # generate fake ligands
        if not prior: # from posterior

            # get real receptors and ligands
            data.forward()
            rec = data.blobs['rec'].data
            lig = data.blobs['lig'].data

            gen.net.blobs['rec'].data[...] = rec
            gen.net.blobs['lig'].data[...] = lig

            if 'cond_rec' in gen.net.blobs:
                gen.net.blobs['cond_rec'].data[...] = rec

            if args.gen_spectral_norm:
                spectral_norm_forward(gen.net, args.gen_spectral_norm)

            gen.net.forward()

        else: # from prior

            # for prior sampling, set the latent mean to 0.0 and std to 1.0
            # and then call net.forward() from the noise source onwards

            # this assumes only one variational latent space in the net
            # and only samples the prior on the first one if there are multiple

            # for CVAEs, reuse the same recs as the last posterior forward pass
            # this will forward the conditional branch again if and only if
            # it's located after the encoder branch in the model file

            if args.gen_spectral_norm:
                spectral_norm_forward(gen.net, args.gen_spectral_norm)

            gen.net.blobs[latent_mean].data[...] = 0.0
            gen.net.blobs[latent_std].data[...] = 1.0
            gen.net.forward(start=latent_noise)

        lig_gen = gen.net.blobs['lig_gen'].data

        # cross_entropy_loss(y_t, y_p) = -[y_t*log(y_p) + (1 - y_t)*log(1 - y_p)]
        # for original minmax GAN loss, set lig_gen label = 0.0 and ascend gradient
        # for non-saturating GAN loss, set lig_gen label = 1.0 and descend gradient

        disc.net.blobs['rec'].data[...] = rec
        disc.net.blobs['lig'].data[...] = lig_gen
        disc.net.blobs['label'].data[...] = 1.0

        if args.instance_noise:
            noise = np.random.normal(0, args.instance_noise, lig.shape)
            disc.net.blobs['lig'].data[...] += noise

        if args.disc_spectral_norm:
            spectral_norm_forward(disc.net, args.disc_spectral_norm)

        disc.net.forward()

        # record generator loss
        if not prior:
            for l in gen_loss_names:
                loss = float(gen.net.blobs[l].data)
                metrics['gen_' + l][i] = loss

        # record discriminator loss
        for l in disc_loss_names:

            loss = float(disc.net.blobs[l].data)
            metrics['gen_adv_' + l][i] = loss

            if args.alternate: # also record separate prior and posterior loss
                if prior:
                    metrics['gen_adv_prior_' + l][i] = loss
                else:
                    metrics['gen_adv_post_' + l][i] = loss

        metrics['gen_iter'][i] = gen.iter

        if train or compute_metrics: # compute gradient

            disc.net.clear_param_diffs()
            disc.net.backward()

            if args.disc_spectral_norm:
                spectral_norm_backward(disc.net, args.disc_spectral_norm)

            if args.disc_grad_norm:
                gradient_normalize(disc.net)

            gen.net.blobs['lig_gen'].diff[...] = disc.net.blobs['lig'].diff
            gen.net.clear_param_diffs()

            # set non-GAN loss weights
            for l, w in loss_weights.items():
                gen.net.blobs[l].diff[...] = 0 if prior else w * args.loss_weight

            if prior: # only backprop gradient to noise source (what about cond branch??)
                gen.net.backward(end=latent_noise)
                gen.net.blobs[latent_mean].diff[...] = 0.0
                gen.net.blobs[latent_std].diff[...] = 0.0
                gen.net.backward(start=latent_std)

                lig_grad_norm = np.linalg.norm(gen.net.blobs['lig'].diff)
                assert np.isclose(lig_grad_norm, 0), lig_grad_norm
    
            else:
                gen.net.backward()

            if args.gen_spectral_norm:
                spectral_norm_backward(gen.net, args.gen_spectral_norm)

            if args.gen_grad_norm:
                gradient_normalize(gen.net)

            if compute_metrics:
                metrics['gen_grad_norm'][i] = get_gradient_norm(gen.net)
                metrics['gen_adv_grad_norm'][i] = get_gradient_norm(disc.net)
                metrics['gen_loss_weight'][i] = args.loss_weight

            if train:
                gen.apply_update()

    return {m: np.nanmean(metrics[m]) for m in metrics}


def insert_metrics(loss_df, iter_, phase, metrics):

    for m in metrics:
        loss_df.loc[(iter_, phase), m] = metrics[m]


def write_and_plot_metrics(loss_df, loss_file, plot_file):

    loss_df.to_csv(loss_file, sep=' ')
    fig = plot_lines(plot_file, loss_df, x='iteration', y=loss_df.columns, hue='phase')
    plt.close(fig)


def train_GAN_model(train_data, test_data, gen, disc, loss_df, loss_file, plot_file, args):
    '''
    Train a GAN using the provided train_data net, gen solver, and disc solver.
    Return loss_df of metrics evaluated on train and test data, while also writing
    to loss_file and plotting to plot_file as training progresses.
    '''
    # training flags for dynamic balancing
    train_disc = True
    train_gen  = True

    # init spectral norm params
    if args.disc_spectral_norm:
        args.disc_spectral_norm = spectral_norm_setup(disc.net)

    if args.gen_spectral_norm:
        args.gen_spectral_norm = spectral_norm_setup(gen.net)

    test_times = []
    train_times = []

    for i in range(args.cont_iter, args.max_iter+1):

        if i % args.snapshot == 0:
            disc.snapshot()
            gen.snapshot()

        if i % args.test_interval == 0: # test nets
            t_start = time.time()

            for d in test_data:

                disc_metrics = disc_step(test_data[d], gen, disc, args.test_iter, args,
                                         train=False, compute_metrics=True)

                gen_metrics  = gen_step(test_data[d], gen, disc, args.test_iter, args,
                                        train=False, compute_metrics=True)

                insert_metrics(loss_df, i, d, disc_metrics)
                insert_metrics(loss_df, i, d, gen_metrics)

            test_times.append(time.time() - t_start)

            t_train = np.sum(train_times)
            t_test = np.sum(test_times)
            t_total = t_train + t_test
            pct_train = 100*t_train/t_total
            pct_test = 100*t_test/t_total
            t_per_iter = t_total/(i - args.cont_iter)
            t_left = t_per_iter * (args.max_iter - i)
            t_total = dt.timedelta(seconds=t_total)
            if i > args.cont_iter:
                t_per_iter = dt.timedelta(seconds=t_per_iter)
                t_left = dt.timedelta(seconds=t_left)

            print('Iteration {} / {}'.format(i, args.max_iter))
            print('  {} elapsed ({:.1f}% training, {:.1f}% testing)'
                  .format(t_total, pct_train, pct_test))
            print('  {} left (~{} / iteration)'.format(t_left, t_per_iter))
            for d in test_data:
                for m in sorted(loss_df.columns):
                    print('  {} {} = {}'.format(d, m, loss_df.loc[(i, d), m]))

            write_and_plot_metrics(loss_df, loss_file, plot_file)

        if i == args.max_iter: # return after final test evaluation
            return

        t_start = time.time()

        # train nets
        if train_disc:
            disc_step(train_data, gen, disc, args.disc_train_iter, args,
                      train=True, compute_metrics=False)

        if train_gen:
            gen_step(train_data, gen, disc, args.gen_train_iter, args,
                     train=train_gen, compute_metrics=False)

        if i+1 == args.max_iter:
            train_disc = False
            train_gen = False

        elif args.balance: # dynamically balance G/D training

            # how much better is D than G?
            if 'disc_wass_loss' in disc_metrics:
                train_gen_loss = gen_metrics['gen_adv_wass_loss']
                train_disc_loss = disc_metrics['disc_wass_loss']
                train_loss_balance = train_gen_loss - train_disc_loss
            else:
                train_gen_loss = gen_metrics['gen_adv_log_loss']
                train_disc_loss = disc_metrics['disc_log_loss']
                train_loss_balance = train_gen_loss / train_disc_loss

            if train_disc and train_loss_balance > 10:
                train_disc = False
            if not train_disc and train_loss_balance < 2:
                train_disc = True

            if train_gen and train_loss_balance < 1:
                train_gen = False
            if not train_gen and train_loss_balance > 2:
                train_gen = True

        # update non-GAN generator loss weight
        if args.loss_weight_decay:
            args.loss_weight *= (1.0 - args.loss_weight_decay)

        train_times.append(time.time() - t_start)

        disc.increment_iter()
        gen.increment_iter()


def get_crossval_data_files(data_prefix, fold_num, ext='.types'):
    '''
    Return train and test data files for a given
    fold number of a k-fold cross-validation split.
    If fold_num == 'all', return the full data set
    as both the train and test data file.
    '''
    if fold_num == 'all':
        train_data_file = test_data_file = data_prefix + ext
    else:
        fold_suffix = str(fold_num) + ext
        train_data_file = data_prefix + 'train' + fold_suffix
        test_data_file = data_prefix + 'test' + fold_suffix
    return train_data_file, test_data_file


def parse_args(argv):
    parser = argparse.ArgumentParser(description='train ligand GAN models with Caffe')
    parser.add_argument('-o', '--out_prefix', default='', help='common prefix for all output files')
    parser.add_argument('-d', '--data_model_file', required=True, help='prototxt file for reading data')
    parser.add_argument('-g', '--gen_model_file', required=True, help='prototxt file for generative model')
    parser.add_argument('-a', '--disc_model_file', required=True, help='prototxt file for discriminative model')
    parser.add_argument('-s', '--solver_file', required=True, help='prototxt file for solver hyperparameters')
    parser.add_argument('-p', '--data_prefix', required=True, help='prefix for data train/test fold files')
    parser.add_argument('-n', '--fold_nums', default='0,1,2,all', help='comma-separated fold numbers to run (default 0,1,2,all)')
    parser.add_argument('-r', '--data_root', required=True, help='root directory of data files (prepended to paths in train/test fold files)')
    parser.add_argument('--random_seed', default=0, type=int, help='random seed for Caffe initialization and training (default 0)')
    parser.add_argument('--gen_weights_file', help='.caffemodel file to initialize gen weights')
    parser.add_argument('--disc_weights_file', help='.caffemodel file to initialize disc weights')
    parser.add_argument('--cont_iter', default=0, type=int, help='continue training from iteration #')
    parser.add_argument('--max_iter', default=100000, type=int, help='total number of train iterations (default 10000)')
    parser.add_argument('--gen_train_iter', default=2, type=int, help='number of sub-iterations to train gen model each train iter (default 20)')
    parser.add_argument('--disc_train_iter', default=2, type=int, help='number of sub-iterations to train disc model each train iter (default 20)')
    parser.add_argument('--test_interval', default=10, type=int, help='evaluate test data every # train iters (default 10)')
    parser.add_argument('--test_iter', default=10, type=int, help='number of iterations of each test data evaluation (default 10)')
    parser.add_argument('--snapshot', default=10000, type=int, help='save .caffemodel weights and solver state every # train iters (default 1000)')
    parser.add_argument('--alternate', default=False, action='store_true', help='alternate between encoding and sampling latent prior')
    parser.add_argument('--balance', default=False, action='store_true', help='dynamically train gen/disc each iter by balancing GAN loss')
    parser.add_argument('--instance_noise', type=float, default=0.0, help='standard deviation of disc instance noise (default 0.0)')
    parser.add_argument('--gen_grad_norm', default=False, action='store_true', help='gen gradient normalization')
    parser.add_argument('--disc_grad_norm', default=False, action='store_true', help='disc gradient normalization')
    parser.add_argument('--gen_spectral_norm', default=False, action='store_true', help='gen spectral normalization')
    parser.add_argument('--disc_spectral_norm', default=False, action='store_true', help='disc spectral normalization')
    parser.add_argument('--loss_weight', default=1.0, type=float, help='initial value for non-GAN generator loss weight')
    parser.add_argument('--loss_weight_decay', default=0.0, type=float, help='decay rate for non-GAN generator loss weight')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    # read data params
    data_net_param = cu.NetParameter.from_prototxt(args.data_model_file)
    assert data_net_param.layer[0].type == 'MolGridData'
    data_param = data_net_param.layer[0].molgrid_data_param

    # read model and solver params
    gen_param = cu.NetParameter.from_prototxt(args.gen_model_file)
    disc_param = cu.NetParameter.from_prototxt(args.disc_model_file)
    solver_param = cu.SolverParameter.from_prototxt(args.solver_file)

    for fold_num in args.fold_nums.split(','):

        train_data_file, test_data_file = get_crossval_data_files(
            args.data_prefix, fold_num
        )

        train_data = MolGridData.from_param(data_param)
        train_data.populate(train_data_file)

        test_data = MolGridData.from_param(data_param)
        test_data.populate(test_data_file)

        gan = CaffeGAN(
            batch_size=args.batch_size,
            gen_net_param=gen_param,
            disc_net_param=disc_param,
            solver_param=solver_param,
            random_seed=args.random_seed,
            out_prefix=args.out_prefix + '_' + str(fold),
        )

        gan.scaffold(
            cont_iter=args.cont_iter,
            gen_weights=args.gen_weights_file,
            disc_weights=args.disc_weights_file,
        )

        try:
            gan.train(
                train_data=train_data,
                test_data=test_data,
                n_iter=args.max_iter,
                gen_train_iter=args.gen_train_iter,
                disc_train_iter=args.disc_train_iter,
                test_interval=args.test_interval,
                test_iter=args.test_iter,
                snapshot=args.snapshot,
                alternate=args.alternate,
                balance=args.balance,
            )
        except:
            gan.snapshot()
            raise


if __name__ == '__main__':
    main(sys.argv[1:])

