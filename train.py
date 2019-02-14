from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import sys, os, argparse, time
import collections
import itertools
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('whitegrid')

import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

from caffe_util import NetParameter, SolverParameter, Net, Solver
from results import plot_lines


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

    metrics = collections.defaultdict(lambda: np.full(n_iter, np.nan))

    for i in range(n_iter):

        if i % 2 == 0: # get real receptors and ligands

            data.forward()
            rec = data.blobs['rec'].data
            lig = data.blobs['lig'].data

            disc.net.blobs['rec'].data[...] = rec
            disc.net.blobs['lig'].data[...] = lig
            disc.net.blobs['label'].data[...] = 1.0

        else: # generate fake ligands

            gen.net.blobs['rec'].data[...] = rec
            gen.net.blobs['lig'].data[...] = lig

            if args.gen_spectral_norm:
                spectral_norm_forward(gen.net, args.gen_spectral_norm)

            gen.net.forward()
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
            metrics['disc_' + l][i] = float(disc.net.blobs[l].data)
        
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
    gen_loss_names  = [b for b in gen.net.blobs if b.endswith('loss')]
    disc_loss_names = [b for b in disc.net.blobs if b.endswith('loss')]
    lig_grid_names = ['lig', 'lig_gen']

    metrics = collections.defaultdict(lambda: np.full(n_iter, np.nan))

    # set loss weights
    for l in gen_loss_names:
        gen.net.blobs[l].diff[...] = args.loss_weight

    for i in range(n_iter):

        # get real receptors and ligands
        data.forward()
        rec = data.blobs['rec'].data
        lig = data.blobs['lig'].data

        # generate fake ligands
        gen.net.blobs['rec'].data[...] = rec
        gen.net.blobs['lig'].data[...] = lig

        if args.gen_spectral_norm:
            spectral_norm_forward(gen.net, args.gen_spectral_norm)

        gen.net.forward()
        lig_gen = gen.net.blobs['lig_gen'].data

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
        for l in gen_loss_names:
            metrics['gen_' + l][i] = float(gen.net.blobs[l].data)

        # record discriminator loss
        for l in disc_loss_names:
            metrics['gen_adv_' + l][i] = float(disc.net.blobs[l].data)

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
            gen.net.backward()

            if args.gen_spectral_norm:
                spectral_norm_backward(gen.net, args.gen_spectral_norm)

            if args.gen_grad_norm:
                gradient_normalize(gen.net)

            if compute_metrics:
                metrics['gen_grad_norm'][i] = get_gradient_norm(gen.net)
                metrics['gen_adv_grad_norm'][i] = get_gradient_norm(disc.net)

            if train:
                gen.apply_update()

    if compute_metrics: # compute additional grid metrics

        grid_axes = (1, 2, 3, 4)
        for g in lig_grid_names:
            grids = gen.net.blobs[g].data
            metrics[g + '_norm'][-1] = ((grids**2).sum(grid_axes)**0.5).mean()

        for g, g2 in itertools.combinations(lig_grid_names, 2):
            grids  = gen.net.blobs[g].data
            grids2 = gen.net.blobs[g2].data
            diffs =  grids2 - grids
            metrics[g + '_' + g2 + '_dist'][-1] = ((diffs**2).sum(grid_axes)**0.5).mean()

        for g in lig_grid_names:
            grids = gen.net.blobs[g].data
            diffs = grids[np.newaxis,:,...] - grids[:,np.newaxis,...]
            metrics[g + '_' + g + '_dist'][-1] = ((diffs**2).sum(grid_axes)**0.5).mean()

    metrics['gen_loss_weight'][-1] = args.loss_weight

    return {m: np.nanmean(metrics[m]) for m in metrics}


def insert_metrics(loss_df, iter_, test_data, metrics):

    for m in metrics:
        loss_df.loc[(iter_, test_data), m] = metrics[m]


def write_and_plot_metrics(loss_df, loss_out, plot_out):

    loss_out.seek(0)
    loss_df.to_csv(loss_out, sep=' ')
    loss_out.flush()

    plot_out.seek(0)
    plot_lines(plot_out, loss_df, x='iteration', y=loss_df.columns, hue='test_data')
    plot_out.flush()


def train_GAN_model(train_data, test_data, gen, disc, loss_df, loss_out, plot_out, args):
    '''
    Train a GAN using the provided train_data net, gen solver, and disc solver.
    Return loss_df of metrics evaluated on train and test data, while also writing
    to loss_out and plotting to plot_out as training progresses.
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

            write_and_plot_metrics(loss_df, loss_out, plot_out)

        if i == arg.max_iter: # return after final test evaluation
            return
        
        t_start = time.time()

        # train nets       
        disc_step(train_data, gen, disc, args.disc_train_iter, args,
                  train=train_disc, compute_metrics=False)

        gen_step(train_data, gen, disc, args.gen_train_iter, args,
                 train=train_gen, compute_metrics=False)

        if i+1 == args.max_iter:
            train_disc = False
            train_gen = False

        elif args.balance: # dynamically balance G/D training

            # how much better is D than G?
            train_loss_ratio = gen_metrics['gen_adv_log_loss'] / disc_metrics['disc_log_loss']

            if train_disc and train_loss_ratio > 10:
                train_disc = False
            if not train_disc and train_loss_ratio < 2:
                train_disc = True
            if train_gen and train_loss_ratio < 1:
                train_gen = False
            if not train_gen and train_loss_ratio > 2:
                train_gen = True

        # update non-GAN generator loss weight
        if args.loss_weight_decay:
            args.loss_weight *= (1.0 - args.loss_weight_decay)

        train_times.append(time.time() - t_start)

        disc.increment_iter()
        gen.increment_iter()


def get_train_and_test_files(data_prefix, fold_nums):
    '''
    Yield tuples of fold name, train file, test file from a
    data_prefix and comma-delimited fold_nums string.
    '''
    for fold in fold_nums.split(','):
        if fold == 'all':
            train_file = test_file = '{}.types'.format(data_prefix)
        else:
            train_file = '{}train{}.types'.format(data_prefix, fold)
            test_file = '{}test{}.types'.format(data_prefix, fold)
        yield fold, train_file, test_file


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
    parser.add_argument('--max_iter', default=10000, type=int, help='total number of train iterations (default 10000)')
    parser.add_argument('--snapshot', default=1000, type=int, help='save .caffemodel weights and solver state every # train iters (default 1000)')
    parser.add_argument('--test_interval', default=10, type=int, help='evaluate test data every # train iters (default 10)')
    parser.add_argument('--test_iter', default=10, type=int, help='number of iterations of each test data evaluation (default 10)')
    parser.add_argument('--gen_train_iter', default=20, type=int, help='number of sub-iterations to train gen model each train iter (default 20)')
    parser.add_argument('--disc_train_iter', default=20, type=int, help='number of sub-iterations to train disc model each train iter (default 20)')
    parser.add_argument('--cont_iter', default=0, type=int, help='continue training from iteration #')
    parser.add_argument('--alternate', default=False, action='store_true', help='alternate between encoding and sampling latent prior')
    parser.add_argument('--balance', default=False, action='store_true', help='dynamically train gen/disc each iter by balancing GAN loss')
    parser.add_argument('--instance_noise', type=float, default=0.0, help='standard deviation of disc instance noise (default 0.0)')
    parser.add_argument('--gen_grad_norm', default=False, action='store_true', help='gen gradient normalization')
    parser.add_argument('--disc_grad_norm', default=False, action='store_true', help='disc gradient normalization')
    parser.add_argument('--gen_spectral_norm', default=False, action='store_true', help='gen spectral normalization')
    parser.add_argument('--disc_spectral_norm', default=False, action='store_true', help='disc spectral normalization')
    parser.add_argument('--gen_weights_file', help='.caffemodel file to initialize gen weights')
    parser.add_argument('--disc_weights_file', help='.caffemodel file to initialize disc weights')
    parser.add_argument('--loss_weight', default=1.0, type=float, help='initial value for non-GAN generator loss weight')
    parser.add_argument('--loss_weight_decay', default=0.0, type=float, help='decay rate for non-GAN generator loss weight')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    # read solver and model param files and set general params
    data_param = NetParameter.from_prototxt(args.data_model_file)

    gen_param = NetParameter.from_prototxt(args.gen_model_file)
    disc_param = NetParameter.from_prototxt(args.disc_model_file)
    
    gen_param.force_backward = True
    disc_param.force_backward = True

    solver_param = SolverParameter.from_prototxt(args.solver_file)
    solver_param.max_iter = args.max_iter
    solver_param.test_interval = args.max_iter + 1
    solver_param.random_seed = args.random_seed

    for fold, train_file, test_file in get_train_and_test_files(args.data_prefix, args.fold_nums):

        # create nets for producing train and test data
        data_param.set_molgrid_data_source(train_file, args.data_root)
        train_data = Net.from_param(data_param, phase=caffe.TRAIN)

        test_data = {}
        data_param.set_molgrid_data_source(train_file, args.data_root)
        test_data['train'] = Net.from_param(data_param, phase=caffe.TEST)
        if test_file != train_file:
            data_param.set_molgrid_data_source(test_file, args.data_root)
            test_data['test'] = Net.from_param(data_param, phase=caffe.TEST)

        # create solver for training generator net
        gen_prefix = '{}.{}_gen'.format(args.out_prefix, fold)
        gen = Solver.from_param(solver_param, net_param=gen_param, snapshot_prefix=gen_prefix)
        if args.gen_weights_file:
            gen.net.copy_from(args.gen_weights_file)
        if 'lig_gauss_conv' in gen.net.blobs:
            gen.net.copy_from('lig_gauss_conv.caffemodel')

        # create solver for training discriminator net
        disc_prefix = '{}.{}_disc'.format(args.out_prefix, fold)
        disc = Solver.from_param(solver_param, net_param=disc_param, snapshot_prefix=disc_prefix)
        if args.disc_weights_file:
            disc.net.copy_from(args.disc_weights_file)

        # continue previous training state, or start new training output file
        loss_file = '{}.{}.training_output'.format(args.out_prefix, fold)
        if args.cont_iter:
            gen.restore('{}_iter_{}.solverstate'.format(gen_prefix, args.cont_iter))
            disc.restore('{}_iter_{}.solverstate'.format(disc_prefix, args.cont_iter))
            loss_df = pd.read_csv(loss_file, sep=' ', header=0, index_col=[0, 1])
            loss_df = loss_df[:args.cont_iter+1]
        else:
            columns = ['iteration', 'test_data']
            loss_df = pd.DataFrame(columns=columns).set_index(columns)
        loss_out = open(loss_file, 'w')

        plot_file = '{}.{}.png'.format(args.out_prefix, fold)
        plot_out = open(plot_file, 'w')

        # begin training GAN
        try:
            train_GAN_model(train_data, test_data, gen, disc, loss_df, loss_out, plot_out, args)

        except:
            gen.snapshot()
            disc.snapshot()
            raise

        finally:
            loss_out.close()
            plot_out.close()


if __name__ == '__main__':
    main(sys.argv[1:])
