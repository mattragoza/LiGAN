from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import sys, os, argparse, time
from collections import OrderedDict
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

import caffe_util
import results
sns.set_style('whitegrid')


def get_metric_from_net(net, metric):
    if metric in net.blobs:
        return np.array(net.blobs[metric].data)
    elif metric == 'grad_norm':
        return get_gradient_norm(net)
    elif metric == 'last_conv':
        last_conv = [n for n in net.blobs if 'conv' in n][-1]
        return np.mean(net.blobs[last_conv].data)
    else:
        raise KeyError(metric)


def get_gradient_norm(net, ord=2):
    grad_norm = 0.0
    for blob_vec in net.params.values():
        for blob in blob_vec:
            grad_norm += np.sum(np.abs(blob.diff)**ord)
    return grad_norm**1/ord


def gradient_normalize(net, ord=2):
    grad_norm = get_gradient_norm(net, ord)
    if grad_norm > 1.0:
        for blob_vec in net.params.values():
            for blob in blob_vec:
                blob.diff[...] /= grad_norm


def normalize(x, ord=2):
    return x / np.linalg.norm(x, ord)


def spectral_power_iterate(W, u, n_iter):

    W = W.reshape(W.shape[0], -1) # treat as matrix

    for i in range(n_iter):
        v = normalize(np.matmul(W.T, u))
        u = normalize(np.matmul(W, v))

    sigma = np.matmul(u.T, np.matmul(W, v))
    return u, v, sigma


def spectral_norm_setup(net):

    params = OrderedDict()
    for layer in net.params:
        W = net.params[layer][0]
        u = np.random.normal(0, 1, W.shape[0])
        u, v, sigma = spectral_power_iterate(W.data, u, 10)
        params[layer] = (u, v, sigma)

    return params


def spectral_norm_forward(net, params):

    for layer in net.params:
        W = net.params[layer][0]
        u, v, sigma = params[layer]
        u, v, sigma = spectral_power_iterate(W.data, u, 1)
        W.data[...] /= sigma
        params[layer] = (u, v, sigma)


def spectral_norm_backward(net, params):

    for layer in net.params:
        W = net.params[layer][0]
        y = net.blobs[layer]
        u, v, sigma = params[layer]
        lambda_ = np.sum(y.diff * y.data) / y.shape[0]
        W.diff[...] -= (lambda_ * np.outer(u, v)).reshape(W.shape)
        W.diff[...] /= sigma


def disc_step(data_net, gen_solver, disc_solver, n_iter, train, args):
    '''
    Train or test the discriminative GAN component for n_iter iterations.
    '''
    gen_net  = gen_solver.net
    disc_net = disc_solver.net
    compute_grad = True

    batch_size = data_net.blobs['data'].shape[0]
    half1 = np.arange(batch_size) < batch_size//2
    half2 = ~half1

    disc_metrics = {}
    for blob_name in disc_net.blobs:
        if blob_name.endswith('loss'):
            disc_metrics[blob_name] = np.full(n_iter, np.nan)
    if compute_grad:
        disc_metrics['grad_norm'] = np.full(n_iter, np.nan)

    if 'info_loss' in disc_net.blobs: # ignore info_loss
        del disc_metrics['info_loss']
        info_loss_weight = disc_net.blobs['info_loss'].diff
        disc_net.blobs['info_loss'].diff[...] = 0.0

    for i in range(n_iter):

        if i%2 == 0:

            # get real receptors and ligands
            data_net.forward()
            rec_real = data_net.blobs['rec'].data
            lig_real = data_net.blobs['lig'].data

            # generate fake ligands conditioned on receptors
            gen_net.blobs['rec'].data[...] = rec_real
            gen_net.blobs['lig'].data[...] = lig_real
            if args.alternate:
                # sample ligand prior (for rvl-l models)
                gen_net.forward(start='rec', end='rec_latent_fc')
                gen_net.blobs['lig_latent_mean'].data[...] = 0.0
                gen_net.blobs['lig_latent_std'].data[...] = 1.0
                gen_net.forward(start='lig_latent_noise', end='lig_gen')
            else:
                gen_net.forward()
            lig_gen = gen_net.blobs['lig_gen'].data

            # create batch of real and generated ligands
            lig_bal = np.concatenate([lig_real[half1,...],
                                      lig_gen[half2,...]])

            # compute likelihood that generated ligands fool discriminator
            disc_net.blobs['rec'].data[...] = rec_real
            disc_net.blobs['lig'].data[...] = lig_bal
            disc_net.blobs['label'].data[...] = half1[:,np.newaxis]
            if 'info_label' in disc_net.blobs:
                info_label = np.zeros_like(disc_net.blobs['info_label'].data)
                disc_net.blobs['info_label'].data[...] = info_label
            elif 'lig_instance_std' in disc_net.blobs:
                disc_net.blobs['lig_instance_std'].data[...] = args.instance_noise

            if args.disc_spectral_norm:
                spectral_norm_forward(disc_net, args.disc_spectral_norm)

            disc_net.forward()

            if train or compute_grad: # update D only

                disc_net.clear_param_diffs()
                disc_net.backward()

                if args.disc_spectral_norm:
                    spectral_norm_backward(disc_net, args.disc_spectral_norm)

                if args.disc_grad_norm:
                    gradient_normalize(disc_net)

                if train:
                    disc_solver.apply_update()

            # record discriminator metrics
            for n in disc_metrics:
                disc_metrics[n][i] = get_metric_from_net(disc_net, n)

        else: # use the other half of the real/generated ligands

            if args.alternate:
                # autoencode real ligands (for rvl-l models)
                gen_net.forward(start='lig_level0_conv0', end='lig_gen')
                lig_gen = gen_net.blobs['lig_gen'].data

            # create complementary batch of real and generated ligands
            lig_bal = np.concatenate([lig_gen[half1,...],
                                      lig_real[half2,...]])

            # compute likelihood that generated ligands fool discriminator
            disc_net.blobs['rec'].data[...] = rec_real
            disc_net.blobs['lig'].data[...] = lig_bal
            disc_net.blobs['label'].data[...] = half2[:,np.newaxis]
            if 'info_label' in disc_net.blobs:
                info_label = np.zeros_like(disc_net.blobs['info_label'].data)
                disc_net.blobs['info_label'].data[...] = info_label
            elif 'lig_instance_std' in disc_net.blobs:
                disc_net.blobs['lig_instance_std'].data[...] = args.instance_noise

            if args.disc_spectral_norm:
                spectral_norm_forward(disc_net, args.disc_spectral_norm)

            disc_net.forward()

            if train or compute_grad: # update D only

                disc_net.clear_param_diffs()
                disc_net.backward()

                if args.disc_spectral_norm:
                    spectral_norm_backward(disc_net, args.disc_spectral_norm)

                if args.disc_grad_norm:
                    gradient_normalize(disc_net)

                if train:
                    disc_solver.apply_update()

            # record discriminator metrics
            for n in disc_metrics:
                disc_metrics[n][i] = get_metric_from_net(disc_net, n)

    if 'info_loss' in disc_net.blobs:
        disc_net.blobs['info_loss'].diff[...] = info_loss_weight

    return {n: m.mean() for n,m in disc_metrics.items()}


def gen_step(data_net, gen_solver, disc_solver, n_iter, train, args):
    '''
    Train or test the generative GAN component for n_iter iterations.
    '''
    gen_net  = gen_solver.net
    disc_net = disc_solver.net

    batch_size = data_net.blobs['data'].shape[0]
    compute_grad = True

    gen_metrics = {}
    for blob_name in gen_net.blobs:
        if blob_name.endswith('loss'):
            gen_metrics[blob_name] = np.full(n_iter, np.nan)
    if compute_grad:
        gen_metrics['grad_norm'] = np.full(n_iter, np.nan)

    disc_metrics = {}
    for blob_name in disc_net.blobs:
        if blob_name.endswith('loss'):
            disc_metrics[blob_name] = np.full(n_iter, np.nan)
    if compute_grad:
        disc_metrics['grad_norm'] = np.full(n_iter, np.nan)

    for i in range(n_iter):

        # get real receptors and ligands
        data_net.forward()
        rec_real = data_net.blobs['rec'].data
        lig_real = data_net.blobs['lig'].data

        # generate fake ligands conditioned on receptors
        gen_net.blobs['rec'].data[...] = rec_real
        gen_net.blobs['lig'].data[...] = lig_real

        if args.gen_spectral_norm:
            spectral_norm_forward(gen_net, args.gen_spectral_norm)

        if args.alternate and i%2:
            # sample ligand prior (for rvl-l models)
            gen_net.forward(start='rec', end='rec_latent_fc')
            gen_net.blobs['lig_latent_mean'].data[...] = 0.0
            gen_net.blobs['lig_latent_std'].data[...] = 1.0
            gen_net.forward(start='lig_latent_noise', end='lig_gen')
        else:
            gen_net.forward()
        lig_gen = gen_net.blobs['lig_gen'].data

        # compute likelihood that generated ligands fool discriminator
        disc_net.blobs['rec'].data[...] = rec_real
        disc_net.blobs['lig'].data[...] = lig_gen
        disc_net.blobs['label'].data[...] = 1.0
        if 'info_label' in disc_net.blobs:
            info_label = np.concatenate([gen_net.blobs['lig_latent_mean'].data,
                                         gen_net.blobs['lig_latent_log_std'].data], axis=1)
            disc_net.blobs['info_label'].data[...] = info_label
        elif 'lig_instance_std' in disc_net.blobs:
            disc_net.blobs['lig_instance_std'].data[...] = args.instance_noise
        disc_net.forward()

        if train or compute_grad: # backpropagate through D and G and apply update

            if 'info_loss' in disc_net.blobs: # TODO normalize gradients
                # apply info loss update to discriminator
                disc_net.clear_param_diffs()
                disc_net.backward(start='info_loss') # exclude GAN loss (should be after info_loss)
                if train:
                    disc_solver.apply_update()

            disc_net.clear_param_diffs()
            disc_net.backward()
            gen_net.blobs['lig_gen'].diff[...] = disc_net.blobs['lig'].diff
            gen_net.clear_param_diffs()

            if args.alternate and i%2: # skip gen_loss and lig encoder
                gen_net.backward(start='lig_gen', end='lig_latent_noise')
                gen_net.backward(start='rec_latent_fc', end='rec')
            else:
                gen_net.backward()

            if args.gen_spectral_norm:
                spectral_norm_backward(gen_net, args.gen_spectral_norm)

            if args.gen_grad_norm:
                gradient_normalize(gen_net)

            if train:
                gen_solver.apply_update()

        # record generator metrics
        for n in gen_metrics:
            gen_metrics[n][i] = get_metric_from_net(gen_net, n)

        # record discriminator metrics (generative-adversarial)
        for n in disc_metrics:
            disc_metrics[n][i] = get_metric_from_net(disc_net, n)

    return {n: np.nanmean(m) for n,m in gen_metrics.items()}, \
           {n: np.nanmean(m) for n,m in disc_metrics.items()}


def train_GAN_model(train_data_net, test_data_nets, gen_solver, disc_solver,
                    loss_df, loss_out, plot_out, args):
    '''
    Train a GAN using the provided train_data_net, gen_solver, and disc_solver.
    Return the loss output from periodically testing on each of test_data_nets
    as a data frame, and write it to loss_out as training to proceeds.
    '''
    train_disc_loss = np.nan
    train_gen_adv_loss = np.nan
    train_disc = True
    train_gen = True
    disc_iter = 0
    gen_iter = 0
    times = []

    if args.disc_spectral_norm:
        args.disc_spectral_norm = spectral_norm_setup(disc_solver.net)

    if args.gen_spectral_norm:
        args.gen_spectral_norm = spectral_norm_setup(gen_solver.net)

    for i in range(args.cont_iter, args.max_iter+1):
        start = time.time()

        if i%args.snapshot == 0:
            disc_solver.snapshot()
            gen_solver.snapshot()

        first_cont_iter = (args.cont_iter and i == args.cont_iter)
        if i%args.test_interval == 0 and not first_cont_iter: # test

            for test_data, test_data_net in test_data_nets.items():

                disc_metrics = \
                    disc_step(test_data_net, gen_solver, disc_solver, args.test_iter, False, args)

                gen_metrics, gen_adv_metrics = \
                    gen_step(test_data_net, gen_solver, disc_solver, args.test_iter, False, args)

                for n in disc_metrics:
                    loss_df.loc[(i, test_data), 'disc_' + n] = disc_metrics[n]

                for n in gen_metrics:
                    loss_df.loc[(i, test_data), 'gen_' + n] = gen_metrics[n]

                for n in gen_adv_metrics:
                    loss_df.loc[(i, test_data), 'gen_adv_' + n] = gen_adv_metrics[n]

            loss_df.loc[(i, 'train'), 'disc_iter'] = disc_iter
            loss_df.loc[(i, 'train'), 'gen_iter'] = gen_iter

            loss_out.seek(0)
            loss_df.to_csv(loss_out, sep=' ')
            loss_out.flush()

            plot_out.seek(0)
            results.plot_lines(plot_out, loss_df,
                               x='iteration',
                               y=loss_df.columns,
                               hue='test_data')
            plot_out.flush()

            time_elapsed = np.sum(times)
            time_mean = time_elapsed // len(times)
            iters_left = args.max_iter - i
            time_left = time_mean*iters_left

            print('Iteration {} / {}'.format(i, args.max_iter))
            print('  {} elapsed'.format(time_elapsed))
            print('  {} per iter'.format(time_mean))
            print('  {} left'.format(time_left))
            for test_data in test_data_nets:
                for n in loss_df:
                    print('  {} {} = {}'.format(test_data, n, loss_df.loc[(i, test_data), n]))
            sys.stdout.flush()

        if i == args.max_iter:
            break

        # dynamic G/D balancing
        if args.balance:

            # how much better is D than G?
            train_loss_ratio = train_gen_adv_loss / train_disc_loss

            if train_disc and train_loss_ratio > 10:
                train_disc = False

            if not train_disc and train_loss_ratio < 2:
                train_disc = True

            if train_gen and train_loss_ratio < 1:
                train_gen = False

            if not train_gen and train_loss_ratio > 2:
                train_gen = True

        # train
        disc_metrics = \
            disc_step(train_data_net, gen_solver, disc_solver, args.disc_train_iter, train_disc, args)

        gen_metrics, gen_adv_metrics = \
            gen_step(train_data_net, gen_solver, disc_solver, args.gen_train_iter, train_gen, args)

        train_disc_loss = disc_metrics['log_loss']
        train_gen_adv_loss = gen_adv_metrics['log_loss']

        disc_solver.increment_iter()
        gen_solver.increment_iter()
        disc_iter += train_disc * args.disc_train_iter
        gen_iter += train_gen * args.gen_train_iter

        # check common failure cases
        lig_gen = gen_solver.net.blobs['lig_gen']
        assert not np.all(lig_gen.data == 0.0)
        assert train_disc_loss > 0.0

        times.append(dt.timedelta(seconds=time.time() - start))


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
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_prefix', default='')
    parser.add_argument('-d', '--data_model_file', required=True)
    parser.add_argument('-g', '--gen_model_file', required=True)
    parser.add_argument('-a', '--disc_model_file', required=True)
    parser.add_argument('-s', '--solver_file', required=True)
    parser.add_argument('-p', '--data_prefix', required=True)
    parser.add_argument('-n', '--fold_nums', default='0,1,2,all')
    parser.add_argument('-r', '--data_root', required=True)
    parser.add_argument('--max_iter', default=10000, type=int)
    parser.add_argument('--snapshot', default=1000, type=int)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--test_iter', default=10, type=int)
    parser.add_argument('--test_interval', default=10, type=int)
    parser.add_argument('--gen_train_iter', default=20, type=int)
    parser.add_argument('--disc_train_iter', default=20, type=int)
    parser.add_argument('--cont_iter', default=0, type=int)
    parser.add_argument('--alternate', default=False, action='store_true')
    parser.add_argument('--balance', default=False, action='store_true')
    parser.add_argument('--instance_noise', type=float, default=0.0)
    parser.add_argument('--gen_grad_norm', default=False, action='store_true')
    parser.add_argument('--disc_grad_norm', default=False, action='store_true')
    parser.add_argument('--gen_spectral_norm', default=False, action='store_true')
    parser.add_argument('--disc_spectral_norm', default=False, action='store_true')
    parser.add_argument('--gen_weights_file')
    parser.add_argument('--disc_weights_file')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    # read solver and model param files and set general params
    data_net_param = caffe_util.NetParameter.from_prototxt(args.data_model_file)

    gen_net_param = caffe_util.NetParameter.from_prototxt(args.gen_model_file)
    gen_net_param.force_backward = True

    disc_net_param = caffe_util.NetParameter.from_prototxt(args.disc_model_file)
    disc_net_param.force_backward = True

    solver_param = caffe_util.SolverParameter.from_prototxt(args.solver_file)
    solver_param.max_iter = args.max_iter
    solver_param.test_interval = args.max_iter+1 # will test manually
    solver_param.random_seed = args.random_seed

    for fold, train_file, test_file in get_train_and_test_files(args.data_prefix, args.fold_nums):

        # create data net for training (on train set)
        data_net_param.set_molgrid_data_source(train_file, args.data_root, caffe.TRAIN)
        train_data_net = caffe_util.Net.from_param(data_net_param, phase=caffe.TRAIN)

        # create data nets for testing on both test and train sets
        test_data_nets = dict()
        data_net_param.set_molgrid_data_source(test_file, args.data_root, caffe.TEST)
        test_data_nets['test'] = caffe_util.Net.from_param(data_net_param, phase=caffe.TEST)
        data_net_param.set_molgrid_data_source(train_file, args.data_root, caffe.TEST)
        test_data_nets['train'] = caffe_util.Net.from_param(data_net_param, phase=caffe.TEST)

        # create solver for generative model
        gen_prefix = '{}.{}_gen'.format(args.out_prefix, fold)
        gen_solver = caffe_util.Solver.from_param(solver_param, net_param=gen_net_param,
                                                  snapshot_prefix=gen_prefix)
        if args.gen_weights_file:
            gen_solver.net.copy_from(args.gen_weights_file)

        elif any(n == 'lig_gauss_conv' for n in gen_solver.net.blobs):
            gen_solver.net.copy_from('lig_gauss_conv.caffemodel')

        # create solver for discriminative model
        disc_prefix = '{}.{}_disc'.format(args.out_prefix, fold)
        disc_solver = caffe_util.Solver.from_param(solver_param, net_param=disc_net_param,
                                                   snapshot_prefix=disc_prefix)
        if args.disc_weights_file:
            disc_solver.net.copy_from(args.disc_weights_file)

        # continue previous training state, or start new training output file
        loss_file = '{}.{}.training_output'.format(args.out_prefix, fold)
        if args.cont_iter:
            gen_state_file = '{}_iter_{}.solverstate'.format(gen_prefix, args.cont_iter)
            disc_state_file = '{}_iter_{}.solverstate'.format(disc_prefix, args.cont_iter)
            gen_solver.restore(gen_state_file)
            disc_solver.restore(disc_state_file)
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
            train_GAN_model(train_data_net, test_data_nets, gen_solver, disc_solver,
                            loss_df, loss_out, plot_out, args)
        finally:
            disc_solver.snapshot()
            gen_solver.snapshot()
            loss_out.close()
            plot_out.close()


if __name__ == '__main__':
    main(sys.argv[1:])
