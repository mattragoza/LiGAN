#!/usr/bin/env python3
from __future__ import print_function, division

import matplotlib
matplotlib.use('Agg')
import sys, os, argparse, time
import collections
import itertools
import datetime as dt
import numpy as np
import pandas as pd
import interrupt
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('whitegrid')

import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

import generate
from caffe_util import NetParameter, SolverParameter, Net, Solver
from results import plot_lines

try:
    import wandb
except:
    print("wandb not available")

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
            rec = data.blobs['rec']
            lig = data.blobs['lig']

            disc.net.blobs['rec'].copyfrom(rec)
            disc.net.blobs['lig'].copyfrom(lig)
            disc.net.blobs['label'].set_data(1.0)

        else: # generate fake ligands

            # reuse rec and lig from last real forward pass
            gen.net.blobs['rec'].copyfrom(rec)
            gen.net.blobs['lig'].copyfrom(lig)

            if args.gen_spectral_norm:
                spectral_norm_forward(gen.net, args.gen_spectral_norm)

            if not prior: # posterior

                gen.net.forward()

            else: # prior

                gen.net.blobs[latent_mean].set_data(0.0)
                gen.net.blobs[latent_std].set_data(1.0)
                gen.net.forward(start=latent_noise) # assumes cond branch is after latent space

            lig_gen = gen.net.blobs['lig_gen']

            disc.net.blobs['rec'].copyfrom(rec)
            disc.net.blobs['lig'].copyfrom(lig_gen)
            disc.net.blobs['label'].set_data(0.0)

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

            if compute_metrics and False:
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
            rec = data.blobs['rec']
            lig = data.blobs['lig']

            gen.net.blobs['rec'].copyfrom(rec)
            gen.net.blobs['lig'].copyfrom(lig)

            if 'cond_rec' in gen.net.blobs:
                gen.net.blobs['cond_rec'].copyfrom(rec)

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

            gen.net.blobs[latent_mean].set_data(0.0)
            gen.net.blobs[latent_std].set_data(1.0)
            gen.net.forward(start=latent_noise)

        lig_gen = gen.net.blobs['lig_gen']

        # cross_entropy_loss(y_t, y_p) = -[y_t*log(y_p) + (1 - y_t)*log(1 - y_p)]
        # for original minmax GAN loss, set lig_gen label = 0.0 and ascend gradient
        # for non-saturating GAN loss, set lig_gen label = 1.0 and descend gradient

        disc.net.blobs['rec'].copyfrom(rec)
        disc.net.blobs['lig'].copyfrom(lig_gen)
        disc.net.blobs['label'].set_data(1.0)

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

            gen.net.blobs['lig_gen'].copyfrom(disc.net.blobs['lig'],True)
            gen.net.clear_param_diffs()

            # set non-GAN loss weights
            for l, w in loss_weights.items():
                gen.net.blobs[l].set_diff(0) if prior else w * args.loss_weight

            if prior: # only backprop gradient to noise source (what about cond branch??)
                gen.net.backward(end=latent_noise)
                gen.net.blobs[latent_mean].set_diff(0.0)
                gen.net.blobs[latent_std].set_diff(0.0)
                gen.net.backward(start=latent_std)

                #why check this? won't it have leftover values?
                #lig_grad_norm = np.linalg.norm(gen.net.blobs['lig'].diff)
                #assert np.isclose(lig_grad_norm, 0), lig_grad_norm
    
            else:
                gen.net.backward()

            if args.gen_spectral_norm:
                spectral_norm_backward(gen.net, args.gen_spectral_norm)

            if args.gen_grad_norm:
                gradient_normalize(gen.net)
            if compute_metrics and False: # dkoes - these are done on the CPU and are SUPER expensive
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
    dtime = 0
    gtime = 0
    dcnt = 0
    gcnt = 0
    for i in range(args.cont_iter, args.max_iter+1):

        if i % args.snapshot == 0 and i != 0:
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
            t_per_iter = t_total/(i+1 - args.cont_iter)
            t_left = t_per_iter * (args.max_iter - i)
            t_total = dt.timedelta(seconds=t_total)
            if i > args.cont_iter:
                t_per_iter = dt.timedelta(seconds=t_per_iter)
                t_left = dt.timedelta(seconds=t_left)

            print('Iteration {} / {}'.format(i, args.max_iter))
            print('  {} elapsed ({:.1f}% training, {:.1f}% testing)'
                  .format(t_total, pct_train, pct_test))
            print("Disc cnt/time: %d %f, Gen cnt/time: %d %f"%(dcnt,dtime,gcnt,gtime))
            tolog = {'discnt':dcnt,'disctime':dtime,'gencnt':gcnt,'gentime':gtime,'iteration':i}
            dcnt = gcnt = dtime = gtime = 0
            print('  {} left (~{} / iteration)'.format(t_left, t_per_iter))            
            for d in test_data:
                for m in sorted(loss_df.columns):
                    print('  {} {} = {}'.format(d, m, loss_df.loc[(i, d), m]))
                    tolog['{} {}'.format(d,m)] = loss_df.loc[(i,d),m]
            if args.wandb:
                wandb.log(tolog)

            write_and_plot_metrics(loss_df, loss_file, plot_file)
            sys.stdout.flush()
            
        if i == args.max_iter: # return after final test evaluation
            return

        t_start = time.time()

        # disc then gen; don't have to do backward if doing balanced training,
        # but still need forward for loss computation
        dstart = time.time()
        disc_metrics = disc_step(train_data, gen, disc, args.disc_train_iter, args,            
              train=train_disc, compute_metrics=False)
        dtime += time.time()-dstart
        if train_disc: dcnt += 1

        gstart = time.time()
        gen_metrics = gen_step(train_data, gen, disc, args.gen_train_iter, args,
             train=train_gen, compute_metrics=False)
        gtime += time.time()-gstart
        if train_gen: gcnt += 1

        if 'disc_wass_loss' in disc_metrics:
            train_gen_loss = gen_metrics['gen_adv_wass_loss']
            train_disc_loss = disc_metrics['disc_wass_loss']
            train_loss_balance = train_gen_loss - train_disc_loss
        else:
            train_gen_loss = gen_metrics['gen_adv_log_loss']
            train_disc_loss = disc_metrics['disc_log_loss']
            train_loss_balance = train_gen_loss / train_disc_loss
            
        assert np.isfinite(train_gen_loss)
        assert np.isfinite(train_disc_loss)

        if i+1 == args.max_iter:
            train_disc = False
            train_gen = False            
        elif args.balance: # dynamically balance G/D training

            # how much better is D than G?
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
    parser.add_argument('-s', '--solver_file', required=False, help='prototxt file for solver hyperparameters, can be overriden by command line options')
    parser.add_argument('-p', '--data_prefix', required=True, help='prefix for data train/test fold files')
    parser.add_argument('-n', '--fold_nums', default='0,1,2,all', help='comma-separated fold numbers to run (default 0,1,2,all)')
    parser.add_argument('-r', '--data_root', required=True, help='root directory of data files (prepended to paths in train/test fold files)')
    parser.add_argument('--random_seed', default=0, type=int, help='random seed for Caffe initialization and training (default 0)')
    parser.add_argument('--max_iter', default=100000, type=int, help='total number of train iterations (default 10000)')
    parser.add_argument('--snapshot', default=10000, type=int, help='save .caffemodel weights and solver state every # train iters (default 1000)')
    parser.add_argument('--test_interval', default=10, type=int, help='evaluate test data every # train iters (default 10)')
    parser.add_argument('--test_iter', default=10, type=int, help='number of iterations of each test data evaluation (default 10)')
    parser.add_argument('--gen_train_iter', default=2, type=int, help='number of sub-iterations to train gen model each train iter (default 2)')
    parser.add_argument('--disc_train_iter', default=2, type=int, help='number of sub-iterations to train disc model each train iter (default 2)')
    parser.add_argument('--cont_iter', default=0, type=int, help='continue training from iteration #')
    parser.add_argument('--alternate', default=0, type=int, help='alternate between encoding and sampling latent prior')
    parser.add_argument('--balance', default=0, type=int, help='dynamically train gen/disc each iter by balancing GAN loss')
    parser.add_argument('--instance_noise', type=float, default=0.0, help='standard deviation of disc instance noise (default 0.0)')
    parser.add_argument('--gen_grad_norm', default=False, action='store_true', help='gen gradient normalization')
    parser.add_argument('--disc_grad_norm', default=False, action='store_true', help='disc gradient normalization')
    parser.add_argument('--gen_spectral_norm', default=False, action='store_true', help='gen spectral normalization')
    parser.add_argument('--disc_spectral_norm', default=False, action='store_true', help='disc spectral normalization')
    parser.add_argument('--gen_weights_file', help='.caffemodel file to initialize gen weights')
    parser.add_argument('--disc_weights_file', help='.caffemodel file to initialize disc weights')
    parser.add_argument('--loss_weight', default=1.0, type=float, help='initial value for non-GAN generator loss weight')
    parser.add_argument('--loss_weight_decay', default=0.0, type=float, help='decay rate for non-GAN generator loss weight')
    parser.add_argument('--batch_size',default=5, type=int, help='value to substitute for BATCH_SIZE in models')
    parser.add_argument('--wandb',action='store_true',help='enable weights and biases')
    #solver arguments
    parser.add_argument('--clip_gradients',type=float, help='amount to clip gradients by in solver')
    parser.add_argument('--solver',type=str, help='solver to use')
    parser.add_argument('--momentum',type=float, help='momentum')
    parser.add_argument('--momentum2',type=float, help='momentum2 for adam')
    parser.add_argument('--lr_policy',type=str, help='lr policy')
    parser.add_argument('--base_lr',type=float, help='base learning rate')
    parser.add_argument('--weight_decay',type=float, help='weight decay (L2 regularization)')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    if args.wandb:
        wandb.init(project='gentrain',config=args)
        if args.out_prefix == '':
            try:
                os.mkdir('wandb_output')
            except FileExistsError:
                pass
            args.out_prefix = 'wandb_output/'+wandb.run.id
            sys.stderr.write("Setting output prefix to %s\n"%args.out_prefix)

    config = open('%s.config'%args.out_prefix,'wt')
    config.write('\n'.join(map(lambda kv: '%s : %s'%kv, vars(args).items())))
    config.close()
    
    # read solver and model param files and set general params
    # batch size is set through string replacement because
    data_str = open(args.data_model_file).read()
    data_str = data_str.replace('BATCH_SIZE',str(args.batch_size))
    data_param = NetParameter.from_prototxt_str(data_str)

    gen_str = open(args.gen_model_file).read()
    gen_str = gen_str.replace('BATCH_SIZE',str(args.batch_size))
    gen_param = NetParameter.from_prototxt_str(gen_str)
    
    disc_str = open(args.disc_model_file).read()
    disc_str = disc_str.replace('BATCH_SIZE',str(args.batch_size))    
    disc_param = NetParameter.from_prototxt_str(disc_str)

    gen_param.force_backward = True
    disc_param.force_backward = True

    if args.solver_file:
        solver_param = SolverParameter.from_prototxt(args.solver_file)
    else:
        solver_param = SolverParameter()
    solver_param.max_iter = args.max_iter
    solver_param.test_interval = args.max_iter + 1
    solver_param.random_seed = args.random_seed
    caffe.set_random_seed(args.random_seed) #this should be redundant
    
    #check for cmdline overrides
    if args.solver is not None:
        solver_param.type = args.solver
    if args.clip_gradients is not None:        
        solver_param.clip_gradients = args.clip_gradients
    if args.momentum is not None:
        solver_param.momentum = args.momentum
    if args.momentum2 is not None:
        solver_param.momentum2 = args.momentum2
    if args.lr_policy is not None:
        solver_param.lr_policy = args.lr_policy
    if args.base_lr is not None:
        solver_param.base_lr = args.base_lr
    if args.weight_decay is not None:
        solver_param.weight_decay = args.weight_decay
        
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
        gen_prefix = '{}_{}_gen'.format(args.out_prefix, fold)
        gen = Solver.from_param(solver_param, net_param=gen_param, snapshot_prefix=gen_prefix)
        if args.gen_weights_file:
            gen.net.copy_from(args.gen_weights_file)
        if 'lig_gauss_conv' in gen.net.blobs:
            gen.net.copy_from('lig_gauss_conv.caffemodel')

        # create solver for training discriminator net
        disc_prefix = '{}_{}_disc'.format(args.out_prefix, fold)
        disc = Solver.from_param(solver_param, net_param=disc_param, snapshot_prefix=disc_prefix)
        if args.disc_weights_file:
            disc.net.copy_from(args.disc_weights_file)

        # continue previous training state, or start new training output file
        loss_file = '{}_{}.training_output'.format(args.out_prefix, fold)
        print('loss file',loss_file)
        if args.cont_iter:
            gen.restore('{}_iter_{}.solverstate'.format(gen_prefix, args.cont_iter))
            disc.restore('{}_iter_{}.solverstate'.format(disc_prefix, args.cont_iter))
            loss_df = pd.read_csv(loss_file, sep=' ', header=0, index_col=[0, 1])
            loss_df = loss_df[:args.cont_iter+1]
        else:
            columns = ['iteration', 'phase']
            loss_df = pd.DataFrame(columns=columns).set_index(columns)

        plot_file = '{}_{}.png'.format(args.out_prefix, fold)

        # begin training GAN
        try:
            train_GAN_model(train_data, test_data, gen, disc, loss_df, loss_file, plot_file, args)
        except:
            raise
            gen.snapshot()
            disc.snapshot()
            raise


if __name__ == '__main__':
    interrupt.listen()
    main(sys.argv[1:])

