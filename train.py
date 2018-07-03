from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import sys, os, argparse, time
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

import caffe_util


def training_plot(plot_file, loss_df, binsize=1):

    loss_df = loss_df.groupby(np.arange(len(loss_df))//binsize).mean()
    loss_df.index = binsize*(loss_df.index + 1)

    fig, ax = plt.subplots()
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    for column in loss_df:
        ax.plot(loss_df.index, loss_df[column], label=column, linewidth=2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(plot_file)


def disc_step(data_net, gen_solver, disc_solver, n_iter, train, alternate):
    '''
    Train or test the discriminative GAN component for n_iter iterations.
    '''
    gen_net  = gen_solver.net
    disc_net = disc_solver.net

    batch_size = data_net.blobs['lig'].shape[0]
    half1 = np.arange(batch_size) < batch_size//2
    half2 = ~half1

    disc_loss_dict = {n: np.full(n_iter, np.nan) for n in disc_net.blobs if n.endswith('loss')}

    if 'info_loss' in disc_net.blobs: # ignore info_loss
        del disc_loss_dict['info_loss']
        info_loss_weight = disc_net.blobs['info_loss'].diff
        disc_net.blobs['info_loss'].diff[...] = 0.0

    for i in range(n_iter):

        if i%2 == 0: # first half real, second half gen

            data_out = data_net.forward()
            rec_real = data_out['rec']
            lig_real = data_out['lig']

            if alternate: # sample unit ligand latent space
                gen_net.forward(start='rec', end='rec_latent_fc', rec=rec_real, lig=lig_real)
                gen_net.blobs['lig_latent_mean'].data[...] = 0.0
                gen_net.blobs['lig_latent_std'].data[...] = 1.0
                gen_out = gen_net.forward(start='lig_latent_noise', end='lig_gen')
                lig_gen = gen_out['lig_gen']
            else:
                gen_out = gen_net.forward(rec=rec_real, lig=lig_real)
                lig_gen = gen_out['lig_gen']

            lig_bal = np.concatenate([lig_real[half1,...], lig_gen[half2,...]])
            if 'info_label' in disc_net.blobs:
                info_label = np.zeros_like(disc_net.blobs['info_label'].data)
                disc_out = disc_net.forward(rec=rec_real, lig=lig_bal, label=half1, info_label=info_label)
            else:
                disc_out = disc_net.forward(rec=rec_real, lig=lig_bal, label=half1)

            for n in disc_loss_dict:
                disc_loss_dict[n][i] = disc_out[n]

            if train:
                disc_net.clear_param_diffs()
                disc_net.backward()
                disc_solver.apply_update()

        else: # first half gen, second half real

            if alternate: # autoencode real ligand
                gen_out = gen_net.forward(start='lig_level0_conv0', end='lig_gen')
                lig_gen = gen_out['lig_gen']     

            lig_bal = np.concatenate([lig_gen[half1,...], lig_real[half2,...]])
            if 'info_label' in disc_net.blobs:
                info_label = np.zeros_like(disc_net.blobs['info_label'].data)
                disc_out = disc_net.forward(rec=rec_real, lig=lig_bal, label=half2, info_label=info_label)
            else:
                disc_out = disc_net.forward(rec=rec_real, lig=lig_bal, label=half2)

            for n in disc_loss_dict:
                disc_loss_dict[n][i] = disc_out[n]

            if train:
                disc_net.clear_param_diffs()
                disc_net.backward()
                disc_solver.apply_update()

    if 'info_loss' in disc_net.blobs:
        disc_net.blobs['info_loss'].diff[...] = info_loss_weight

    return {n: l.mean() for n,l in disc_loss_dict.items()}


def gen_step(data_net, gen_solver, disc_solver, n_iter, train, alternate):
    '''
    Train or test the generative GAN component for n_iter iterations.
    '''
    gen_net  = gen_solver.net
    disc_net = disc_solver.net

    batch_size = data_net.blobs['lig'].shape[0]

    gen_loss_dict = {n: np.full(n_iter, np.nan) for n in gen_net.blobs if n.endswith('loss')}
    disc_loss_dict = {n: np.full(n_iter, np.nan) for n in disc_net.blobs if n.endswith('loss')}

    for i in range(n_iter):

        # get real receptors and ligands
        data_out = data_net.forward()
        rec_real = data_out['rec']
        lig_real = data_out['lig']

        # generate fake ligands
        if alternate and i%2:
            # sample unit ligand latent space instead of conditioning on real ligand, no gen_loss
            gen_net.forward(start='rec', end='rec_latent_fc', rec=rec_real, lig=lig_real)
            gen_net.blobs['lig_latent_mean'].data[...] = 0.0
            gen_net.blobs['lig_latent_std'].data[...] = 1.0
            gen_out = gen_net.forward(start='lig_latent_noise', end='lig_gen')
            lig_gen = gen_out['lig_gen']
        else:
            gen_out = gen_net.forward(rec=rec_real, lig=lig_real)
            lig_gen = gen_out['lig_gen']

            # record generative loss, if applicable
            for n in gen_loss_dict:
                gen_loss_dict[n][i] = gen_out[n]

        # score fake ligands labeled as real (and include latent stats for info_loss)
        if 'info_label' in disc_net.blobs:
            info_label = np.concatenate([gen_net.blobs['lig_latent_mean'].data,
                                         gen_net.blobs['lig_latent_log_std'].data], axis=1)
            disc_out = disc_net.forward(rec=rec_real, lig=lig_gen, label=np.ones(batch_size), info_label=info_label)
        else:
            disc_out = disc_net.forward(rec=rec_real, lig=lig_gen, label=np.ones(batch_size))

        # record discriminative loss
        for n in disc_loss_dict:
            disc_loss_dict[n][i] = disc_out[n]

        if train: # backpropagate through D and G and apply solver update(s)

            if 'info_loss' in disc_net.blobs:
                # apply info loss update to discriminator
                disc_net.clear_param_diffs()
                disc_net.backward(start='info_loss') # exclude GAN loss (should be after info_loss)
                disc_solver.apply_update()

            disc_net.clear_param_diffs()
            disc_net.backward() # now includes GAN loss and other losses
            gen_net.blobs['lig_gen'].diff[...] = disc_net.blobs['lig'].diff
            gen_net.clear_param_diffs()

            if alternate and i%2: # skip gen_loss and lig encoder
                gen_net.backward(start='lig_gen', end='lig_latent_noise')
                gen_net.backward(start='rec_latent_fc', end='rec')
            else:
                gen_net.backward()

            gen_solver.apply_update()

    return {n: np.nanmean(l) for n,l in gen_loss_dict.items()}, \
           {n: np.nanmean(l) for n,l in disc_loss_dict.items()}


def train_GAN_model(train_data_net, test_data_nets, gen_solver, disc_solver, loss_out, args):
    '''
    Train a GAN using the provided train_data_net, gen_solver, and disc_solver.
    Return the loss output from periodically testing on each of test_data_nets
    as a data frame, and write it to loss_out as training to proceeds.
    '''
    loss_df = pd.DataFrame(index=range(args.cont_iter, args.max_iter+1, args.test_interval))
    loss_df.index.name = 'iteration'

    times = []
    for i in range(args.cont_iter, args.max_iter+1):
        start = time.time()

        if i%args.snapshot == 0:
            disc_solver.snapshot()
            gen_solver.snapshot()

        first_cont_iter = (args.cont_iter and i == args.cont_iter)
        if i%args.test_interval == 0 and not first_cont_iter: # test

            for data_name, test_data_net in test_data_nets.items():

                disc_loss_dict = \
                    disc_step(test_data_net, gen_solver, disc_solver, args.test_iter, False, args.alternate)

                gen_loss_dict, gen_adv_loss_dict = \
                    gen_step(test_data_net, gen_solver, disc_solver, args.test_iter, False, args.alternate)

                if 'info_loss' in gen_adv_loss_dict:
                    loss = gen_adv_loss_dict.pop('info_loss')
                    loss_df.loc[i, '{}_info_loss'.format(data_name)] = loss

                for loss_name, loss in disc_loss_dict.items():
                    loss_df.loc[i, '{}_disc_{}'.format(data_name, loss_name)] = loss

                for loss_name, loss in gen_loss_dict.items():
                    loss_df.loc[i, '{}_gen_{}'.format(data_name, loss_name)] = loss

                for loss_name, loss in gen_adv_loss_dict.items():
                    loss_df.loc[i, '{}_gen_adv_{}'.format(data_name, loss_name)] = loss

            first_test = (i == 0)
            loss_df.loc[i:i+1].to_csv(loss_out, header=first_test, sep=' ')
            loss_out.flush()

            time_elapsed = np.sum(times)
            time_mean = time_elapsed // len(times)
            iters_left = args.max_iter - i
            time_left = time_mean*iters_left

            print('Iteration {} / {}'.format(i, args.max_iter))
            print('  {} elapsed'.format(time_elapsed))
            print('  {} per iter'.format(time_mean))
            print('  {} left'.format(time_left))
            for loss_name in loss_df:
                loss = loss_df.loc[i, loss_name]
                print('  {} = {}'.format(loss_name, loss))
            sys.stdout.flush()

        if i == args.max_iter:
            break

        # train
        disc_step(train_data_net, gen_solver, disc_solver, args.disc_train_iter, True, args.alternate)
        gen_step(train_data_net, gen_solver, disc_solver, args.gen_train_iter, True, args.alternate)
 
        disc_solver.increment_iter()
        gen_solver.increment_iter()
        times.append(dt.timedelta(seconds=time.time() - start))

    return loss_df


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
    parser.add_argument('--max_iter', default=20000, type=int)
    parser.add_argument('--snapshot', default=1000, type=int)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--test_iter', default=10, type=int)
    parser.add_argument('--test_interval', default=100, type=int)
    parser.add_argument('--gen_train_iter', default=1, type=int)
    parser.add_argument('--disc_train_iter', default=2, type=int)
    parser.add_argument('--cont_iter', default=0, type=int)
    parser.add_argument('--alternate', default=False, action='store_true')
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
            loss_out = open(loss_file, 'a')
        else:
            loss_out = open(loss_file, 'w')

        # begin training GAN
        try:
            loss_df = train_GAN_model(train_data_net, test_data_nets, gen_solver, disc_solver,
                                      loss_out, args)
        except:
            disc_solver.snapshot()
            gen_solver.snapshot()
            raise

        # plot training output against iteration
        plot_file = '{}.{}.pdf'.format(args.out_prefix, fold)
        training_plot(plot_file, loss_df)


if __name__ == '__main__':
    main(sys.argv[1:])
