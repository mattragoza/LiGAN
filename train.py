from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import sys
import os
import argparse
import time
from datetime import timedelta
from operator import itemgetter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

import caffe_util


def training_plot(plot_file, loss_df, binsize=100):

    colors = ['r','g','b']
    loss_df = loss_df.groupby(np.arange(len(loss_df))//binsize).mean()
    loss_df.index = binsize*(loss_df.index + 1)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('cross entropy loss')
    for column in ['disc_loss', 'gen_adv_loss']:
        ax1.plot(loss_df.index, loss_df[column], label=column,
                 color=colors.pop(0), linewidth=1)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('L2 loss')
    for column in ['gen_L2_loss']:
        ax2.plot(loss_df.index, loss_df[column], label=column,
                 color=colors.pop(0), linewidth=1)
    ax2.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(plot_file)


def disc_step(data_net, gen_solver, disc_solver, n_iter, train):

    gen_net  = gen_solver.net
    disc_net = disc_solver.net

    batch_size = data_net.blobs['lig'].shape[0]
    half1 = np.arange(batch_size) < batch_size//2
    half2 = ~half1

    disc_loss = np.float64(0)
    n_forward = 0
    for it in range(n_iter):

        if it%2 == 0:
            data_net.forward()
            rec_real = data_net.blobs['rec'].data
            lig_real = data_net.blobs['lig'].data

            gen_net.forward(rec=rec_real, lig=lig_real)
            lig_gen = gen_net.blobs['lig_gen'].data

            lig_bal = np.concatenate([lig_real[half1,...], lig_gen[half2,...]])
            disc_net.forward(rec=rec_real, lig=lig_bal, label=half1)
            disc_loss += disc_net.blobs['loss'].data

            if train:
                disc_net.clear_param_diffs()
                disc_net.backward()
                disc_solver.apply_update()
        else:
            lig_bal = np.concatenate([lig_gen[half1,...], lig_real[half2,...]])
            disc_net.forward(rec=rec_real, lig=lig_bal, label=half2)
            disc_loss += disc_net.blobs['loss'].data
            n_forward += 1

            if train:
                disc_net.clear_param_diffs()
                disc_net.backward()
                disc_solver.apply_update()

    return dict(disc_loss=disc_loss/n_iter)


def gen_step(data_net, gen_solver, disc_solver, n_iter, train, lambda_L2, lambda_adv):

    gen_net  = gen_solver.net
    disc_net = disc_solver.net

    batch_size = data_net.blobs['lig'].shape[0]

    gen_L2_loss = np.float64(0)
    gen_adv_loss = np.float64(0)
    for i in range(n_iter):

        data_net.forward()
        rec_real = data_net.blobs['rec'].data
        lig_real = data_net.blobs['lig'].data

        gen_net.forward(rec=rec_real, lig=lig_real)
        gen_L2_loss += gen_net.blobs['loss'].data
        lig_gen = gen_net.blobs['lig_gen'].data

        disc_net.forward(rec=rec_real, lig=lig_gen, label=np.ones(batch_size))
        gen_adv_loss += disc_net.blobs['loss'].data

        if train:
            disc_net.clear_param_diffs()
            disc_net.backward()

            gen_net.blobs['loss'].diff[...] = lambda_L2
            gen_net.blobs['lig_gen'].diff[...] = lambda_adv*disc_net.blobs['lig'].diff
            gen_net.clear_param_diffs()
            gen_net.backward()
            gen_solver.apply_update()

    return dict(gen_L2_loss=gen_L2_loss/n_iter,
                gen_adv_loss=gen_adv_loss/n_iter)


def train_gan_model(train_data_net, test_data_nets, gen_solver, disc_solver, solver_param,
                    gen_iter_mult, disc_iter_mult, lambda_L2, lambda_adv, loss_out):

    loss_df = pd.DataFrame(columns=['iteration', 'fold', 'disc_loss',
                                    'gen_L2_loss', 'gen_adv_loss'])
    loss_df.set_index(['iteration', 'fold'], inplace=True)

    max_iter = solver_param.max_iter
    snapshot = solver_param.snapshot
    test_interval = solver_param.test_interval
    test_iter = solver_param.test_iter[0]

    times = []
    for it in range(max_iter+1):

        start = time.time()

        if it%snapshot == 0:
            disc_solver.snapshot()
            gen_solver.snapshot()

        if it%test_interval == 0:

            for fold, test_data_net in test_data_nets.items():

                loss = dict()
                loss.update(disc_step(test_data_net, gen_solver, disc_solver,
                                      test_iter, False))
                loss.update(gen_step(test_data_net, gen_solver, disc_solver,
                                     test_iter, False, lambda_L2, lambda_adv))
                for name in loss:
                    loss_df.loc[(it, fold), name] = loss[name]

                #TODO write to loss_out
                #loss_df.loc[it:it].to_csv(loss_out, header=(it==1), sep=' ')
                #loss_out.flush()

            times.append(timedelta(seconds=time.time() - start))
            time_elapsed = np.sum(times)
            time_mean = time_elapsed // len(times)
            iters_left = max_iter - it
            time_left = time_mean*iters_left

            print('Iteration {}'.format(it))
            print('  {} elapsed'.format(time_elapsed))
            print('  {} mean'.format(time_mean))
            print('  {} left'.format(time_left))
            for fold in test_data_nets:
                for name in loss:
                    print('  {}_{} = {}'.format(fold, name, loss_df.loc[(it, fold), name]))

        if it == max_iter:
            break

        disc_step(train_data_net, gen_solver, disc_solver,
                  disc_iter_mult, True)
        gen_step(train_data_net, gen_solver, disc_solver,
                 gen_iter_mult, True, lambda_L2, lambda_adv)
        it += 1
        disc_solver.increment_iter()
        gen_solver.increment_iter()

    return loss_df


def get_train_and_test_files(data_prefix, fold_nums):
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
    parser.add_argument('--gen_iter_mult', default=1, type=int)
    parser.add_argument('--disc_iter_mult', default=2, type=int)
    parser.add_argument('--gen_weights_file')
    parser.add_argument('--disc_weights_file')
    parser.add_argument('--lambda_L2', type=float, default=1.0)
    parser.add_argument('--lambda_adv', type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    for fold, train_file, test_file in get_train_and_test_files(args.data_prefix, args.fold_nums):

        data_net_param = caffe_util.NetParameter.from_prototxt(args.data_model_file)

        data_net_param.set_molgrid_data_source(train_file, args.data_root, caffe.TRAIN)
        train_data_net = caffe_util.Net.from_param(data_net_param, phase=caffe.TRAIN)

        test_data_nets = dict()
        data_net_param.set_molgrid_data_source(test_file, args.data_root, caffe.TEST)
        test_data_nets['test'] = caffe_util.Net.from_param(data_net_param, phase=caffe.TEST)

        data_net_param.set_molgrid_data_source(train_file, args.data_root, caffe.TEST)
        test_data_nets['train'] = caffe_util.Net.from_param(data_net_param, phase=caffe.TEST)

        solver_param = caffe_util.SolverParameter.from_prototxt(args.solver_file)
        solver_param.max_iter = args.max_iter
        solver_param.random_seed = args.random_seed
        solver_param.snapshot = args.snapshot
        solver_param.test_iter.append(args.test_iter)
        solver_param.test_interval = args.test_interval

        gen_net_param = caffe_util.NetParameter.from_prototxt(args.gen_model_file)
        gen_net_param.force_backward = True
        gen_solver = caffe_util.Solver.from_param(solver_param, net_param=gen_net_param,
                                                  snapshot_prefix=args.out_prefix + '_gen')
        if args.gen_weights_file:
            gen_solver.net.copy_from(args.gen_weights_file)

        disc_net_param = caffe_util.NetParameter.from_prototxt(args.disc_model_file)
        disc_net_param.force_backward = True
        disc_solver = caffe_util.Solver.from_param(solver_param, net_param=disc_net_param,
                                                   snapshot_prefix=args.out_prefix + '_disc')
        if args.disc_weights_file:
            disc_solver.net.copy_from(args.disc_weights_file)

        loss_file = '{}_{}_loss.csv'.format(args.out_prefix, fold)
        loss_out = open(loss_file, 'w')

        try:
            loss_df = train_gan_model(train_data_net, test_data_nets, gen_solver, disc_solver,
                                      solver_param, args.gen_iter_mult, args.disc_iter_mult,
                                      args.lambda_L2, args.lambda_adv, loss_out)
        except:
            disc_solver.snapshot()
            gen_solver.snapshot()
            raise

        print(loss_df)

       # plot_file = '{}_{}_loss.pdf'.format(args.out_prefix, fold)
       # training_plot(plot_file, loss_df)


if __name__ == '__main__':
    main(sys.argv[1:])
