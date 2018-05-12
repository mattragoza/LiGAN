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


def disc_step(data_net, gen_solver, disc_solver, n_iter, train):

    gen_net  = gen_solver.net
    disc_net = disc_solver.net

    batch_size = data_net.blobs['lig'].shape[0]
    half1 = np.arange(batch_size) < batch_size//2
    half2 = ~half1

    disc_loss_names = [n for n in disc_net.blobs if n.endswith('loss')]
    disc_loss_dict = {n: np.zeros(n_iter) for n in disc_loss_names}

    for i in range(n_iter):

        if i%2 == 0: # first half real, second half gen

            data_out = data_net.forward()
            rec_real = data_out['rec']
            lig_real = data_out['lig']

            gen_out = gen_net.forward(rec=rec_real, lig=lig_real)
            lig_gen = gen_out['lig_gen']

            lig_bal = np.concatenate([lig_real[half1,...], lig_gen[half2,...]])
            disc_out = disc_net.forward(rec=rec_real, lig=lig_bal, label=half1)

            for n in disc_loss_names:
                disc_loss_dict[n][i] = disc_out[n]

            if train:
                disc_net.clear_param_diffs()
                disc_net.backward()
                disc_solver.apply_update()

        else: # first half gen, second half real

            lig_bal = np.concatenate([lig_gen[half1,...], lig_real[half2,...]])
            disc_out = disc_net.forward(rec=rec_real, lig=lig_bal, label=half2)

            for n in disc_loss_names:
                disc_loss_dict[n][i] = disc_out[n]

            if train:
                disc_net.clear_param_diffs()
                disc_net.backward()
                disc_solver.apply_update()

    return {n: l.mean() for n,l in disc_loss_dict.items()}


def gen_step(data_net, gen_solver, disc_solver, n_iter, train):

    gen_net  = gen_solver.net
    disc_net = disc_solver.net

    batch_size = data_net.blobs['lig'].shape[0]

    gen_loss_names = [n for n in gen_net.blobs if n.endswith('loss')]
    gen_loss_dict = {n: np.zeros(n_iter) for n in gen_loss_names}

    disc_loss_names = [n for n in disc_net.blobs if n.endswith('loss')]
    disc_loss_dict = {n: np.zeros(n_iter) for n in disc_loss_names}

    for i in range(n_iter):

        data_out = data_net.forward()
        rec_real = data_out['rec']
        lig_real = data_out['lig']

        gen_out = gen_net.forward(rec=rec_real, lig=lig_real)
        lig_gen = gen_out['lig_gen']

        for n in gen_loss_names:
            gen_loss_dict[n][i] = gen_out[n]

        disc_out = disc_net.forward(rec=rec_real, lig=lig_gen, label=np.ones(batch_size))

        for n in disc_loss_names:
            disc_loss_dict[n][i] = disc_out[n]

        if train:
            disc_net.clear_param_diffs()
            disc_net.backward()
            gen_net.blobs['lig_gen'].diff[...] = disc_net.blobs['lig'].diff
            gen_net.clear_param_diffs()
            gen_net.backward()
            gen_solver.apply_update()

    return {n: l.mean() for n,l in gen_loss_dict.items()}, \
           {n: l.mean() for n,l in disc_loss_dict.items()}


def train_gan_model(train_data_net, test_data_nets, gen_solver, disc_solver, loss_out, args):

    loss_df = pd.DataFrame(index=range(args.max_iter+1))
    loss_df.index.name = 'iteration'

    times = []
    for i in range(args.max_iter+1):

        start = time.time()

        if i%args.snapshot == 0:
            disc_solver.snapshot()
            gen_solver.snapshot()

        if i%args.test_interval == 0: # test

            for fold, test_data_net in test_data_nets.items():

                disc_loss_dict = \
                    disc_step(test_data_net, gen_solver, disc_solver, args.test_iter, False)

                gen_loss_dict, gen_adv_loss_dict = \
                    gen_step(test_data_net, gen_solver, disc_solver, args.test_iter, False)

                for name, loss in disc_loss_dict.items():
                    loss_df.loc[i, fold+'_disc_'+name] = loss

                for name, loss in gen_loss_dict.items():
                    loss_df.loc[i, fold+'_gen_'+name] = loss

                for name, loss in gen_adv_loss_dict.items():
                    loss_df.loc[i, fold+'_gen_adv_'+name] = loss

            loss_df.tail(1).to_csv(loss_out, header=(i==0), sep=' ')
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

        if i == args.max_iter:
            break

        # train
        disc_step(train_data_net, gen_solver, disc_solver, args.disc_train_iter, True)
        gen_step(train_data_net, gen_solver, disc_solver, args.gen_train_iter, True)
 
        disc_solver.increment_iter()
        gen_solver.increment_iter()
        times.append(dt.timedelta(seconds=time.time() - start))

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
    parser.add_argument('--gen_train_iter', default=1, type=int)
    parser.add_argument('--disc_train_iter', default=2, type=int)
    parser.add_argument('--gen_weights_file')
    parser.add_argument('--disc_weights_file')
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
        solver_param.test_interval = args.max_iter+1
        solver_param.random_seed = args.random_seed

        gen_net_param = caffe_util.NetParameter.from_prototxt(args.gen_model_file)
        gen_net_param.force_backward = True
        gen_solver = caffe_util.Solver.from_param(solver_param, net_param=gen_net_param,
                                                  snapshot_prefix=args.out_prefix+'_gen'+'.'+fold)
        if args.gen_weights_file:
            gen_solver.net.copy_from(args.gen_weights_file)

        disc_net_param = caffe_util.NetParameter.from_prototxt(args.disc_model_file)
        disc_net_param.force_backward = True
        disc_solver = caffe_util.Solver.from_param(solver_param, net_param=disc_net_param,
                                                   snapshot_prefix=args.out_prefix+'_disc'+'.'+fold)
        if args.disc_weights_file:
            disc_solver.net.copy_from(args.disc_weights_file)

        loss_file = '{}.{}.training_output'.format(args.out_prefix, fold)
        loss_out = open(loss_file, 'w')

        try:
            loss_df = train_gan_model(train_data_net, test_data_nets, gen_solver, disc_solver,
                                      loss_out, args)
        except:
            disc_solver.snapshot()
            gen_solver.snapshot()
            raise

        plot_file = '{}.{}.pdf'.format(args.out_prefix, fold)
        training_plot(plot_file, loss_df)


if __name__ == '__main__':
    main(sys.argv[1:])
