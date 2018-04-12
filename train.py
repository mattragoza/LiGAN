from __future__ import print_function, division
import sys
import os
import argparse
import time
from datetime import timedelta
from operator import itemgetter
import numpy as np
import pandas as pd
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

import caffe_util


def disc_step(data_solver, gen_solver, disc_solver, n_iter):

    data_net = data_solver.net
    gen_net  = gen_solver.net
    disc_net = disc_solver.net

    batch_size = data_net.blobs['lig'].shape[0]
    half1 = np.arange(batch_size) < batch_size//2
    half2 = ~half1

    disc_loss = np.float32(0)
    for i in range(n_iter//2):

        data_net.forward()
        rec_real = data_net.blobs['rec'].data
        lig_real = data_net.blobs['lig'].data

        gen_net.forward(rec=rec_real, lig=lig_real)
        lig_gen = gen_net.blobs['lig_gen'].data

        lig_bal = np.concatenate([lig_real[half1,...], lig_gen[half2,...]])
        disc_net.forward(rec=rec_real, lig=lig_bal, label=half1)
        disc_loss += disc_net.blobs['loss'].data
        disc_net.clear_param_diffs()
        disc_net.backward()
        disc_solver.apply_update()

        lig_bal = np.concatenate([lig_gen[half1,...], lig_real[half2,...]])
        disc_net.forward(rec=rec_real, lig=lig_bal, label=half2)
        disc_loss += disc_net.blobs['loss'].data
        disc_net.clear_param_diffs()
        disc_net.backward()
        disc_solver.apply_update()

    return disc_loss/n_iter


def gen_step(data_solver, gen_solver, disc_solver, n_iter, lambda_l2, lambda_adv):

    data_net = data_solver.net
    gen_net  = gen_solver.net
    disc_net = disc_solver.net

    batch_size = data_net.blobs['lig'].shape[0]

    gen_l2_loss = np.float32(0)
    gen_adv_loss = np.float32(0)
    for i in range(n_iter):

        data_net.forward()
        rec_real = data_net.blobs['rec'].data
        lig_real = data_net.blobs['lig'].data

        gen_net.forward(rec=rec_real, lig=lig_real)
        gen_l2_loss += gen_net.blobs['loss'].data
        lig_gen = gen_net.blobs['lig_gen'].data

        disc_net.forward(rec=rec_real, lig=lig_gen, label=np.ones(batch_size))
        gen_adv_loss += disc_net.blobs['loss'].data
        disc_net.clear_param_diffs()
        disc_net.backward()

        gen_net.blobs['loss'].diff[...] = lambda_l2
        gen_net.blobs['lig_gen'].diff[...] = lambda_adv*disc_net.blobs['lig'].diff
        gen_net.clear_param_diffs()
        gen_net.backward()
        gen_solver.apply_update()

    return gen_l2_loss/n_iter, gen_adv_loss/n_iter


def train_gan_model(data_solver, gen_solver, disc_solver, max_iter, snapshot_iter,
                    gen_iter_mult, disc_iter_mult, lambda_l2, lambda_adv, loss_out):

    loss_df = pd.DataFrame(index=pd.RangeIndex(1, max_iter+1, name='iter'),
                           columns=('disc_loss', 'gen_l2_loss', 'gen_adv_loss'))

    times = []
    for i in range(max_iter):

        start = time.time()

        if i%snapshot_iter == 0:
            disc_solver.snapshot()
            gen_solver.snapshot()

        disc_loss = disc_step(data_solver, gen_solver, disc_solver, disc_iter_mult)

        gen_l2_loss, gen_adv_loss = gen_step(data_solver, gen_solver, disc_solver,
                                             gen_iter_mult, lambda_l2, lambda_adv)
        it = i+1

        loss_df.loc[it] = (disc_loss, gen_l2_loss, gen_adv_loss)
        loss_df.loc[it:it].to_csv(loss_out, header=(it==1), sep=' ')
        loss_out.flush()

        times.append(timedelta(seconds=time.time() - start))
        time_elapsed = np.sum(times)
        time_mean = time_elapsed // len(times)
        iters_left = max_iter - it
        time_left = time_mean*iters_left

        print('Iteration {}'.format(it))
        print('  {} elapsed'.format(time_elapsed))
        print('  {} mean'.format(time_mean))
        print('  {} left'.format(time_left))
        print('  Discriminator iteration {}'.format(it*disc_iter_mult))
        print('    disc_loss = {}'.format(disc_loss))
        print('  Generator iteration {}'.format(it*gen_iter_mult))
        print('    gen_adv_loss = {} (x{})'.format(gen_adv_loss, lambda_adv))
        print('    gen_l2_loss = {} (x{})'.format(gen_l2_loss, lambda_l2))

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
    parser.add_argument('-i', '--max_iter', default=20000, type=int)
    parser.add_argument('--snapshot_iter', default=1000, type=int)
    parser.add_argument('--gen_iter_mult', default=1, type=int)
    parser.add_argument('--disc_iter_mult', default=2, type=int)
    parser.add_argument('--gen_weights_file')
    parser.add_argument('--disc_weights_file')
    parser.add_argument('--lambda_l2', type=float, default=1.0)
    parser.add_argument('--lambda_adv', type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    for fold, train_file, test_file in get_train_and_test_files(args.data_prefix, args.fold_nums):

        solver_param = caffe_util.SolverParameter.from_prototxt(args.solver_file)
        solver_param.max_iter = args.max_iter

        data_net_param = caffe_util.NetParameter.from_prototxt(args.data_model_file)
        for layer_param in data_net_param.layer:
            if layer_param.type == 'MolGridData':
                data_param = layer_param.molgrid_data_param
                if layer_param.include[0].phase == caffe.TRAIN:
                    data_param.source = train_file
                elif layer_param.include[0].phase == caffe.TEST:
                    data_param.source = test_file
                data_param.root_folder = args.data_root
        data_solver = caffe_util.Solver.from_param(solver_param, net_param=data_net_param)

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
            loss_df = train_gan_model(data_solver, gen_solver, disc_solver,
                            args.max_iter, args.snapshot_iter,
                            args.gen_iter_mult, args.disc_iter_mult,
                            args.lambda_l2, args.lambda_adv, loss_out)
        finally:
            disc_solver.snapshot()
            gen_solver.snapshot()
            loss_out.close()


if __name__ == '__main__':
    main(sys.argv[1:])
