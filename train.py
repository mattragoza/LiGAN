from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import sys, os, argparse, time
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


def disc_step(data_net, gen_solver, disc_solver, n_iter, train, args):
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
    elif 'lig_instance_std' in disc_net.blobs:
        instance_noise_std = np.full_like(disc_net.blobs['lig_instance_std'].data, args.instance_noise)

    for i in range(n_iter):

        if i%2 == 0: # first half real, second half gen

            data_out = data_net.forward()
            rec_real = data_out['rec']
            lig_real = data_out['lig']

            if args.alternate: # sample unit ligand latent space
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
                disc_out = disc_net.forward(rec=rec_real, lig=lig_bal, label=half1,
                                            info_label=info_label)
            elif 'lig_instance_std' in disc_net.blobs:
                disc_out = disc_net.forward(rec=rec_real, lig=lig_bal, label=half1,
                                            lig_instance_std=instance_noise_std)
            else:
                disc_out = disc_net.forward(rec=rec_real, lig=lig_bal, label=half1)

            for n in disc_loss_dict:
                disc_loss_dict[n][i] = disc_out[n]

            if train:
                disc_net.clear_param_diffs()
                disc_net.backward()
                disc_solver.apply_update()

        else: # first half gen, second half real

            if args.alternate: # autoencode real ligand
                gen_out = gen_net.forward(start='lig_level0_conv0', end='lig_gen')
                lig_gen = gen_out['lig_gen']     

            lig_bal = np.concatenate([lig_gen[half1,...], lig_real[half2,...]])
            if 'info_label' in disc_net.blobs:
                info_label = np.zeros_like(disc_net.blobs['info_label'].data)
                disc_out = disc_net.forward(rec=rec_real, lig=lig_bal, label=half2,
                                            info_label=info_label)
            elif 'lig_instance_std' in disc_net.blobs:
                disc_out = disc_net.forward(rec=rec_real, lig=lig_bal, label=half2,
                                            lig_instance_std=instance_noise_std)
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


def gen_step(data_net, gen_solver, disc_solver, n_iter, train, args):
    '''
    Train or test the generative GAN component for n_iter iterations.
    '''
    gen_net  = gen_solver.net
    disc_net = disc_solver.net

    batch_size = data_net.blobs['lig'].shape[0]

    gen_loss_dict = {n: np.full(n_iter, np.nan) for n in gen_net.blobs if n.endswith('loss')}
    disc_loss_dict = {n: np.full(n_iter, np.nan) for n in disc_net.blobs if n.endswith('loss')}

    if 'lig_instance_std' in disc_net.blobs:
        instance_noise_std = np.full_like(disc_net.blobs['lig_instance_std'].data, args.instance_noise)

    for i in range(n_iter):

        # get real receptors and ligands
        data_out = data_net.forward()
        rec_real = data_out['rec']
        lig_real = data_out['lig']

        # generate fake ligands
        if args.alternate and i%2:
            # sample unit ligand latent space instead of conditioning on real ligand, no gen_loss
            gen_net.forward(end='latent_concat', rec=rec_real, lig=lig_real)
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
            disc_out = disc_net.forward(rec=rec_real, lig=lig_gen, label=np.ones(batch_size),
                                        info_label=info_label)
        elif 'lig_instance_std' in disc_net.blobs:
            disc_out = disc_net.forward(rec=rec_real, lig=lig_gen, label=np.ones(batch_size),
                                        lig_instance_std=instance_noise_std)
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

            if args.alternate and i%2: # skip gen_loss and lig encoder
                gen_net.backward(start='lig_gen', end='lig_latent_noise')
                gen_net.backward(start='rec_latent_fc', end='rec')
            else:
                gen_net.backward()

            gen_solver.apply_update()

    return {n: np.nanmean(l) for n,l in gen_loss_dict.items()}, \
           {n: np.nanmean(l) for n,l in disc_loss_dict.items()}


def train_GAN_model(train_data_net, test_data_nets, gen_solver, disc_solver,
                    loss_df, loss_out, plot_out, args):
    '''
    Train a GAN using the provided train_data_net, gen_solver, and disc_solver.
    Return the loss output from periodically testing on each of test_data_nets
    as a data frame, and write it to loss_out as training to proceeds.
    '''
    times = []
    train_disc_loss = np.nan
    train_gen_adv_loss = np.nan
    train_disc = True
    train_gen = True
    for i in range(args.cont_iter, args.max_iter+1):
        start = time.time()

        if i%args.snapshot == 0:
            disc_solver.snapshot()
            gen_solver.snapshot()

        first_cont_iter = (args.cont_iter and i == args.cont_iter)
        if i%args.test_interval == 0 and not first_cont_iter: # test

            for test_data, test_data_net in test_data_nets.items():

                disc_loss_dict = \
                    disc_step(test_data_net, gen_solver, disc_solver, args.test_iter, False, args)

                gen_loss_dict, gen_adv_loss_dict = \
                    gen_step(test_data_net, gen_solver, disc_solver, args.test_iter, False, args)

                if 'info_loss' in gen_adv_loss_dict:
                    loss = gen_adv_loss_dict.pop('info_loss')
                    loss_df.loc[(i, test_data), 'info_loss'] = loss

                for loss_name, loss in disc_loss_dict.items():
                    loss_df.loc[(i, test_data), 'disc_{}'.format(loss_name)] = loss

                for loss_name, loss in gen_loss_dict.items():
                    loss_df.loc[(i, test_data), 'gen_{}'.format(loss_name)] = loss

                for loss_name, loss in gen_adv_loss_dict.items():
                    loss_df.loc[(i, test_data), 'gen_adv_{}'.format(loss_name)] = loss

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
                for loss_name in loss_df:
                    loss = loss_df.loc[(i, test_data), loss_name]
                    print('  {} {} = {}'.format(test_data, loss_name, loss))
            sys.stdout.flush()

        if i == args.max_iter:
            break

        # dynamic G/D balancing
        if args.balance:
            if train_disc and train_disc_loss < 0.1*train_gen_adv_loss:
                train_disc = False
                train_gen = True
                print('TRAIN G ONLY')

            if not train_disc and train_disc_loss > 0.5*train_gen_adv_loss:
                train_disc = True
                train_gen = True
                print('TRAIN G AND D')

            if train_gen and train_disc_loss > train_gen_adv_loss:
                train_disc = True
                train_gen = False
                print('TRAIN D ONLY')
        else:
            train_disc = True
            train_gen = True

        # train
        disc_loss_dict = \
            disc_step(train_data_net, gen_solver, disc_solver, args.disc_train_iter, train_disc, args)
        train_disc_loss = disc_loss_dict['loss']

        gen_loss_dict, gen_adv_loss_dict = \
            gen_step(train_data_net, gen_solver, disc_solver, args.gen_train_iter, train_gen, args)
        train_gen_adv_loss = gen_adv_loss_dict['loss']

        disc_solver.increment_iter()
        gen_solver.increment_iter()

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
