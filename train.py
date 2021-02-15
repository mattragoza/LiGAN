#!/usr/bin/env python3
from __future__ import print_function, division
import sys, os, argparse, torch

import liGAN


def str_to_bool(s):
    '''
    Command line-friendly conversion of string to boolean.
    '''
    s = s.lower()
    if s in {'true', 'yes', 't', 'y', '1'}:
        return True
    elif s in {'false', 'no', 'f', 'n', '0'}:
        return False
    else:
        raise ValueError(
            'expected one of (true|false|yes|no|t|f|y|n|1|0)'
        )


def parse_args(argv):
    parser = argparse.ArgumentParser(description='train a deep neural network to generate atomic density grids')
    parser.add_argument('--random_seed', default=0, type=int, help='random seed for initialization/sampling')
    parser.add_argument('--data_root', required=True, help='root directory of data files (prepended to paths in train/test files)')
    parser.add_argument('--train_file', required=True, help='file of paths to training data, relative to data_root')
    parser.add_argument('--test_file', required=True, help='file of paths to test data, relative to data_root')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--rec_map_file', required=True)
    parser.add_argument('--lig_map_file', required=True)
    parser.add_argument('--resolution', default=0.5, type=float)
    parser.add_argument('--grid_dim', default=48, type=int)
    parser.add_argument('--shuffle', default=True, type=str_to_bool)
    parser.add_argument('--random_rotation', default=True, type=str_to_bool)
    parser.add_argument('--random_translation', default=2.0, type=float)
    parser.add_argument('--rec_molcache', default='')
    parser.add_argument('--lig_molcache', default='')
    parser.add_argument('--model_type', required=True, help='AE|CE|VAE|CVAE')
    parser.add_argument('--n_filters', default=32, type=int)
    parser.add_argument('--width_factor', default=2, type=int)
    parser.add_argument('--n_levels', default=4, type=int)
    parser.add_argument('--conv_per_level', default=3, type=int)
    parser.add_argument('--kernel_size', default=3, type=int)
    parser.add_argument('--relu_leak', default=0.1, type=float)
    parser.add_argument('--pool_type', default='a', help='m|a|c')
    parser.add_argument('--unpool_type', default='n', help='n|c')
    parser.add_argument('--pool_factor', default=2, type=int)
    parser.add_argument('--n_latent', default=1024, type=int)
    parser.add_argument('--init_conv_pool', default=False, type=str_to_bool)
    parser.add_argument('--optim_type', default='Adam', help='SGD|Adam')
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--momentum2', default=0.999, type=float)
    parser.add_argument('--clip_gradient', default=0, type=float)
    parser.add_argument('--max_iter', default=100000, type=int, help='maximum number of training iterations (default 100,000)')
    parser.add_argument('--test_interval', default=100, type=int, help='evaluate test data every # train iters (default 100)')
    parser.add_argument('--n_test_batches', default=10, type=int, help='# test batches to evaluate every test_interval (default 10)')
    parser.add_argument('--save_interval', default=10000, type=int, help='save weights every # train iters (default 10,000)')
    parser.add_argument('--out_prefix', required=True)

    # TODO reimplement the following arguments
    parser.add_argument('--gen_weights_file', help='.caffemodel file to initialize gen weights')
    parser.add_argument('--disc_weights_file', help='.caffemodel file to initialize disc weights')
    parser.add_argument('--cont_iter', default=0, type=int, help='continue training from iteration #')
    parser.add_argument('--gen_train_iter', default=2, type=int, help='number of sub-iterations to train gen model each train iter (default 20)')
    parser.add_argument('--disc_train_iter', default=2, type=int, help='number of sub-iterations to train disc model each train iter (default 20)')
    parser.add_argument('--alternate', default=False, action='store_true', help='alternate between encoding and sampling latent prior')
    parser.add_argument('--balance', default=False, action='store_true', help='dynamically train gen/disc each iter by balancing GAN loss')
    parser.add_argument('--instance_noise', type=float, default=0.0, help='standard deviation of disc instance noise (default 0.0)')
    parser.add_argument('--gen_grad_norm', default=False, action='store_true', help='gen gradient normalization')
    parser.add_argument('--disc_grad_norm', default=False, action='store_true', help='disc gradient normalization')
    parser.add_argument('--gen_spectral_norm', default=False, action='store_true', help='gen spectral normalization')
    parser.add_argument('--disc_spectral_norm', default=False, action='store_true', help='disc spectral normalization')
    parser.add_argument('--loss_weight', default=1.0, type=float, help='initial value for non-GAN generator loss weight')
    parser.add_argument('--loss_weight_decay', default=0.0, type=float, help='decay rate for non-GAN generator loss weight')
    parser.add_argument('--wandb',action='store_true',help='enable weights and biases')
    parser.add_argument('--lr_policy', type=str, help='learning rate policy')
    parser.add_argument('--weight_decay', type=float, help='weight decay (L2 regularization)')
    parser.add_argument('--weight_l2_only', default=0, type=int, help='apply loss weight to L2 loss only')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    if args.wandb:
        import wandb
        wandb.init(project='gentrain', config=args)
        if args.out_prefix == '':
            try:
                os.mkdir('wandb_output')
            except FileExistsError:
                pass
            args.out_prefix = 'wandb_output/' + wandb.run.id
            sys.stderr.write("Setting output prefix to %s\n" % args.out_prefix)

    with open('%s.config' % args.out_prefix, 'wt') as f:
        f.write('\n'.join(map(lambda kv: '%s : %s' % kv, vars(args).items())))

    train_data, test_data = (
        liGAN.data.AtomGridData(
            data_root=args.data_root,
            batch_size=args.batch_size,
            rec_map_file=args.rec_map_file,
            lig_map_file=args.lig_map_file,
            resolution=args.resolution,
            dimension=liGAN.atom_grids.AtomGrid.compute_dimension(
                args.grid_dim, args.resolution
            ),
            shuffle=args.shuffle,
            random_rotation=args.random_rotation,
            random_translation=args.random_translation,
            split_rec_lig=args.model_type in {'CE', 'CVAE', 'CGAN', 'CVAEGAN'},
            ligand_only=args.model_type in {'AE', 'VAE', 'GAN', 'VAEGAN'},
            rec_molcache=args.rec_molcache,
            lig_molcache=args.lig_molcache,
            device='cuda'
        ) for i in range(2))

    train_data.populate(args.train_file)
    test_data.populate(args.test_file)

    model = liGAN.models.Generator(
        n_channels_in=train_data.n_channels,
        n_channels_out=train_data.n_lig_channels,
        grid_dim=args.grid_dim,
        n_filters=args.n_filters,
        width_factor=args.width_factor,
        n_levels=args.n_levels,
        conv_per_level=args.conv_per_level,
        kernel_size=args.kernel_size,
        relu_leak=args.relu_leak,
        pool_type=args.pool_type,
        unpool_type=args.unpool_type,
        pool_factor=args.pool_factor,
        n_latent=args.n_latent,
        init_conv_pool=args.init_conv_pool,
        var_input=dict(VAE=0, CVAE=1).get(args.model_type, None)
    ).cuda()

    solver = getattr(
        liGAN.training, args.model_type + 'Solver'
    )(
        train_data=train_data,
        test_data=test_data,
        model=model,
        loss_fn=lambda yp, yt: ((yt - yp)**2).sum() / 2 / yt.shape[0],
        optim_type=dict(
            SGD=torch.optim.SGD,
            Adam=torch.optim.Adam
        )[args.optim_type],
        save_prefix=args.out_prefix,
        lr=args.learning_rate,
        #momentum=args.momentum,
        betas=(args.momentum, args.momentum2),
    )

    solver.train(
        max_iter=args.max_iter,
        test_interval=args.test_interval,
        n_test_batches=args.n_test_batches,
        save_interval=args.save_interval
    )

    for fold, train_file, test_file in get_train_and_test_files(args.data_prefix, args.fold_nums):

        # create nets for producing train and test data
        print('Creating train data net')
        data_param.set_molgrid_data_source(train_file, args.data_root)
        train_data = Net.from_param(data_param, phase=caffe.TRAIN)

        print('Creating test data net')
        test_data = {}
        data_param.set_molgrid_data_source(train_file, args.data_root)
        test_data['train'] = Net.from_param(data_param, phase=caffe.TEST)
        if test_file != train_file:
            data_param.set_molgrid_data_source(test_file, args.data_root)
            test_data['test'] = Net.from_param(data_param, phase=caffe.TEST)

        # create solver for training generator net
        print('Creating generator solver')
        gen_prefix = '{}_{}_gen'.format(args.out_prefix, fold)
        gen = Solver.from_param(solver_param, net_param=gen_param, snapshot_prefix=gen_prefix)
        if args.gen_weights_file:
            gen.net.copy_from(args.gen_weights_file)
        if 'lig_gauss_conv' in gen.net.blobs:
            gen.net.copy_from('lig_gauss_conv.caffemodel')

        # create solver for training discriminator net
        print('Creating discriminator solver')
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
