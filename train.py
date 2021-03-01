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
    parser.add_argument('--grid_size', default=48, type=int)
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
    parser.add_argument('--kldiv_loss_wt', default=1.0, type=float)
    parser.add_argument('--recon_loss_wt', default=1.0, type=float)
    parser.add_argument('--gan_loss_wt', default=1.0, type=float)
    parser.add_argument('--recon_loss_type', default='2', help='1|2')
    parser.add_argument('--gan_loss_type', default='x', help='x|w')
    parser.add_argument('--optim_type', default='Adam', help='SGD|Adam')
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--max_iter', default=100000, type=int, help='maximum number of training iterations (default 100,000)')
    parser.add_argument('--n_gen_train_iters', default=2, type=int, help='number of sub-iterations to train gen model each train iter (default 20)')
    parser.add_argument('--n_disc_train_iters', default=2, type=int, help='number of sub-iterations to train disc model each train iter (default 20)')
    parser.add_argument('--gen_grad_norm', default='0', help='0|2|s')
    parser.add_argument('--disc_grad_norm', default='0', help='0|2|s')
    parser.add_argument('--test_interval', default=100, type=int, help='evaluate test data every # train iters (default 100)')
    parser.add_argument('--n_test_batches', default=10, type=int, help='# test batches to evaluate every test_interval (default 10)')
    parser.add_argument('--save_interval', default=10000, type=int, help='save weights every # train iters (default 10,000)')
    parser.add_argument('--out_prefix', required=True)

    # TODO reimplement the following arguments
    parser.add_argument('--gen_weights_file', help='.caffemodel file to initialize gen weights')
    parser.add_argument('--disc_weights_file', help='.caffemodel file to initialize disc weights')
    parser.add_argument('--cont_iter', default=0, type=int, help='continue training from iteration #')
    parser.add_argument('--alternate', default=False, action='store_true', help='alternate between encoding and sampling latent prior')
    parser.add_argument('--balance', default=False, action='store_true', help='dynamically train gen/disc each iter by balancing GAN loss')
    parser.add_argument('--instance_noise', type=float, default=0.0, help='standard deviation of disc instance noise (default 0.0)')
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

    solver = getattr(
        liGAN.training, args.model_type + 'Solver'
    )(
        data_root=args.data_root,
        train_file=args.train_file,
        test_file=args.test_file,
        batch_size=args.batch_size,
        rec_map_file=args.rec_map_file,
        lig_map_file=args.lig_map_file,
        resolution=args.resolution,
        grid_size=args.grid_size,
        shuffle=args.shuffle,
        random_rotation=args.random_rotation,
        random_translation=args.random_translation,
        rec_molcache=args.rec_molcache,
        lig_molcache=args.lig_molcache,
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
        loss_weights=dict(
            kldiv_loss=args.kldiv_loss_wt,
            recon_loss=args.recon_loss_wt,
            gan_loss=args.gan_loss_wt
        ),
        loss_types=dict(
            recon_loss=args.recon_loss_type,
            gan_loss=args.gan_loss_type,
        ),
        grad_norms=dict(
            gen=args.gen_grad_norm,
            disc=args.disc_grad_norm,
        ),
        optim_type=getattr(torch.optim, args.optim_type),
        optim_kws=dict([
            ('lr', args.learning_rate)
        ] + dict(
            SGD=[('momentum', args.momentum)],
            Adam=[('betas', (args.beta1, args.beta2))]
        ).get(args.optim_type, [])
        ),
        save_prefix='TEST',
        device='cuda'
    )

    if solver.adversarial:
        solver.train(
            max_iter=args.max_iter,
            n_gen_train_iters=args.n_gen_train_iters,
            n_disc_train_iters=args.n_disc_train_iters,
            test_interval=args.test_interval,
            n_test_batches=args.n_test_batches, 
            save_interval=args.save_interval,
        )
    else:
        solver.train(
            max_iter=args.max_iter,
            test_interval=args.test_interval,
            n_test_batches=args.n_test_batches,
            save_interval=args.save_interval
        )


if __name__ == '__main__':
    main(sys.argv[1:])
