#!/usr/bin/env python3
import sys, os, argparse, yaml, ast

import liGAN


def parse_args(argv):
    parser = argparse.ArgumentParser(description='train a deep neural network to generate atomic density grids')
    parser.add_argument('config_file')
    parser.add_argument('--debug', default=False, action='store_true')

    # TODO reimplement the following arguments
    parser.add_argument('--gen_weights_file', help='.caffemodel file to initialize gen weights')
    parser.add_argument('--disc_weights_file', help='.caffemodel file to initialize disc weights')
    parser.add_argument('--cont_iter', default=0, type=int, help='continue training from iteration #')
    parser.add_argument('--alternate', default=False, action='store_true', help='alternate between encoding and sampling latent prior')
    parser.add_argument('--balance', default=False, action='store_true', help='dynamically train gen/disc each iter by balancing GAN loss')
    parser.add_argument('--instance_noise', type=float, default=0.0, help='standard deviation of disc instance noise (default 0.0)')
    parser.add_argument('--wandb',action='store_true',help='enable weights and biases')
    parser.add_argument('--lr_policy', type=str, help='learning rate policy')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    if args.wandb:
        import wandb
        wandb.init(project='gentrain', config=config)
        if 'out_prefix' not in config:
            try:
                os.mkdir('wandb_output')
            except FileExistsError:
                pass
            config['out_prefix'] = 'wandb_output/' + wandb.run.id
            sys.stderr.write(
                'Setting output prefix to {}\n'.format(config['out_prefix'])
            )

    solver_type = getattr(
        liGAN.training, config['model_type'] + 'Solver'
    )

    solver = solver_type(
        train_file=config['data'].pop('train_file'),
        test_file=config['data'].pop('test_file'),
        data_kws=config['data'],
        gen_model_kws=config['gen_model'],
        disc_model_kws=config['disc_model'],
        loss_fn_kws=config['loss_fn'],
        gen_optim_kws=config['gen_optim'],
        disc_optim_kws=config['disc_optim'],
        atom_fitting_kws=config['atom_fitting'],
        out_prefix=config['out_prefix'],
        random_seed=config['random_seed'],
        device='cuda',
        debug=args.debug
    )

    if config['continue']:
        solver.load_state()
        solver.load_metrics()

    solver.train(**config['train'])


if __name__ == '__main__':
    main(sys.argv[1:])
