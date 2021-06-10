#!/usr/bin/env python3
import sys, os, argparse, yaml

import liGAN


def parse_args(argv):
    parser = argparse.ArgumentParser(description='train a deep neural network to generate atomic density grids')
    parser.add_argument('config_file')
    parser.add_argument('--debug', default=False, action='store_true')
    # TODO reimplement the following arguments
    parser.add_argument('--instance_noise', type=float, default=0.0, help='standard deviation of disc instance noise (default 0.0)')
    parser.add_argument('--wandb', action='store_true', help='enable weights and biases')
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

    if 'random_seed' in config:
        liGAN.set_random_seed(config['random_seed'])

    solver_type = getattr(
        liGAN.training, config['model_type'] + 'Solver'
    )

    solver = solver_type(
        train_file=config['data'].pop('train_file'),
        test_file=config['data'].pop('test_file'),
        data_kws=config['data'],
        gen_model_kws=config['gen_model'],
        disc_model_kws=config.get('disc_model', None),
        loss_fn_kws=config['loss_fn'],
        gen_optim_kws=config['gen_optim'],
        disc_optim_kws=config.get('disc_optim', None),
        atom_fitting_kws=config['atom_fitting'],
        bond_adding_kws=config.get('bond_adding', {}),
        out_prefix=config['out_prefix'],
        caffe_init=config['caffe_init'],
        balance=config['balance'],
        device='cuda',
        debug=args.debug
    )

    if config['continue']:
        solver.load_state()
        solver.load_metrics()

    if 'max_n_iters' in config:
        config['train']['max_iter'] = min(
            config['train']['max_iter'],
            solver.curr_iter + config['max_n_iters']
        )

    solver.train(**config['train'])


if __name__ == '__main__':
    main(sys.argv[1:])
