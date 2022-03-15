#!/usr/bin/env python3
import sys, os, argparse, yaml

import liGAN
from openbabel import openbabel as ob


def parse_args(argv):
    parser = argparse.ArgumentParser(description='train a deep neural network to generate atomic density grids')
    parser.add_argument('config_file')
    parser.add_argument('--debug', default=False, action='store_true')
    # TODO reimplement the following arguments
    parser.add_argument('--instance_noise', type=float, default=0.0, help='standard deviation of disc instance noise (default 0.0)')
    # removing wandb option in place of putting wandb configs in the config file
    # parser.add_argument('--wandb', action='store_true', help='enable weights and biases')
    parser.add_argument('--lr_policy', type=str, help='learning rate policy')
    return parser.parse_args(argv)


def main(argv):
    ob.obErrorLog.SetOutputLevel(0)
    args = parse_args(argv)

    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    if 'wandb' in config and 'use_wandb' not in config['wandb']:
        raise Exception('use_wandb must be included in wandb configs')

    if 'wandb' in config and config['wandb']['use_wandb']:
        import wandb
        if 'init_kwargs' in config['wandb']:
            wandb.init(settings=wandb.Settings(start_method="fork"),
            config=config, **config['wandb']['init_kwargs'])
        else:
            wandb.init(settings=wandb.Settings(start_method="fork"), config=config)
        if 'out_prefix' not in config:
            try:
                os.mkdir('wandb_output')
            except FileExistsError:
                pass
            config['out_prefix'] = 'wandb_output/' + wandb.run.id
            sys.stderr.write(
                'Setting output prefix to {}\n'.format(config['out_prefix'])
            )

    device = 'cuda'
    liGAN.set_random_seed(config.get('random_seed', None))

    solver_type = getattr(
        liGAN.training, config['model_type'] + 'Solver'
    )
    solver = solver_type(
        data_kws=config['data'],
        wandb_kws=config.get('wandb', {'use_wandb': False}),
        gen_model_kws=config['gen_model'],
        disc_model_kws=config.get('disc_model', {}),
        prior_model_kws=config.get('prior_model', {}),
        loss_fn_kws=config['loss_fn'],
        gen_optim_kws=config['gen_optim'],
        disc_optim_kws=config.get('disc_optim', {}),
        prior_optim_kws=config.get('prior_optim', {}),
        atom_fitting_kws=config['atom_fitting'],
        bond_adding_kws=config.get('bond_adding', {}),
        out_prefix=config['out_prefix'],
        device=device,
        debug=args.debug,
        sync_cuda=config.get('sync_cuda', False)
    )

    if config['continue']:
        if config['continue'] is True:
            cont_iter = None
        else:
            cont_iter = config['continue']
        try:
            solver.load_state_and_metrics(cont_iter)
        except FileNotFoundError:
            pass

    if 'max_n_iters' in config:
        config['train']['max_iter'] = min(
            config['train']['max_iter'],
            solver.gen_iter + config['max_n_iters']
        )

    solver.train_and_test(**config['train'])


if __name__ == '__main__':
    main(sys.argv[1:])
