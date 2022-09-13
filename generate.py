#!/usr/bin/env python3
import sys, os, argparse, yaml

import ligan


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Generate atomic density grids from generative model'
    )
    parser.add_argument('config_file')
    parser.add_argument('--debug', default=False, action='store_true')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    ligan.set_random_seed(config.get('random_seed'))

    generator_type = config.get('model_type', 'Molecule')
    generator_type = generator_type + 'Generator'
    generator_type = getattr(ligan.generating, generator_type)
    generator = generator_type(
        out_prefix=config['out_prefix'],
        n_samples=config['generate']['n_samples'],
        fit_atoms=config['generate'].get('fit_atoms', True),
        data_kws=config['data'],
        gen_model_kws=config.get('gen_model', {}),
        prior_model_kws=config.get('prior_model', {}),
        bond_adding_kws=config.get('bond_adding', {}),
        output_kws=config['output'],
        device=config.get('device', 'cuda'),
        verbose=config.get('verbose', False),
        debug=args.debug,
    )
    generator.generate(**config['generate'])
    print('Done')


if __name__ == '__main__':
    main(sys.argv[1:])
