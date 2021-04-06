import sys, os, argparse, yaml
import pandas as pd
import torch
import liGAN


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='generate atomic density grids from a deep neural network'
    )
    parser.add_argument('config_file')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    with open(args.config_file) as f:
        config = yaml.load(f)

    device = 'cuda'
    if 'random_seed' in config:
        liGAN.set_random_seed(config['random_seed'])

    print('Loading data')
    data_file = config['data'].pop('data_file')
    data = liGAN.data.AtomGridData(device=device, **config['data'])
    data.populate(data_file)

    print('Initializing generative model')
    gen_model_type = getattr(
        liGAN.models, config.pop('model_type')
    )
    gen_model_state = config['gen_model'].pop('state')
    gen_model = gen_model_type(
        n_channels_in=(data.n_lig_channels + data.n_rec_channels),
        n_channels_cond=data.n_rec_channels,
        n_channels_out=data.n_lig_channels,
        grid_size=data.grid_size,
        device=device,
        **config['gen_model']
    )
    print('Loading generative model state')
    gen_model.load_state_dict(torch.load(gen_model_state))

    print('Initializing atom fitter')
    atom_fitter = liGAN.atom_fitting.AtomFitter(
        device=device, **config['atom_fitting']
    )

    index_cols = ['lig_name', 'sample_idx']
    metrics = pd.DataFrame(columns=index_cols).set_index(index_cols)

    do_the_thing(data, gen_model, atom_fitter, metrics)


if __name__ == '__main__':
    main(sys.argv[1:])
