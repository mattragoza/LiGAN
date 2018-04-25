from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import sys
import os
import re
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('talk')

import generate


def plot_lines(plot_file, mean_data, models, colors, ylim=None, sem_data=None):
    # data should be indexed by model, iteration
    fig, axes = plt.subplots(1,2, figsize=(12,9))
    for i, loss in enumerate(['recon']):
        for j, part in enumerate(['train', 'test']):
            column = '%s_%s_loss / data_size' % (part, loss)
            ax = axes[j] #axes[i][j]
            ax.set_xlabel('iteration')
            ax.set_ylabel(column)
            if ylim:
                ax.set_ylim(*ylim)
            for model, color in zip(models, colors):
                color = np.array(color)
                light_color = (1. + color)/2.
                mean_series = mean_data.loc[model][column]
                if sem_data:
                    sem_series = sem_data.loc[model][column]
                if len(mean_series.dropna()) > 0:
                    if sem_data:
                        ax.fill_between(mean_series.index, mean_series-sem_series, mean_series+sem_series, color=light_color, alpha=0.5)
                    ax.plot(mean_series.index, mean_series, label=model, color=color, linewidth=1.0)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    #lgd = axes[0].legend(loc='upper left', bbox_to_anchor=(-0.1, -0.1), ncol=5)
    #fig.savefig(plot_file, bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig(plot_file, bbox_inches='tight')


def rename_column(c):
    return c.replace('y_loss', 'recon_loss') \
            .replace('rmsd_loss', 'spatial_loss') \
            .replace('aff_loss', 'kldiv_loss')


def column_key(c):
    return 'kldiv' in c, 'spatial' in c, 'recon' in c, 'test' in c


def read_training_output_files(model_names, data_name, seeds, folds):

    all_model_dfs = []
    for model_name in model_names:
        try:
            model_dfs = []
            model_prefix = os.path.join(model_name, model_name)
            for seed in seeds:
                for fold in folds:
                    train_out_file = '{}.{}.{}.{}.training_output' \
                                     .format(model_prefix, data_name, seed, fold)
                    train_out_df = pd.read_csv(train_out_file, sep=' ', index_col=0)
                    train_out_df['model_name'] = model_name
                    train_out_df['data_name'] = data_name
                    train_out_df['seed'] = seed
                    train_out_df['fold'] = fold
                    model_dfs.append(train_out_df)
            dfs.extend(model_dfs)
        except (IOError, pd.io.common.EmptyDataError, IndexError) as e:
            print(e, file=sys.stderr)

    return pd.concat(all_model_dfs)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_pattern')
    parser.add_argument('-o', '--out_prefix', default='')
    parser.add_argument('-s', '--seeds', default='0')
    parser.add_argument('-n', '--folds', default='0,1,2')
    parser.add_argument('-i', '--iteration', default=20000, type=int)
    parser.add_argument('--plot_ext', default='png')
    parser.add_argument('--n_channels', type=int)
    parser.add_argument('--data_dim', type=int)
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    model_names = [d.rstrip('/\\') for d in glob.glob(args.dir_pattern)]
    seeds = map(int, args.seeds.split(','))
    folds = map(int, args.folds.split(','))

    # output file names
    plot_ext = 'png'
    bar_plot_file   = '{}_bars.{}'.format(args.out_prefix, args.plot_ext)
    line_plot_file  = '{}_lines.{}'.format(args.out_prefix, args.plot_ext)
    strip_plot_file = '{}_strips.{}'.format(args.out_prefix, args.plot_ext)
    agg_pymol_file = '{}.pymol'.format(args.out_prefix)

    # autoencoder predicts rec and lig
    if dir_pattern[:2] == 'ae':
        ylim = [0.0, 0.0022] #[0.0, 0.005]

    # context encoder predicts lig only
    elif dir_pattern[:2] == 'ce':
        ylim = [0.0016, 0.0022]
        ylim = [0.00024, 0.00028]

    data_size = args.n_channels*args.data_dim**3

    data = read_training_output_files(model_names, seeds, folds, args.iteration)

    # format data and edit columns
    data.columns = data.columns.map(rename_column)
    data['test_recon_loss / data_size'] = data['test_recon_loss'] / data_size
    data['train_recon_loss / data_size'] = data['train_recon_loss'] / data_size
    data = data.groupby(['model', 'iteration']).mean().reset_index() # average across seeds and folds

    if False:
        data['n_levels']       = data['model'].apply(lambda x: int(x.split('_')[2]))
        data['conv_per_level'] = data['model'].apply(lambda x: int(x.split('_')[3]))
        data['n_filters']      = data['model'].apply(lambda x: int(x.split('_')[4]))
        data['pool_type']      = data['model'].apply(lambda x: x.split('_')[5])
        data['depool_type']    = data['model'].apply(lambda x: x.split('_')[6])
        data['n_conv']         = 2 * data['n_levels'] * data['conv_per_level']
    else: #TODO ce12_24_0.5_3_3_32_1_cf
        n_levels_idx = 3
        data['resolution']     = data['model'].apply(lambda x: float(x.split('_')[2]))
        data['n_levels']       = data['model'].apply(lambda x: int(x.split('_')[n_levels_idx]))
        data['conv_per_level'] = data['model'].apply(lambda x: int(x.split('_')[4]))
        data['n_filters']      = data['model'].apply(lambda x: int(x.split('_')[5]))
        data['growth_factor']  = data['model'].apply(lambda x: int(x.split('_')[6]))
        data['loss_types']     = data['model'].apply(lambda x: x.split('_')[7])
    data = data.set_index(['model', 'iteration'])

    # group models by n_levels or whether their baselines
    baseline_models = []
    models_by_n_levels = [[], [], [], [], []]
    for model in models:
        parts = model.split('_')
        if '0' in parts:
            baseline_models.append(model)
        else:
            n_levels = int(parts[n_levels_idx])
            models_by_n_levels[n_levels-1].append(model)
    model_groups = models_by_n_levels + [baseline_models]
    color_groups = ['Blues', 'Greens', 'Reds', 'Purples', 'YlOrBr', 'Greys']

    # sort each name group by n_levels and final test_recon_loss and construct color palette
    models = []
    colors = []
    for model_group, color_group in zip(model_groups, color_groups):
        models.extend(sorted(model_group, key=lambda m: (m.split('_')[2], data['test_recon_loss'][m][iteration])))
        colors.extend(sns.color_palette(color_group + '_r', len(model_group)))
    color_dict = dict(zip(models, colors))

    # plot reconstruction loss during training
    plot_lines(line_plot_file, data, models, colors)#, ylim=ylim)

    # look at final iteration only
    data = data.reorder_levels(['iteration', 'model']).loc[iteration]
    data = data.drop(baseline_models)
    print data

    fig, axes = plt.subplots(3,2, figsize=(8, 12), sharey=True)
    if ylim: axes[0][0].set_ylim(*ylim)
    kwargs = dict(hue='n_filters')
    xs = ['resolution', 'n_levels', 'conv_per_level', 'n_filters', 'growth_factor', 'loss_types']
    for x, ax in zip(xs, axes.flatten()):
        sns.barplot(data=data, x=x, y='test_recon_loss / data_size', ax=ax, **kwargs)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper center', ncol=3)
    #axes[2][1].axis('off')
    fig.tight_layout()
    fig.savefig(bar_plot_file)

    fig, axes = plt.subplots(3,2, figsize=(8, 12), sharey=True)
    if ylim: axes[0][0].set_ylim(*ylim)
    kwargs = dict(jitter=True, hue='n_filters')
    xs = ['resolution', 'n_levels', 'conv_per_level', 'n_filters', 'growth_factor', 'loss_types']
    for x, ax in zip(xs, axes.flatten()):
        sns.stripplot(data=data, x=x, y='test_recon_loss / data_size', ax=ax, **kwargs)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper center', ncol=3)
    #axes[2][1].axis('off')
    fig.tight_layout()
    fig.savefig(strip_plot_file)

    # identify best model(s)
    best_models = data['test_recon_loss'].nsmallest(1)
    print best_models

    pymol_files = []
    for model in best_models.index:

        model_file = '{}.model'.format(model)
        weights_file = os.path.join(model, '{}.{}.0.1_iter_{}.caffemodel'.format(model, data_name, 20000))
        data_file = 'data/lowrmsdtest0.types'
        data_root = '/home/mtr22/PDBbind/refined-set/'
        out_prefix = model
        loss_name = 'loss'

        #rec_file, lig_file = generate.best_loss_rec_and_lig(model_file, weights_file, data_file, \
        #                                                    data_root, loss_name)
        rec_file = '2v3u/2v3u_rec.pdb'
        lig_file = '2v3u/2v3u_min.sdf'

        rec_file = os.path.join(data_root, re.sub('.gninatypes', '.pdb', rec_file))
        lig_file = os.path.join(data_root, re.sub('.gninatypes', '.sdf', lig_file))
        print(model, lig_file)

        center = generate.get_center_from_sdf_file(lig_file)
        resolution = generate.get_resolution_from_model_file(model_file)

        grids = generate.generate_grids(model_file, weights_file, 'level0_deconv(\d+)', rec_file, lig_file, '')

        dx_files = generate.write_grids_to_dx_files(out_prefix, grids, center, resolution)

        atoms = generate.fit_atoms_to_grids(grids, center, resolution, max_iter=0)
        pred_file = '{}.sdf'.format(out_prefix)
        generate.write_atoms_to_sdf_file(pred_file, atoms)
        extra_files = [rec_file, lig_file, pred_file]

        pymol_file = '{}.pymol'.format(out_prefix)
        generate.write_pymol_script(pymol_file, dx_files, *extra_files)
        pymol_files.append(pymol_file)

    with open(agg_pymol_file, 'w') as f:
        f.write('\n'.join('@' + p for p in pymol_files))


if __name__ == '__main__':
    main(sys.argv[1:])
