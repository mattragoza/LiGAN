from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import sys
import os
import re
import glob
import argparse
import parse
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('talk', rc={'lines.linewidth': 1.0})
sns.set_palette('deep')

import models


def plot_lines(plot_file, df, x, y, hue, n_cols=None):
    if hue:
        df = df.reset_index().set_index([hue, x])
    else:
        df = df.reset_index().set_index(x)
    if n_cols is None:
        n_cols = len(y)
    n_axes = len(y)
    n_rows = (n_axes + n_cols-1)//n_cols
    n_cols = min(n_axes, n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.reshape(n_rows, n_cols)
    axes = iter(axes.flatten())
    for i, y_ in enumerate(y):
        ax = next(axes)
        ax.set_xlabel(x)
        ax.set_ylabel(y_)
        if hue:
            for j, _ in df.groupby(level=0):
                mean = df.loc[j][y_].groupby(level=0).mean()
                sem = df.loc[j][y_].groupby(level=0).sem()
                ax.fill_between(mean.index, mean-sem, mean+sem, alpha=0.5/df.index.get_level_values(hue).nunique())
            for j, _ in df.groupby(level=0):
                mean = df.loc[j][y_].groupby(level=0).mean()
                ax.plot(mean.index, mean, label=j)
        else:
            mean = df[y_].groupby(level=0).mean()
            sem = df[y_].groupby(level=0).sem()
            ax.fill_between(mean.index, mean-sem, mean+sem, alpha=0.5)
            ax.plot(mean.index, mean)
    fig.tight_layout()
    extra = []
    if hue:
        lgd = ax.legend(loc='upper left', bbox_to_anchor=(1.00, 1.025), ncol=1)
        lgd.set_title(hue, prop=dict(size='small'))
        extra.append(lgd)
    for ax in axes:
        ax.axis('off')
    fig.savefig(plot_file, bbox_extra_artists=extra, bbox_inches='tight')


def plot_strips(plot_file, df, x, y, hue, n_cols=None):
    df = df.reset_index()
    if n_cols is None:
        n_cols = len(x)
    n_axes = len(x)*len(y)
    n_rows = (n_axes + n_cols-1)//n_cols
    n_cols = min(n_axes, n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows),
                             sharex=len(x) == 1, sharey=len(y) == 1, squeeze=False)
    axes = axes.reshape(n_rows, n_cols)
    axes = iter(axes.flatten())
    for i, y_ in enumerate(y):
        for j, x_ in enumerate(x):
            ax = next(axes)
            sns.stripplot(data=df, x=x_, y=y_, hue=hue, jitter=True, alpha=0.5, ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            sns.pointplot(data=df, x=x_, y=y_, hue=hue, dodge=True, markers='', capsize=0.1, ax=ax)
            if hue:
                ax.legend_.remove()
    fig.tight_layout()
    extra = []
    if hue:
        lgd = ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.00, 1.025), ncol=1)
        lgd.set_title(hue, prop=dict(size='small'))
        extra.append(lgd)
    for ax in axes:
        ax.axis('off')
    fig.savefig(plot_file, bbox_extra_artists=extra, bbox_inches='tight')


def read_training_output_files(model_dirs, data_name, seeds, folds, iteration):
    all_model_dfs = []
    for model_dir in model_dirs:
        model_dfs = []
        model_name = model_dir.rstrip('/\\')
        model_prefix = os.path.join(model_dir, model_name)
        model_errors = dict()
        for seed in seeds:
            for fold in folds:
                try:
                    file_ = '{}.{}.{}.{}.training_output'.format(model_prefix, data_name, seed, fold)
                    file_df = pd.read_csv(file_, sep=' ')
                    file_df['model_name'] = model_name
                    #file_df['data_name'] = data_name #TODO allow multiple data sets
                    file_df['seed'] = seed
                    file_df['fold'] = fold
                    file_df['iteration'] = file_df['iteration'].astype(int)
                    del file_df['base_lr']
                    max_iter = file_df['iteration'].max()
                    assert iteration in file_df['iteration'].unique(), \
                        'No training output for iteration {} ({})'.format(iteration, max_iter)
                    model_dfs.append(file_df)
                except (IOError, pd.io.common.EmptyDataError, AssertionError) as e:
                    model_errors[file_] = e
        if not model_errors:
            all_model_dfs.extend(model_dfs)
        else:
            for f, e in model_errors.items():
                print('{}: {}'.format(f, e))
    return pd.concat(all_model_dfs)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--dir_pattern', default=[], action='append', required=True)
    parser.add_argument('-d', '--data_name', default='lowrmsd')
    parser.add_argument('-o', '--out_prefix', default='')
    parser.add_argument('-s', '--seeds', default='0')
    parser.add_argument('-f', '--folds', default='0,1,2')
    parser.add_argument('-i', '--iteration', default=20000, type=int)
    parser.add_argument('--plot_ext', default='png')
    parser.add_argument('-r', '--rename_col', default=[], action='append')
    parser.add_argument('-x', '--x', default=[], action='append')
    parser.add_argument('-y', '--y', default=[], action='append')
    parser.add_argument('--hue', default=None)
    parser.add_argument('--n_cols', default=4, type=int)
    parser.add_argument('--masked', default=False, action='store_true')
    parser.add_argument('--plot_lines', default=False, action='store_true')
    parser.add_argument('--plot_strips', default=False, action='store_true')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    # read training output files from found model directories
    model_dirs = sorted(d for p in args.dir_pattern for d in glob.glob(p) if os.path.isdir(d))
    seeds = map(int, args.seeds.split(','))
    folds = map(int, args.folds.split(','))
    df = read_training_output_files(model_dirs, args.data_name, seeds, folds, args.iteration)

    # aggregate output values for each model across seeds and folds
    index_cols = ['model_name', 'iteration']
    f = {col: pd.Series.nunique if col in {'seed', 'fold'} else np.mean \
            for col in df if col not in index_cols}
    agg_df = df.groupby(index_cols).agg(f)
    assert np.all(agg_df['seed'] == len(seeds))
    assert np.all(agg_df['fold'] == len(folds))

    # add columns from parsing model name fields
    for model_name, model_df in agg_df.groupby(level=0):
        model_version = tuple(int(c) for c in re.match(r'^.+e(\d+)_', model_name).group(1))
        agg_df.loc[model_name, 'model_version'] = str(model_version)
        name_format = models.NAME_FORMATS[model_version]
        name_parse = parse.parse(name_format, model_name)
        name_fields = sorted(name_parse.named, key=name_parse.spans.get)
        for field in name_fields:
            value = name_parse.named[field]
            agg_df.loc[model_name, field] = value
    name_fields.insert(0, 'model_version')

    # fill in default values so that different model versions may be compared
    if 'resolution' in agg_df:
        agg_df['resolution'] = agg_df['resolution'].fillna(0.5)

    if 'conv_per_level' in agg_df:
        agg_df = agg_df[agg_df['conv_per_level'] > 0]

    if 'width_factor' in agg_df:
        agg_df['width_factor'] = agg_df['width_factor'].fillna(1)
        agg_df = agg_df[agg_df['width_factor'] < 3]

    if 'n_latent' in agg_df:
        agg_df['n_latent'] = agg_df['n_latent'].fillna(0)

    if 'loss_types' in agg_df:
        agg_df['loss_types'] = agg_df['loss_types'].fillna('e')
        agg_df = agg_df[(agg_df['loss_types'] == 'e') | (agg_df['loss_types'] == 'em')]

    if args.masked: # treat aff_loss as masked y_loss; adjust for resolution

        no_aff = agg_df['test_aff_loss'].isnull()
        agg_df.loc[no_aff, 'test_aff_loss'] = agg_df[no_aff]['test_y_loss']
        agg_df['test_aff_loss'] *= agg_df['resolution']**3

        no_aff = agg_df['train_aff_loss'].isnull()
        agg_df.loc[no_aff, 'train_aff_loss'] = agg_df[no_aff]['train_y_loss']
        agg_df['train_aff_loss'] *= agg_df['resolution']**3

    # rename columns if necessary
    agg_df.reset_index(inplace=True)
    col_name_map = {col: col for col in agg_df}
    col_name_map.update(dict(r.split(':') for r in args.rename_col))
    agg_df.rename(columns=col_name_map, inplace=True)
    name_fields = [col_name_map[n] for n in name_fields]

    # by default, plot the test_y_loss on the y-axis
    if not args.y:
        args.y.append(col_name_map['test_y_loss'])

    # by default, don't make separate plots for the hue variable or variables with 1 unique value
    if not args.x:
        args.x = [n for n in name_fields if n != args.hue and agg_df[n].nunique() > 1]

    if args.plot_lines: # plot training progress
        line_plot_file = '{}_lines.{}'.format(args.out_prefix, args.plot_ext)
        plot_lines(line_plot_file, agg_df, x=col_name_map['iteration'], y=args.y, hue=args.hue,
                   n_cols=args.n_cols)

    final_df = agg_df.set_index(col_name_map['iteration']).loc[args.iteration]
    
    if args.plot_strips: # plot final loss distributions
        strip_plot_file = '{}_strips.{}'.format(args.out_prefix, args.plot_ext)
        plot_strips(strip_plot_file, final_df, x=args.x, y=args.y, hue=args.hue,
                    n_cols=args.n_cols)

    # display names of best models
    for y in args.y:
        print(final_df.sort_values(y).head(5).loc[:, (col_name_map['model_name'], y)])


if __name__ == '__main__':
    main(sys.argv[1:])
