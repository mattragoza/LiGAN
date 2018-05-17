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


def plot_lines(plot_file, df, x, y, hue, ylim=None):
    df = df.reset_index().set_index([hue, x])
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if ylim:
        ax.set_ylim(*ylim)
    for i, _ in df.groupby(level=0):
        mean = df.loc[i][y].groupby(level=0).mean()
        sem = df.loc[i][y].groupby(level=0).sem()
        ax.fill_between(mean.index, mean-sem, mean+sem, alpha=0.5/df.index.get_level_values(hue).nunique())
    for i, _ in df.groupby(level=0):
        mean = df.loc[i][y].groupby(level=0).mean()
        ax.plot(mean.index, mean, label=i)
    fig.tight_layout()
    lgd = ax.legend(loc='upper left', bbox_to_anchor=(1.00, 1.025), ncol=1)
    lgd.set_title(hue, prop=dict(size='small'))
    fig.savefig(plot_file, bbox_extra_artists=(lgd,), bbox_inches='tight')


def plot_strips(plot_file, df, x, y, hue, ylim=None, n_cols=None):
    df = df.reset_index()
    if n_cols is None:
        n_cols = len(x)
    n_ax = len(x)*len(y)
    n_rows = (n_ax + n_cols-1)//n_cols
    n_cols = min(n_ax, n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows),
                             sharey=len(y) == 1, squeeze=False)
    axes = axes.reshape(n_rows, n_cols)
    if ylim:
        axes[0][0].set_ylim(*ylim)
    axes = iter(axes.flatten())
    for i, y_ in enumerate(y):
        for j, x_ in enumerate(x):
            ax = next(axes)
            sns.stripplot(data=df, x=x_, y=y_, hue=hue, jitter=True, alpha=0.5, ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            sns.pointplot(data=df, x=x_, y=y_, hue=hue, dodge=True, markers='', capsize=0.1, ax=ax)
            ax.legend_.remove()
    fig.tight_layout()
    lgd = ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.00, 1.025), ncol=1)
    lgd.set_title(hue, prop=dict(size='small'))
    for ax in axes:
        ax.axis('off')
    fig.savefig(plot_file, bbox_extra_artists=(lgd,), bbox_inches='tight')


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
    parser.add_argument('-m', '--dir_pattern', required=True)
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
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    model_dirs = sorted(d for d in glob.glob(args.dir_pattern) if os.path.isdir(d))
    seeds = map(int, args.seeds.split(','))
    folds = map(int, args.folds.split(','))

    df = read_training_output_files(model_dirs, args.data_name, seeds, folds, args.iteration)

    col_name_map = dict(r.split(':') for r in args.rename_col)
    df.rename(columns=col_name_map, inplace=True)

    index_cols = ['model_name', 'iteration']
    f = {col: pd.Series.nunique if col in {'seed', 'fold'} else np.mean \
            for col in df if col not in index_cols}
    agg_df = df.groupby(index_cols).agg(f)
    assert np.all(agg_df['seed'] == len(seeds))
    assert np.all(agg_df['fold'] == len(folds))

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

    test_loss = col_name_map.get('test_y_loss', 'test_y_loss')
    train_loss = col_name_map.get('train_y_loss', 'train_y_loss')

    if 'resolution' in agg_df:
        agg_df['resolution'] = agg_df['resolution'].fillna(0.5)
        test_loss_res = test_loss + '*resolution^3'
        train_loss_res = train_loss + '*resolution^3'
        agg_df[test_loss_res] = agg_df[test_loss]*agg_df['resolution']**3
        agg_df[train_loss_res] = agg_df[train_loss]*agg_df['resolution']**3

    if 'conv_per_level' in agg_df:
        agg_df = agg_df[agg_df['conv_per_level'] > 0]

    if 'width_factor' in agg_df:
        agg_df['width_factor'] = agg_df['width_factor'].fillna(1)
        agg_df = agg_df[agg_df['width_factor'] < 3]

    if 'n_latent' in agg_df:
        agg_df['n_latent'] = agg_df['n_latent'].fillna(0)

    if 'loss_types' in agg_df:
        agg_df['loss_types'] = agg_df['loss_types'].fillna('e')
        agg_df = agg_df[agg_df['loss_types'] == 'e']

    name_fields = [n for n in name_fields if n != args.hue and agg_df[n].nunique() > 1]

    # plot training progress
    line_plot_file = '{}_lines.{}'.format(args.out_prefix, args.plot_ext)
    plot_lines(line_plot_file, agg_df, x='iteration', y=test_loss, hue=args.hue)

    # plot final loss distributions
    final_df = agg_df.reset_index('model_name').loc[args.iteration]
    strip_plot_file = '{}_strips.{}'.format(args.out_prefix, args.plot_ext)
    plot_strips(strip_plot_file, final_df, x=args.x or name_fields,
                y=args.y or [test_loss], hue=args.hue,  n_cols=4)

    print(final_df.sort_values(test_loss).head(5).loc[:, ('model_name', test_loss)])

if __name__ == '__main__':
    main(sys.argv[1:])
