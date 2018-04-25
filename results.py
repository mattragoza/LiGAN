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
sns.set_context('talk')

import models
import generate


def plot_lines(plot_file, df, x, y, hue, ylim=None):
    df = df.reset_index().set_index([hue, x])
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel('iteration')
    ax.set_ylabel(y)
    if ylim:
        ax.set_ylim(*ylim)
    for i, _ in df.groupby(level=0):
        s = df.loc[i, y]
        ax.plot(s.index, s, label=i, linewidth=1.0)
    fig.tight_layout()
    lgd = ax.legend(loc='upper left', bbox_to_anchor=(1.00, 1.025), ncol=2)
    fig.savefig(plot_file, bbox_extra_artists=(lgd,), bbox_inches='tight')


def plot_strips(plot_file, df, xs, y, hue, ylim=None, n_cols=4):
    df = df.reset_index()
    n_x = len(xs)
    n_rows = (n_x + n_cols-1)//n_cols
    n_cols = min(n_x, n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), sharey=True)
    if ylim:
        axes[0][0].set_ylim(*ylim)
    axes = axes.flatten()
    for i, x in enumerate(xs):
        ax = axes[i]
        sns.stripplot(data=df, x=x, y=y, hue=hue, jitter=True, ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, ncol=1)
    for i in range(i+1, len(axes)):
        axes[i].axis('off')
    fig.tight_layout()
    fig.savefig(plot_file)


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
    return parser.parse_args(argv)


if __name__ == '__main__':
    argv = sys.argv[1:]
    args = parse_args(argv)

    model_dirs = sorted(d for d in glob.glob(args.dir_pattern) if os.path.isdir(d))
    seeds = map(int, args.seeds.split(','))
    folds = map(int, args.folds.split(','))

    model_version = tuple(int(c) for c in re.match(r'^.+e(\d+)_', args.dir_pattern).group(1))
    name_format = models.NAME_FORMATS[model_version]

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
        name_parse = parse.parse(name_format, model_name)
        name_fields = sorted(name_parse.named, key=name_parse.spans.get)
        for field in name_fields:
            agg_df.loc[model_name, field] = name_parse.named[field]

    col = col_name_map.get('test_y_loss', 'test_y_loss')

    # plot training progress
    line_plot_file = '{}_{}_lines.{}'.format(args.out_prefix, col, args.plot_ext)
    plot_lines(line_plot_file, agg_df, x='iteration', y=col, hue='model_name')

    # plot final loss distributions
    agg_df.reset_index('model_name', inplace=True)
    strip_plot_file = '{}_{}_strips.{}'.format(args.out_prefix, col, args.plot_ext)
    plot_strips(strip_plot_file, agg_df.loc[args.iteration], xs=name_fields, y=col, hue='n_filters')
