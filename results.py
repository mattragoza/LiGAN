from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import sys, os, re, glob, argparse, parse, ast, shutil
from collections import defaultdict
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import seaborn as sns
import random

import models
import generate
import experiment


def plot_lines(plot_file, df, x, y, hue, n_cols=None, height=6, width=6, ylim=None, outlier_z=None):

    df = df.reset_index()
    xlim = (df[x].min(), df[x].max())

    if hue:
        df = df.set_index([hue, x])
    elif df.index.name != x:
        df = df.set_index(x)

    if n_cols is None:
        n_cols = len(y)
    n_axes = len(y)
    assert n_axes > 0
    n_rows = (n_axes + n_cols-1)//n_cols
    n_cols = min(n_axes, n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width*n_cols, height*n_rows))
    iter_axes = iter(axes.flatten())

    share_axes = defaultdict(list)
    share_ylim = dict()

    for i, y_ in enumerate(y):

        ax = next(iter_axes)
        ax.set_xlabel(x)
        ax.set_ylabel(y_)

        if hue:
            alpha = 0.5/df.index.get_level_values(hue).nunique()
            for j, _ in df.groupby(level=0):
                mean = df.loc[j][y_].groupby(level=0).mean()
                sem = df.loc[j][y_].groupby(level=0).sem()
                ax.fill_between(mean.index, mean-2*sem, mean+2*sem, alpha=alpha)
            for j, _ in df.groupby(level=0):
                mean = df.loc[j][y_].groupby(level=0).mean()
                ax.plot(mean.index, mean, label=j)
        else:
            mean = df[y_].groupby(level=0).mean()
            sem = df[y_].groupby(level=0).sem()
            ax.fill_between(mean.index, mean-sem, mean+sem, alpha=0.5)
            ax.plot(mean.index, mean)

        handles, labels = ax.get_legend_handles_labels()

        if ylim:
            if len(ylim) > 1:
                ylim_ = ylim[i]
            else:
                ylim_ = ylim[0]
        else:
            ylim_ = ax.get_ylim()

        m = re.match(r'(disc|gen_adv)_(.*)', y_)
        if m and False:
            name = m.group(2)
            share_axes[name].append(ax)
            if name not in share_ylim:
                share_ylim[name] = ylim_
            else:
                ylim_ = share_ylim[name]

        ax.hlines(0, *xlim, linestyle='-', linewidth=1.0)
        if y_.endswith('log_loss'):
            r = -np.log(0.5)
            ax.hlines(r, *xlim, linestyle=':', linewidth=1.0)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim_)

    for n in share_axes:
        share_ylim = (np.inf, -np.inf)
        for ax in share_axes[n]:
            ylim_ = ax.get_ylim()
            share_ylim = (min(ylim_[0], share_ylim[0]),
                          max(ylim_[1], share_ylim[1]))
        for ax in share_axes[n]:
            ax.set_ylim(share_ylim)

    extra = []
    if hue: # add legend
        lgd = fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), ncol=1, frameon=False, borderpad=0.5)
        lgd.set_title(hue, prop=dict(size='small'))
        extra.append(lgd)
    for ax in iter_axes:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(plot_file, bbox_extra_artists=extra, bbox_inches='tight')
    plt.close(fig)


def plot_strips(plot_file, df, x, y, hue, n_cols=None, height=6, width=6, ylim=None, outlier_z=None, \
    violin=False, box=False, jitter=0, alpha=0.5):

    df = df.reset_index()

    if n_cols is None:
        n_cols = len(x)
    n_axes = len(x)*len(y)
    assert n_axes > 0
    n_rows = (n_axes + n_cols-1)//n_cols
    n_cols = min(n_axes, n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width*n_cols, height*n_rows))
    iter_axes = iter(axes.flatten())

    for i, y_ in enumerate(y):

        if outlier_z is not None:
            n_y = len(df[y_])
            df[y_] = replace_outliers(df[y_], np.nan, z=outlier_z)
            print('dropped {} outliers from {}'.format(n_y - len(df[y_]), y_))

        for x_ in x:
            ax = next(iter_axes)

            # plot the means and 95% confidence intervals
            color = 'black' if hue is None else None
            sns.pointplot(data=df, x=x_, y=y_, hue=hue, markers='.', dodge=0.399, color=color, zorder=10, ax=ax)
            #plt.setp(ax.lines, zorder=100)
            #plt.setp(ax.collections, zorder=100)

            if violin: # plot the distributions
                sns.violinplot(data=df, x=x_, y=y_, hue=hue, dodge=True, saturation=1.0, inner=None, ax=ax)
                for c in ax.collections:
                    if isinstance(c, matplotlib.collections.PolyCollection):
                        c.set_alpha(alpha)
                        c.set_edgecolor(None)

            elif box:
                sns.boxplot(data=df, x=x_, y=y_, hue=hue, saturation=1.0, ax=ax)

            else: # plot the individual observations
                sns.stripplot(data=df, x=x_, y=y_, hue=hue, marker='.', dodge=True, jitter=jitter, size=25, alpha=alpha, ax=ax)

            handles, labels = ax.get_legend_handles_labels()
            handles = handles[len(handles)//2:]
            labels = labels[len(labels)//2:]

            if ylim:
                if len(ylim) > 1:
                    ylim_ = ylim[i]
                else:
                    ylim_ = ylim[0]
                ax.set_ylim(ylim_)

            if hue:
                ax.legend_.remove()

    extra = []
    if hue: # add legend
        lgd = fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), ncol=1, frameon=False, borderpad=0.5)
        lgd.set_title(hue, prop=dict(size='small'))
        extra.append(lgd)

    for ax in iter_axes:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(plot_file, bbox_extra_artists=extra, bbox_inches='tight')
    plt.close(fig)


def get_z_bounds(x, z):
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_max = x_mean + z*x_std
    x_min = x_mean - z*x_std
    return x_min, x_max


def replace_outliers(x, value, z=3):
    x_min, x_max = get_z_bounds(x, z)
    return np.where(x > x_max, value, np.where(x < x_min, value, x))


def read_training_output_files(model_dirs, data_name, seeds, folds, iteration, check, gen_metrics):

    all_model_dfs = []
    for model_dir in model_dirs:

        model_dfs = []
        model_name = model_dir.rstrip('/\\')
        model_prefix = os.path.join(model_dir, model_name)
        model_errors = dict()

        for seed in seeds:
            for fold in folds: 
                try:
                    train_file = '{}.{}.{}.{}.training_output'.format(model_prefix, data_name, seed, fold)
                    train_df = pd.read_csv(train_file, sep=' ')
                    train_df['model_name'] = model_name
                    #file_df['data_name'] = data_name #TODO allow multiple data sets
                    train_df['seed'] = seed
                    train_df['fold'] = fold

                    if gen_metrics: #TODO these should be in the train output file
                        gen_file = '{}.{}.{}.{}.gen_metrics'.format(model_prefix, data_name, seed, fold)
                        gen_df = pd.read_csv(gen_file, sep=' ', index_col=0, names=[0]).T
                        for col in gen_df:
                            train_df.loc[:, col] = gen_df[col].values

                    model_dfs.append(train_df)

                except (IOError, pd.io.common.EmptyDataError, AssertionError, KeyError) as e:
                    model_errors[train_file] = e

        if not check or not model_errors:
            all_model_dfs.extend(model_dfs)
        else:
            for f, e in model_errors.items():
                print(e) #'{}: {}'.format(f, e))

    return pd.concat(all_model_dfs)


def read_model_dirs(expt_file):
    with open(expt_file, 'r') as f:
        for line in f:
            yield line.split()[0]


def get_terminal_size():
    with os.popen('stty size') as p:
        return map(int, p.read().split())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Plot results of generative model experiments')
    parser.add_argument('expt_file', help="file specifying experiment directories")
    parser.add_argument('-d', '--data_name', default='lowrmsd', help='base prefix of data used in experiment (default "lowrmsd")')
    parser.add_argument('-s', '--seeds', default='0', help='comma-separated random seeds used in experiment (default 0)')
    parser.add_argument('-f', '--folds', default='0,1,2', help='comma-separated train/test fold numbers used (default 0,1,2)')
    parser.add_argument('-i', '--iteration', type=int, help='iteration for plotting strips')
    parser.add_argument('-o', '--out_prefix', help='common prefix for output files')
    parser.add_argument('-r', '--rename_col', default=[], action='append', help='rename column in results (ex. before_name:after_name)')
    parser.add_argument('-x', '--x', default=[], action='append')
    parser.add_argument('-y', '--y', default=[], action='append')
    parser.add_argument('--hue', default=[], action='append')
    parser.add_argument('--log_y', default=[], action='append')
    parser.add_argument('--outlier_z', default=None, type=float)
    parser.add_argument('--n_cols', default=None, type=int)
    parser.add_argument('--masked', default=False, action='store_true')
    parser.add_argument('--plot_lines', default=False, action='store_true')
    parser.add_argument('--plot_strips', default=False, action='store_true')
    parser.add_argument('--plot_ext', default='png')
    parser.add_argument('--ylim', type=ast.literal_eval, default=[], action='append')
    parser.add_argument('--gen_metrics', default=False, action='store_true')
    parser.add_argument('--violin', default=False, action='store_true')
    parser.add_argument('--test_data')
    parser.add_argument('--avg_seeds', default=False, action='store_true')
    parser.add_argument('--avg_folds', default=False, action='store_true')
    parser.add_argument('--avg_iters', default=1, type=int, help='average over this many consecutive iterations')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    # set up display and plotting options
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_colwidth', 100)
    pd.set_option('display.width', get_terminal_size()[1])
    sns.set_style('whitegrid')
    sns.set_context('talk')
    sns.set_palette('Set1')

    if args.out_prefix is None:
        args.out_prefix = os.path.splitext(args.expt_file)[0]

    seeds = args.seeds.split(',')
    folds = args.folds.split(',')

    # get all training output data from experiment
    expt = experiment.read_expt_file(args.expt_file)
    work_dirs = expt['pbs_file'].apply(os.path.dirname)
    df = read_training_output_files(work_dirs, args.data_name, seeds, folds, args.iteration, True, args.gen_metrics)

    if args.test_data is not None:
        df = df[df['test_data'] == args.test_data]

    group_cols = ['model_name']

    if not args.avg_seeds:
        group_cols.append('seed')

    if not args.avg_folds:
        group_cols.append('fold')

    if args.avg_iters > 1:
        df['iteration'] = args.avg_iters * (df['iteration']//args.avg_iters)
    group_cols.append('iteration')

    agg_df = df.groupby(group_cols).agg({c: np.mean if is_numeric_dtype(df[c]) else lambda x: set(x) \
                                            for c in df if c not in group_cols})
    #assert all(agg_df['seed'] == set(seeds))
    #assert all(agg_df['fold'] == set(folds))

    if not args.y: # use all training output metrics
        args.y = [m for m in agg_df if m not in ['model_name', 'iteration', 'seed', 'fold', 'test_data']]

    # parse model name to get model params and add columns
    for model_name, model_df in agg_df.groupby(level=0):

        try: # try to parse it as a GAN
            model_params = models.parse_gan_name(model_name)
        except AttributeError:
            model_params = models.parse_gen_name(model_name)

        for param, value in model_params.items():
            agg_df.loc[model_name, param] = value

    print('\nAGGREGATED DATA')
    print(agg_df)

    # rename columns if necessary
    agg_df.reset_index(inplace=True)
    col_name_map = {col: col for col in agg_df}
    col_name_map.update(dict(r.split(':') for r in args.rename_col))
    agg_df.rename(columns=col_name_map, inplace=True)
    model_params = {col_name_map[c]: v for c, v in model_params.items()}

    for y in args.log_y: # add log y columns
        log_y = 'log({})'.format(y)
        agg_df[log_y] = agg_df[y].apply(np.log)
        args.y.append(log_y)

    if len(args.hue) > 1: # add column for hue tuple
        hue = '({})'.format(', '.join(args.hue))
        agg_df[hue] = agg_df[args.hue].apply(tuple, axis=1)
    elif len(args.hue) == 1:
        hue = args.hue[0]
    else:
        hue = None

    # by default, don't make plots for the hue variable or variables with 1 unique value
    if not args.x:
        args.x = [c for c in model_params if c != hue and agg_df[c].nunique() > 1]

    if args.plot_lines: # plot training progress
        line_plot_file = '{}_lines.{}'.format(args.out_prefix, args.plot_ext)
        plot_lines(line_plot_file, agg_df, x=col_name_map['iteration'], y=args.y, hue=hue,
                   n_cols=args.n_cols, outlier_z=args.outlier_z, ylim=args.ylim)

    if args.iteration:
        final_df = agg_df.set_index(col_name_map['iteration']).loc[args.iteration]

        print('\nFINAL DATA')
        print(final_df)
        
        if args.plot_strips: # plot final loss distributions
            strip_plot_file = '{}_strips.{}'.format(args.out_prefix, args.plot_ext)
            plot_strips(strip_plot_file, final_df, x=args.x, y=args.y, hue=hue,
                        n_cols=args.n_cols, outlier_z=args.outlier_z, ylim=args.ylim, violin=args.violin)

        # display names of best models
        print('\nBEST MODELS')
        for y in args.y:
            print(final_df.sort_values(y).loc[:, (col_name_map['model_name'], y)]) #.head(5))


if __name__ == '__main__':
    main(sys.argv[1:])
