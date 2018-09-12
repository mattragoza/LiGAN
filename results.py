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
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 200)
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('notebook')#, rc={'lines.linewidth': 1.0})
sns.set_palette('Set1')

import models


def plot_lines(plot_file, df, x, y, hue, n_cols=None, height=3, width=3, y_max=None):
    if hue:
        df = df.reset_index().set_index([hue, x])
    elif df.index.name != x:
        df = df.reset_index().set_index(x)
    if n_cols is None:
        n_cols = len(y)
    n_axes = len(y)
    assert n_axes > 0
    n_rows = (n_axes + n_cols-1)//n_cols
    n_cols = min(n_axes, n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(width*n_cols, height*n_rows),
                             sharex=len(x) == 1,
                             sharey=len(y) == 1,
                             squeeze=False)
    iter_axes = iter(axes.flatten())
    extra = []
    for i, y_ in enumerate(y):
        ax = next(iter_axes)
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
        if y_max is not None:
            ax.set_ylim((0.0, y_max))
        handles, labels = ax.get_legend_handles_labels()
    if hue: # add legend
        lgd = fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), ncol=1, frameon=False, borderpad=0.5)
        lgd.set_title(hue, prop=dict(size='small'))
        extra.append(lgd)
    for ax in iter_axes:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(plot_file, bbox_extra_artists=extra, bbox_inches='tight')
    plt.close(fig)


def plot_strips(plot_file, df, x, y, hue, n_cols=None, height=3, width=3, y_max=None):
    df = df.reset_index()
    if n_cols is None:
        n_cols = len(x)
    n_axes = len(x)*len(y)
    assert n_axes > 0
    n_rows = (n_axes + n_cols-1)//n_cols
    n_cols = min(n_axes, n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(width*n_cols, height*n_rows),
                             sharex=len(x) == 1,
                             sharey=len(y) == 1,
                             squeeze=False)
    iter_axes = iter(axes.flatten())
    extra = []
    for y_ in y:
        if y_max is not None: # apply ceiling at y_max
            df[y_].loc[df[y_] > y_max] = y_max
        for x_ in x:
            ax = next(iter_axes)
            sns.stripplot(data=df, x=x_, y=y_, hue=hue, jitter=True, alpha=1.0, ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            #sns.pointplot(data=df, x=x_, y=y_, hue=hue, dodge=True, alpha=1.0, ax=ax)
            if y_max is not None:
                ax.set_ylim((0.0, y_max))
            if hue:
                ax.legend_.remove()
    if hue: # add legend
        lgd = fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), ncol=1, frameon=False, borderpad=0.5)
        lgd.set_title(hue, prop=dict(size='small'))
        extra.append(lgd)
    for ax in iter_axes:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(plot_file, bbox_extra_artists=extra, bbox_inches='tight')
    plt.close(fig)


def read_training_output_files(model_dirs, data_name, seeds, folds, iteration, check):
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
                    if 'base_lr' in file_df:
                        del file_df['base_lr']
                    max_iter = file_df['iteration'].max()
                    assert iteration in file_df['iteration'].unique(), \
                        'No training output for iteration {} ({})'.format(iteration, max_iter)
                    model_dfs.append(file_df)
                except (IOError, pd.io.common.EmptyDataError, AssertionError, KeyError) as e:
                    model_errors[file_] = e
        if not check or not model_errors:
            all_model_dfs.extend(model_dfs)
        else:
            for f, e in model_errors.items():
                print('{}: {}'.format(f, e))
    return pd.concat(all_model_dfs)


def add_data_from_name_parse(df, index, name_format, name):
    name_parse = parse.parse(name_format, name)
    if name_parse is None:
        raise Exception('could not parse {} with format {}'.format(repr(name), repr(name_format)))
    name_fields = sorted(name_parse.named, key=name_parse.spans.get)
    for field in name_fields:
        value = name_parse.named[field]
        df.loc[index, field] = value
    return name_fields


def model_name_fix(model_name, char, idx):
    fields = model_name.split('_')
    for i in idx:
        fields[i] += char
    return '_'.join(fields)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--dir_pattern', default=[], action='append', required=True)
    parser.add_argument('-d', '--data_name', default='lowrmsd')
    parser.add_argument('-s', '--seeds', default='0')
    parser.add_argument('-f', '--folds', default='0,1,2')
    parser.add_argument('-i', '--iteration', default=20000, type=int)
    parser.add_argument('-o', '--out_prefix', default='')
    parser.add_argument('-r', '--rename_col', default=[], action='append')
    parser.add_argument('-x', '--x', default=[], action='append')
    parser.add_argument('-y', '--y', default=[], action='append')
    parser.add_argument('--y_max', type=float)
    parser.add_argument('--hue', default=None)
    parser.add_argument('--n_cols', default=4, type=int)
    parser.add_argument('--masked', default=False, action='store_true')
    parser.add_argument('--plot_lines', default=False, action='store_true')
    parser.add_argument('--plot_strips', default=False, action='store_true')
    parser.add_argument('--plot_ext', default='png')
    parser.add_argument('--aggregate', default=False, action='store_true')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    # read training output files from found model directories
    model_dirs = sorted(d for p in args.dir_pattern for d in glob.glob(p) if os.path.isdir(d))
    seeds = args.seeds.split(',')
    folds = args.folds.split(',')
    df = read_training_output_files(model_dirs, args.data_name, seeds, folds, args.iteration, True)

    # aggregate output values for each model across seeds and folds
    index_cols = ['model_name', 'iteration']
    if args.aggregate:
        f = {col: pd.Series.nunique if col in {'seed', 'fold'} else np.mean \
                for col in df if col not in index_cols}
        agg_df = df.groupby(index_cols).agg(f)
        assert np.all(agg_df['seed'] == len(seeds))
        assert np.all(agg_df['fold'] == len(folds))
    else:
        agg_df = df.set_index(index_cols)

    # add columns from parsing model name fields
    for model_name, model_df in agg_df.groupby(level=0):

        # try to parse it as a GAN
        m = re.match(r'^(.+_)?(.+e(\d+)_.+)_(disc.+)$', model_name)
        name_fields = add_data_from_name_parse(agg_df, model_name, models.SOLVER_NAME_FORMAT, m.group(1))

        v = tuple(int(c) for c in m.group(3))
        if v == (1, 4):
            gen_model_name = model_name_fix(m.group(2), ' ', [-1, -2])
        elif v == (1, 3):            
            gen_model_name = model_name_fix(m.group(2), ' ', [-1, -5])
        else:
            gen_model_name = m.group(2)
        agg_df.loc[model_name, 'gen_model_version'] = str(v)
        name_fields.append('gen_model_version')
        name_fields += add_data_from_name_parse(agg_df, model_name, models.GEN_NAME_FORMATS[v], gen_model_name)

        try:
            disc_model_name = model_name_fix(m.group(4), '_' , [-1])
            name_fields += add_data_from_name_parse(agg_df, model_name, models.DISC_NAME_FORMAT, disc_model_name)
        except:
            disc_model_name = m.group(4)
            name_fields += add_data_from_name_parse(agg_df, model_name, models.OLD_DISC_NAME_FORMAT, disc_model_name)


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

    if args.masked: # treat rmsd_loss as masked loss; adjust for resolution

        rmsd_losses = [l for l in agg_df if 'rmsd_loss' in l]
        for rmsd_loss in rmsd_losses:

            no_rmsd = agg_df[rmsd_loss].isnull()
            agg_df.loc[no_rmsd, rmsd_loss] = agg_df[no_rmsd][rmsd_loss.replace('rmsd_loss', 'loss')]
            agg_df[rmsd_loss] *= agg_df['resolution']**3

    # rename columns if necessary
    agg_df.reset_index(inplace=True)
    col_name_map = {col: col for col in agg_df}
    col_name_map.update(dict(r.split(':') for r in args.rename_col))
    agg_df.rename(columns=col_name_map, inplace=True)
    name_fields = [col_name_map[n] for n in name_fields]

    # by default, don't make separate plots for the hue variable or variables with 1 unique value
    if not args.x:
        args.x = [n for n in name_fields if n != args.hue and agg_df[n].nunique() > 1]

    if args.plot_lines: # plot training progress
        line_plot_file = '{}_lines.{}'.format(args.out_prefix, args.plot_ext)
        plot_lines(line_plot_file, agg_df, x=col_name_map['iteration'], y=args.y, hue=args.hue,
                   n_cols=args.n_cols, y_max=args.y_max)

    final_df = agg_df.set_index(col_name_map['iteration']).loc[args.iteration]
 
    #print(final_df[col_name_map['model_name']].unique())
    print(final_df)
    
    if args.plot_strips: # plot final loss distributions
        strip_plot_file = '{}_strips.{}'.format(args.out_prefix, args.plot_ext)
        plot_strips(strip_plot_file, final_df, x=args.x, y=args.y, hue=args.hue,
                    n_cols=args.n_cols, y_max=args.y_max)

    # display names of best models
    print()
    for y in args.y:
        print(final_df.sort_values(y).loc[:, (col_name_map['model_name'], y)])


if __name__ == '__main__':
    main(sys.argv[1:])
