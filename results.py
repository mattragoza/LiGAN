from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import sys, os, re, glob, argparse, parse, ast
from collections import defaultdict
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 160)
from pandas.api.types import is_numeric_dtype
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import seaborn as sns
import random
sns.set_style('whitegrid')
sns.set_context('talk')
sns.set_palette('Set1')

import models
import generate


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
                ax.fill_between(mean.index, mean-sem, mean+sem, alpha=alpha)
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


def plot_strips(plot_file, df, x, y, hue, n_cols=None, height=6, width=6, ylim=None, outlier_z=None, violin=False):

    df = df.reset_index()

    if n_cols is None:
        n_cols = len(x) - (hue in x)
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
            sns.pointplot(data=df, x=x_, y=y_, hue=hue, dodge=0.525, markers='.', ax=ax)
            alpha = 0.25

            if violin:
                sns.violinplot(data=df, x=x_, y=y_, hue=hue, dodge=True, inner=None, saturation=1.0, linewidth=0.0, ax=ax)
                for c in ax.collections:
                    if isinstance(c, matplotlib.collections.PolyCollection):
                        c.set_alpha(alpha)
            else:
                sns.stripplot(data=df, x=x_, y=y_, hue=hue, dodge=0.525, jitter=0.2, size=5, alpha=alpha, ax=ax)

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
                    train_df['iteration'] = train_df['iteration'].astype(int)
                    if 'base_lr' in train_df:
                        del train_df['base_lr']
                    max_iter = train_df['iteration'].max()
                    #assert iteration in train_df['iteration'].unique(), \
                    #    'No training output for iteration {} ({})'.format(iteration, max_iter)

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
                print('{}: {}'.format(f, e))

    return pd.concat(all_model_dfs)


def add_data_from_name_parse(df, index, prefix, name_format, name):
    name_parse = parse.parse(name_format, name)
    if name_parse is None:
        raise Exception('could not parse {} with format {}'.format(repr(name), repr(name_format)))
    name_fields = []
    for field in sorted(name_parse.named, key=name_parse.spans.get):
        value = name_parse.named[field]
        if isinstance(value, str):
            value = value.rstrip()
            if not value:
                value = ' '
        if prefix:
            field = '{}_{}'.format(prefix, field)
        df.loc[index, field] = value
        name_fields.append(field)
    return name_fields


def fix_name(name, char, idx):
    '''
    Split name into fields by underscore, append char to
    fields at each index in idx, then rejoin by underscore.
    '''
    fields = name.split('_')
    for i in idx:
        fields[i] += char
    return '_'.join(fields)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Plot results of generative model experiments')
    parser.add_argument('-m', '--dir_pattern', default=[], action='append', help='glob pattern of experiment directories')
    parser.add_argument('-d', '--data_name', default='lowrmsd', help='base prefix of data used in experiment (default "lowrmsd")')
    parser.add_argument('-s', '--seeds', default='0', help='comma-separated random seeds used in experiment (default 0)')
    parser.add_argument('-f', '--folds', default='0,1,2', help='comma-separated train/test fold numbers used (default 0,1,2)')
    parser.add_argument('-i', '--iteration', default=20000, type=int, help='max train iteration for results')
    parser.add_argument('-o', '--out_prefix', help='common prefix for output files')
    parser.add_argument('-r', '--rename_col', default=[], action='append', help='rename column in results (ex. before_name:after_name)')
    parser.add_argument('-x', '--x', default=[], action='append')
    parser.add_argument('-y', '--y', default=[], action='append')
    parser.add_argument('--log_y', default=[], action='append')
    parser.add_argument('--outlier_z', default=None, type=float)
    parser.add_argument('--hue', default=[], action='append')
    parser.add_argument('--n_cols', default=None, type=int)
    parser.add_argument('--masked', default=False, action='store_true')
    parser.add_argument('--plot_lines', default=False, action='store_true')
    parser.add_argument('--plot_strips', default=False, action='store_true')
    parser.add_argument('--plot_ext', default='png')
    parser.add_argument('--aggregate', default=False, action='store_true')
    parser.add_argument('--test_data')
    parser.add_argument('--ylim', default=[], action='append')
    parser.add_argument('--gen_metrics', default=False, action='store_true')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    args.ylim = [ast.literal_eval(yl) for yl in args.ylim]
    seeds = args.seeds.split(',')
    folds = args.folds.split(',')

    if args.dir_pattern: # read training output files from found model directories
        model_dirs = sorted(d for p in args.dir_pattern for d in glob.glob(p) if os.path.isdir(d))
        df = read_training_output_files(model_dirs, args.data_name, seeds, folds, args.iteration, True, args.gen_metrics)
        results_file = '{}.results'.format(args.out_prefix)
        df.to_csv(results_file)
    else:
        results_file = '{}.results'.format(args.out_prefix)
        df = pd.read_csv(results_file)

    if args.test_data is not None:
        df = df[df['test_data'] == args.test_data]

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

        print(model_name)

        # try to parse it as a GAN
        m = re.match(r'^(.+)_([^_]+e(\d+).+)_(d([^_]+).*)$', model_name)
        if m:
            solver_name = m.group(1)
            gen_model_name = m.group(2)
            disc_model_name = m.group(4)

            try:
                solver_name = fix_name(solver_name, ' ', [3])
                name_fields = add_data_from_name_parse(agg_df, model_name, '', models.SOLVER_NAME_FORMAT, solver_name)
            except IndexError:
                agg_df.loc[model_name, 'solver_name'] = solver_name
                agg_df.loc[model_name, 'train_options'] = re.match(r'^(.*)gan$', solver_name).group(1)
                name_fields = ['solver_name', 'train_options']

            gen_v = tuple(int(c) for c in m.group(3))
            if gen_v == (1, 4):
                gen_model_name = fix_name(gen_model_name, ' ', [-1, -2])
            elif gen_v == (1, 3):            
                gen_model_name = fix_name(gen_model_name, ' ', [-1, -5])
            agg_df.loc[model_name, 'gen_model_version'] = str(gen_v)
            name_fields.append('gen_model_version')
            name_fields += add_data_from_name_parse(agg_df, model_name, 'gen', models.GEN_NAME_FORMATS[gen_v], gen_model_name)

            try:
                disc_v = tuple(int(c) for c in m.group(5))
            except ValueError:
                disc_v = (0, 1)
            agg_df.loc[model_name, 'disc_model_version'] = str(disc_v)
            name_fields.append('disc_model_version')
            try:
                disc_model_name = fix_name(disc_model_name, ' ', [-4])
                name_fields += add_data_from_name_parse(agg_df, model_name, 'disc', models.DISC_NAME_FORMATS[disc_v], disc_model_name)
            except IndexError:
                if disc_model_name == 'disc':
                    agg_df.loc[model_name, 'disc_conv_per_level'] = 1
                elif disc_model_name == 'disc2':
                    agg_df.loc[model_name, 'disc_conv_per_level'] = 2
                else:
                    raise NameError(disc_model_name)
                name_fields.append('disc_conv_per_level')
        else:
            m = re.match(r'[^_]+e(\d+).+', model_name)

            gen_v = tuple(int(c) for c in m.group(1))
            if gen_v == (1, 4):
                gen_model_name = fix_name(m.group(), ' ', [-1, -2])
            elif gen_v == (1, 3):            
                gen_model_name = fix_name(m.group(), ' ', [-1, -5])
            else:
                gen_model_name = m.group()
            agg_df.loc[model_name, 'gen_model_version'] = str(gen_v)
            name_fields = ['gen_model_version']
            name_fields += add_data_from_name_parse(agg_df, model_name, 'gen', models.GEN_NAME_FORMATS[gen_v], gen_model_name)

    # fill in default values
    if 'resolution' in agg_df:
        agg_df.loc[agg_df['resolution'].isnull()] = 0.5

    if 'n_latent' in agg_df:
        agg_df.loc[agg_df['n_latent'].isnull()] = 0

    has_gan = ~agg_df['disc_model_version'].isnull()

    if 'disc_loss_types' in agg_df:
        agg_df.loc[has_gan & agg_df['disc_loss_types'].isnull()] = 'x'

    if 'disc_n_levels' in agg_df:
        agg_df.loc[has_gan & agg_df['disc_n_levels'].isnull()] = 3

    if 'disc_conv_per_level' in agg_df:
        agg_df.loc[has_gan & agg_df['disc_conv_per_level'].isnull()] = 1

    if 'disc_n_filters' in agg_df:
        agg_df.loc[has_gan & agg_df['disc_n_filters'].isnull()] = 16

    if 'disc_width_factor' in agg_df:
        agg_df.loc[has_gan & agg_df['disc_width_factor'].isnull()] = 2

    if args.masked: # treat rmsd_loss as masked loss; adjust for resolution

        rmsd_losses = [l for l in agg_df if 'rmsd_loss' in l]
        for rmsd_loss in rmsd_losses:

            no_rmsd = agg_df[rmsd_loss].isnull()
            agg_df.loc[no_rmsd, rmsd_loss] = agg_df[no_rmsd][rmsd_loss.replace('rmsd_loss', 'loss')]
            agg_df[rmsd_loss] *= agg_df['resolution']**3

    #agg_df['gen_fit_L2_loss'] = agg_df.apply(lambda x: x['gen_fit_L2_loss'] or x['gen_L2_loss'], axis=1)

    # rename columns if necessary
    agg_df.reset_index(inplace=True)
    col_name_map = {col: col for col in agg_df}
    col_name_map.update(dict(r.split(':') for r in args.rename_col))
    agg_df.rename(columns=col_name_map, inplace=True)
    name_fields = [col_name_map[n] for n in name_fields]

    for y in args.log_y:
        log_y = 'log({})'.format(y)
        agg_df[log_y] = agg_df[y].apply(np.log)
        args.y.append(log_y)

    if len(args.hue) > 1:
        hue = '({})'.format(', '.join(args.hue))
        agg_df[hue] = agg_df[args.hue].apply(tuple, axis=1)
        print(agg_df[hue])

    elif len(args.hue) == 1:
        hue = args.hue[0]

    else:
        hue = None

    # by default, don't make separate plots for the hue variable or variables with 1 unique value
    if not args.x:
        args.x = [n for n in name_fields if n != hue and agg_df[n].nunique() > 1]

    if args.plot_lines: # plot training progress
        line_plot_file = '{}_lines.{}'.format(args.out_prefix, args.plot_ext)
        plot_lines(line_plot_file, agg_df, x=col_name_map['iteration'], y=args.y, hue=hue,
                   n_cols=args.n_cols, outlier_z=args.outlier_z, ylim=args.ylim)

    final_df = agg_df.set_index(col_name_map['iteration']).loc[args.iteration]
    print('\nfinal data')
    print(final_df)
    
    if args.plot_strips: # plot final loss distributions
        strip_plot_file = '{}_strips.{}'.format(args.out_prefix, args.plot_ext)
        plot_strips(strip_plot_file, final_df, x=args.x, y=args.y, hue=hue,
                    n_cols=args.n_cols, outlier_z=args.outlier_z, ylim=args.ylim)

    # display names of best models
    print('\nbest models')
    for y in args.y:
        print(final_df.sort_values(y).loc[:, (col_name_map['model_name'], y)]) #.head(5))


if __name__ == '__main__':
    main(sys.argv[1:])
