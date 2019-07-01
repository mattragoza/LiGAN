from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import sys, os, re, glob, argparse, parse, ast, shutil
from collections import defaultdict
import numpy as np
import scipy.stats as stats
np.random.seed(0)
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import io

import params
import models
import generate
import experiment


def get_terminal_size():
    with os.popen('stty size') as p:
        return [int(i) for i in p.read().split()]


def annotate_pearson_r(x, y, **kwargs):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("$\\rho = {:.2f}$".format(r), xy=(.5, .8), xycoords='axes fraction', ha='center')


def my_dist_plot(a, **kwargs):
    if 'label' in kwargs:
        kwargs['label'] = str(kwargs['label'])
    return sns.distplot(a, **kwargs)


def plot_corr(plot_file, df, x, y, height=4, width=4, **kwargs):

    df = df.reset_index()
    g = sns.PairGrid(df, x_vars=x, y_vars=y, size=height, aspect=width/float(height), **kwargs)
    g.map_diag(my_dist_plot, kde=False)
    g.map_offdiag(plt.scatter) #, s=1.0, alpha=0.05)
    #g.map_upper(sns.kdeplot)
    g.map_offdiag(annotate_pearson_r)
    fig = g.fig
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    fig.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)


def plot_lines(plot_file, df, x, y, hue, n_cols=None, height=6, width=6, ylim=None, outlier_z=None, lgd_title=True):

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

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width*n_cols, height*n_rows), squeeze=False)
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
        if y_.endswith('log_loss') or 'GAN' in y_:
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
        lgd = fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.975, 0.9), ncol=1, frameon=False, borderpad=0.5)
        if lgd_title:
            lgd.set_title(hue, prop=dict(size='small'))
        extra.append(lgd)
    for ax in iter_axes:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(str(plot_file), format='png', bbox_extra_artists=extra, bbox_inches='tight')
    plt.close(fig)


def plot_dist(plot_file, df, x, hue, n_cols=None, height=6, width=6):

    df = df.reset_index()
    if n_cols is None:
        n_cols = len(x)

    n_axes = len(x)
    assert n_axes > 0
    n_rows = (n_axes + n_cols-1)//n_cols
    n_cols = min(n_axes, n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width*n_cols, height*n_rows), squeeze=False)
    iter_axes = iter(axes.flatten())

    for x_ in x:
        ax = next(iter_axes)
        sns.distplot(df[x_], norm_hist=True, ax=ax)

    for ax in iter_axes:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)



def plot_strips(plot_file, df, x, y, hue, n_cols=None, height=6, width=6, ylim=None,
                strip=False, violin=False, box=False, grouped=False,
                jitter=0, alpha=0.5, outlier_z=None):
    df = df.reset_index()

    if n_cols is None:
        n_cols = len(x)
    n_axes = len(x)*len(y)
    assert n_axes > 0
    n_rows = (n_axes + n_cols-1)//n_cols
    n_cols = min(n_axes, n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width*n_cols, height*n_rows), squeeze=False)
    iter_axes = iter(axes.flatten())

    for i, y_ in enumerate(y):

        for x_ in x:
            ax = next(iter_axes)

            if grouped:
                hue = '({})'.format(', '.join([c for c in x if c not in {x_, 'memory'}]))

            #print('CALLING POINT PLOT')
            #print('  x = {}'.format(x_))
            #print('  y = {}'.format(y_))
            #print('  hue = {}'.format(hue))

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

            if box:
                sns.boxplot(data=df, x=x_, y=y_, hue=hue, saturation=1.0, ax=ax)

            if strip: # plot the individual observations
                sns.stripplot(data=df, x=x_, y=y_, hue=hue, marker='.', dodge=True, jitter=jitter, size=25, alpha=alpha, ax=ax)

            n_plot_types = 1 + strip + violin + box
            handles, labels = ax.get_legend_handles_labels()
            handles = handles[len(handles)//n_plot_types:]
            labels = labels[len(labels)//n_plot_types:]

            xlim = ax.get_xlim()
            ax.hlines(0, *xlim, linestyle='-', linewidth=1.0)
            if y_.endswith('log_loss'):
                r = -np.log(0.5)
                ax.hlines(r, *xlim, linestyle=':', linewidth=1.0)

            if ylim:
                if len(ylim) > 1:
                    ylim_ = ylim[i]
                else:
                    ylim_ = ylim[0]
            else:
            	ylim_ = ax.get_ylim()

            ax.set_ylim(ylim_)

            if hue:
                ax.legend_.remove()

    extra = []
    if hue and not grouped: # add legend
        lgd = fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), ncol=1, frameon=False, borderpad=0.5)
        lgd.set_title(hue, prop=dict(size='small'))
        extra.append(lgd)

    for ax in iter_axes:
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(plot_file, bbox_extra_artists=extra, bbox_inches='tight')
    plt.close(fig)


def get_z_bounds(x, z=3):
    m, s = np.nanmean(x), np.nanstd(x)
    return m - z*s, m + z*s


def get_iqr_bounds(x, k=1.5):
    q1, q3 = np.nanquantile(x, [0.25, 0.75])
    iqr = q3 - q1
    return q1 - k*iqr, q3 + k*iqr


def remove_outliers(x, bounds):   
    lower_bound, upper_bound = bounds
    outlier = (x < lower_bound) | (x > upper_bound)
    return np.where(outlier, np.nan, x)


def get_y_key(col):
    return col.endswith('loss'), col.startswith('lig'), col


def get_x_key(col):
    return col.startswith('gen'), col.startswith('disc'), col


def read_training_output_files(job_files, data_name, seeds, folds, iteration, check, gen_metrics):

    all_model_dfs = []
    for job_file in job_files:

        model_dfs = []
        model_dir = os.path.dirname(job_file)
        model_name = os.path.split(model_dir.rstrip('/\\'))[-1]
        model_prefix = os.path.join(model_dir, model_name)
        model_errors = dict()

        for seed in seeds:
            for fold in folds: 
                try:
                    if 'e11' in model_name:
                        train_file = '{}.{}.{}.training_output'.format(model_prefix, seed, fold)
                    else:
                        train_file = '{}.{}.{}.{}.training_output'.format(model_prefix, data_name, seed, fold)
                    train_df = pd.read_csv(train_file, sep=' ')
                    train_df['job_file'] = job_file
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
                print('{}: {}'.format(f, str(e).replace(' ' + f, '')))

    print(len(all_model_dfs))
    return pd.concat(all_model_dfs)


def read_model_dirs(expt_file):
    with open(expt_file, 'r') as f:
        for line in f:
            yield line.split()[0]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Plot results of generative model experiments')
    parser.add_argument('job_script', nargs='+', help="submission scripts for jobs to plot reuslts for")
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
    parser.add_argument('--outlier_z', default=np.inf, type=float, help='remove outliers beyond this number of SDs from the mean')
    parser.add_argument('--outlier_iqr', default=np.inf, type=float, help='remove outliers beyond this multiple of the IQR from [Q1, Q3]')
    parser.add_argument('--n_cols', default=None, type=int)
    parser.add_argument('--masked', default=False, action='store_true')
    parser.add_argument('--plot_lines', default=False, action='store_true')
    parser.add_argument('--plot_strips', default=False, action='store_true')
    parser.add_argument('--plot_corr', default=False, action='store_true')
    parser.add_argument('--plot_ext', default='png')
    parser.add_argument('--ylim', type=ast.literal_eval, default=[], action='append')
    parser.add_argument('--gen_metrics', default=False, action='store_true')
    parser.add_argument('--test_data')
    parser.add_argument('--avg_seeds', default=False, action='store_true')
    parser.add_argument('--avg_folds', default=False, action='store_true')
    parser.add_argument('--avg_iters', default=1, type=int, help='average over this many consecutive iterations')
    parser.add_argument('--scaffold', default=False, action='store_true')
    parser.add_argument('--strip', default=False, action='store_true')
    parser.add_argument('--violin', default=False, action='store_true')
    parser.add_argument('--box', default=False, action='store_true')
    parser.add_argument('--grouped', default=False, action='store_true')
    return parser.parse_args(argv)


def aggregate_data(df, group_cols):
    f = {c: np.mean if is_numeric_dtype(df[c]) else lambda x: set(x) for c in df if c not in group_cols}
    return df.groupby(group_cols).agg(f)


def prepend_keys(dct, prefix):
    return type(dct)((prefix+k, v) for (k, v) in dct.items())


def add_param_columns(df, scaffold=False):
    for job_file, job_df in df.groupby(level=0):
        if True: # try to parse it as a GAN
            job_params = params.read_params(job_file, line_start='# ')

            data_model_file = os.path.join(os.path.dirname(job_file), job_params['model_dir'], job_params['data_model_name'] + '.model')
            data_model_params = params.read_params(data_model_file, line_start='# ')
            job_params.update(prepend_keys(data_model_params, prefix='data_model_params.'))

            gen_model_file = os.path.join(os.path.dirname(job_file), job_params['model_dir'], job_params['gen_model_name'] + '.model')
            gen_model_params = params.read_params(gen_model_file, line_start='# ')
            if scaffold:
                for k, v in models.scaffold_model(gen_model_file).items():
                    df.loc[job_file, 'gen_'+k] = v
            job_params.update(prepend_keys(gen_model_params, prefix='gen_model_params.'))

            disc_model_file = os.path.join(os.path.dirname(job_file), job_params['model_dir'], job_params['disc_model_name'] + '.model')
            disc_model_params = params.read_params(disc_model_file, line_start='# ')
            if scaffold:
                for k, v in models.scaffold_model(disc_model_file).items():
                    df.loc[job_file, 'disc_'+k] = v
            job_params.update(prepend_keys(disc_model_params, prefix='disc_model_params.'))

            del job_params['seed'] # these already exist
            del job_params['fold']
        try:
            pass
        except AttributeError:
            try: # try to parse model name as a GAN
                model_params = models.parse_gan_name(model_name)
            except AttributeError:
                model_params = models.parse_gen_name(model_name)
        for param, value in job_params.items():
            df.loc[job_file, param] = value
    return job_params


def add_group_column(df, group_cols):
    group = '({})'.format(', '.join(group_cols))
    print('adding group column {}'.format(group))
    df[group] = df[group_cols].apply(lambda x: str(tuple(x)), axis=1)
    return group


def main(argv):
    args = parse_args(argv)

    # set up display and plotting options
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_colwidth', 100)
    pd.set_option('display.width', get_terminal_size()[1])
    sns.set_style('whitegrid')
    sns.set_context('poster')
    #sns.set_palette('Set1')

    if args.out_prefix is None:
        args.out_prefix = os.path.splitext(args.expt_file)[0]

    seeds = args.seeds.split(',')
    folds = args.folds.split(',')

    # get all training output data from experiment
    job_files = args.job_script
    df = read_training_output_files(job_files, args.data_name, seeds, folds, args.iteration, True, args.gen_metrics)

    if args.test_data is not None:
        df = df[df['test_data'] == args.test_data]

    group_cols = ['job_file', 'model_name']

    if not args.avg_seeds:
        group_cols.append('seed')

    if not args.avg_folds:
        group_cols.append('fold')

    if args.avg_iters > 1:
        df['iteration'] = args.avg_iters * (df['iteration']//args.avg_iters)
    group_cols.append('iteration')

    exclude_cols = [
        'job_file',
        'model_name',
        'gen_model_name',
        'disc_model_name',
        'iteration',
        'seed',
        'fold',
        'test_data'
    ]

    agg_df = aggregate_data(df, group_cols)
    #assert all(agg_df['seed'] == set(seeds))
    #assert all(agg_df['fold'] == set(folds))

    if not args.y: # use all training output metrics
        args.y = [m for m in agg_df if m not in exclude_cols]
        if args.scaffold:
            args.y += [p+x for p in ['gen_', 'disc_'] for x in ['n_params', 'n_activs', 'size', 'min_width']]
        args.y = sorted(args.y, key=get_y_key, reverse=True)

    # parse model name to get model params and add columns
    job_params = add_param_columns(agg_df, scaffold=args.scaffold)

    print('\nAGGREGATED DATA')
    print(agg_df)

    # rename columns if necessary
    agg_df.reset_index(inplace=True)
    col_name_map = {col: col for col in agg_df}
    col_name_map.update(dict(r.split(':') for r in args.rename_col))
    agg_df.rename(columns=col_name_map, inplace=True)
    job_params = {col_name_map[c]: v for c, v in job_params.items()}

    for y in args.log_y: # add log y columns
        log_y = 'log({})'.format(y)
        agg_df[log_y] = agg_df[y].apply(np.log)
        args.y.append(log_y)

    if len(args.hue) > 1: # add column for hue tuple
        hue = add_group_column(agg_df, args.hue)
    elif len(args.hue) == 1:
        hue = args.hue[0]
    else:
        hue = None

    # by default, don't make plots for the hue variable or variables with 1 unique value
    if not args.x:
        args.x = [c for c in job_params if c not in exclude_cols and agg_df[c].nunique() > 1]
        args.x = sorted(args.x, key=get_x_key, reverse=True)

    if args.grouped: # add "all but one" group columns
        for col in args.x:
            all_but_col = [c for c in args.x if c not in {col, 'memory'}]
            add_group_column(agg_df, all_but_col)

    agg_df.to_csv('{}_agg_data.csv'.format(args.out_prefix))

    for y in args.y:
        z_bounds = get_z_bounds(agg_df[y], args.outlier_z)
        iqr_bounds = get_iqr_bounds(agg_df[y], args.outlier_iqr)
        print(y, z_bounds, iqr_bounds)
        agg_df[y] = remove_outliers(agg_df[y], z_bounds)
        agg_df[y] = remove_outliers(agg_df[y], iqr_bounds)

    if args.plot_lines: # plot training progress

        line_plot_file = '{}_lines.{}'.format(args.out_prefix, args.plot_ext)
        plot_lines(line_plot_file, agg_df, x=col_name_map['iteration'], y=args.y, hue=None,
                   n_cols=args.n_cols, outlier_z=args.outlier_z, ylim=args.ylim)

        for hue in args.x + ['model_name']:
            line_plot_file = '{}_lines_{}.{}'.format(args.out_prefix, hue, args.plot_ext)
            plot_lines(line_plot_file, agg_df, x=col_name_map['iteration'], y=args.y, hue=hue,
                       n_cols=args.n_cols, outlier_z=args.outlier_z, ylim=args.ylim)

    if args.iteration:
        final_df = agg_df.set_index(col_name_map['iteration']).loc[args.iteration]

        print('\nFINAL DATA')
        print(final_df)

        # display names of best models
        print('\nBEST MODELS')
        for y in args.y:
            print(final_df.sort_values(y).loc[:, (col_name_map['model_name'], y)]) #.head(5))

        if args.plot_strips: # plot final loss distributions

            strip_plot_file = '{}_strips.{}'.format(args.out_prefix, args.plot_ext)
            plot_strips(strip_plot_file, final_df, x=args.x, y=args.y, hue=None,
                        n_cols=args.n_cols, outlier_z=args.outlier_z, ylim=args.ylim)

            if args.grouped:
                strip_plot_file = '{}_grouped_strips.{}'.format(args.out_prefix, args.plot_ext)
                plot_strips(strip_plot_file, final_df, x=args.x, y=args.y, hue=None, grouped=True,
                            n_cols=args.n_cols, outlier_z=args.outlier_z, ylim=args.ylim)

            for hue in args.x + ['model_name']:
                strip_plot_file = '{}_strips_{}.{}'.format(args.out_prefix, hue, args.plot_ext)
                plot_strips(strip_plot_file, final_df, x=args.x, y=args.y, hue=hue,
                            n_cols=args.n_cols, outlier_z=args.outlier_z, ylim=args.ylim)

        if args.plot_corr:

            corr_y = [y for y in args.y if final_df[y].nunique() > 1]

            corr_plot_file = '{}_corr.{}'.format(args.out_prefix, args.plot_ext)
            plot_corr(corr_plot_file, final_df, x=corr_y, y=corr_y)

            for hue in args.x + ['model_name']:
                corr_plot_file = '{}_corr_{}.{}'.format(args.out_prefix, hue, args.plot_ext)
                plot_corr(corr_plot_file, final_df, x=corr_y, y=corr_y, hue=hue)


if __name__ == '__main__':
    main(sys.argv[1:])
