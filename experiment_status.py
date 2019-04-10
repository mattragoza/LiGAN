from __future__ import print_function

import sys, os, re, glob, argparse, string
import datetime as dt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import torque_util


def read_file(file_):
    with open(file_, 'r') as f:
        return f.read()


def parse_pbs_file(pbs_file):
    buf = read_file(pbs_file)
    job_name = re.search(r'#PBS -N (.+)', buf).group(1)
    try:
        data_name = re.search(r'DATA="([^\s]+)"', buf).group(1)
    except AttributeError:
        try:
            data_name = re.search(r'DATA_NAME="([^\s]+)"', buf).group(1)
        except AttributeError:
            data_name = re.search(r'.*train\.py.*-p ([^\s]+)', buf).group(1)
    return job_name, os.path.basename(data_name)


def parse_output_file(out_file, metric):
    df = pd.read_csv(out_file, sep=' ')
    max_iter = df['iteration'].max()
    value = np.nan
    if metric == 'L2':
        for m in ['gen_L2_loss', 'test_y_loss', 'test_loss']:
            if m in df:
                value = df[df['iteration'] == max_iter][m].mean()
                break
    else:
        value = df[df['iteration'] == max_iter][metric].mean()
    return max_iter, value # TODO determine test/train


def parse_stderr_file(stderr_file):
    buf = read_file(stderr_file)
    m = re.search(r'(([^\s]+(Error|Exception|Interrupt|Exit).*)|Segmentation fault|(Check failed.*))', buf)
    return m.group(0)


def submit_incomplete_jobs(job):

    if job['job_state'] not in ['Q', 'R'] and job['error'] is not None:
        torque_util.submit_job((job['pbs_file'], job['array_idx']))


def update_job_fields(job, qstat, metric):
    '''
    Update experiment status by parsing qstat command, training
    output, and stderr files. Expects pbs_file and array_idx fields
    to be present and accurate, while job_id is resolved by first
    checking latest in queue, then latest in work_dir.
    '''
    work_dir = os.path.dirname(job['pbs_file'])
    job_name, data_name = parse_pbs_file(job['pbs_file'])
    full_job_name = '{}-{}'.format(job_name, job['array_idx'])

    try: # find latest job_id in queue
        job_qstat = qstat[qstat['Job_Name'] == full_job_name].iloc[-1]
        job['job_id'] = job_qstat.name
        job['job_state'] = job_qstat['job_state']

    except IndexError: # not in queue, find in work_dir
        job['job_state'] = np.nan
        try:
            job['job_id'] = find_job_ids(job['pbs_file'], job['array_idx'])[-1]

        except IndexError: # could not find job_id
            pass

    # find latest training output file
    seed, fold = job['array_idx']//4, job['array_idx']%4
    if fold == 3:
        fold = 'all'
    out_name = '{}.{}.{}.{}'.format(job_name, data_name, seed, fold)

    if job['job_state'] == 'R': # if running, check scr_dir
        job['node_id'] = job_qstat['gpus_reserved'].split('.')[0]
        scr_dir = '/net/{}/scr/{}'.format(job['node_id'], job['job_id'])
        out_file = os.path.join(scr_dir, '{}.training_output'.format(out_name))

    else: # otherwise check work_dir
        if job['job_state'] == 'C':
            job['node_id'] = job_qstat['gpus_reserved'].split('.')[0]
        out_file = os.path.join(work_dir, '{}.training_output'.format(out_name))
        if not os.path.isfile(out_file):
            out_name = '{}.{}.{}'.format(job_name, seed, fold)
            out_file = os.path.join(work_dir, '{}.training_output'.format(out_name))

    try: # get iteration from training output file
        job['time_modified'] = dt.datetime.fromtimestamp(os.path.getmtime(out_file))
        job['iteration'], job['metric'] = parse_output_file(out_file, metric)

    except ValueError: # training output file is empty
        job['iteration'] = None

    except (OSError, IOError): # couldn't find or read training output file
        pass

    try: # get error type from the stderr_file
        stderr_file = os.path.join(work_dir, '{}.e'.format(out_name))
        if not os.path.isfile(stderr_file):
            job_num, _ = parse_job_id(job['job_id'])
            stderr_file = os.path.join(work_dir, '{}.e{}-{}'.format(job_name, job_num, job['array_idx']))
        job['error'] = parse_stderr_file(stderr_file)

    except AttributeError:
        job['error'] = None

    except TypeError: # job_id is nan = job was never submitted
        pass

    except IOError: # couldn't find or read stderr_file = job not finished or stderr not copied
        pass

    return job


def fix_job_fields(job):
    '''
    Attempt to reconcile different experiment file formats. At minimum
    the correct pbs_file and array_idx are required to update job status.
    '''
    if not is_pbs_file(job['pbs_file']):
        try: # it's the job_id
            job_num, job['array_idx'] = parse_job_id(job['pbs_file'])
            job['job_id'] = job['pbs_file']
            job['pbs_file'] = find_pbs_file(job_num)
        except AttributeError: # it's the work_dir
            job['pbs_file'] = glob.glob(os.path.join(job['pbs_file'], '*.pbs'))[-1]

    if not isinstance(job['array_idx'], int):
        try: # it's the job_id
            job_num, job['array_idx'] = parse_job_id(job['array_idx'])
            job['job_id'] = job['array_idx']
        except TypeError: # it's null
            job['array_idx'] = 3 # TODO this is a hack

    return job


def is_pbs_file(path):
    '''
    Test whether a path is a file with .pbs extension.
    '''
    return path.endswith('.pbs')


def find_pbs_file(job_num):
    '''
    Find the pbs_file associated with a job_num.
    '''
    raise NotImplementedError('TODO')


def find_job_ids(pbs_file, array_idx):
    '''
    Find job_ids associated with a pbs_file and array_idx, sorted by
    job_num, by finding stdout and stderr files in work_dir.
    '''
    work_dir = os.path.dirname(pbs_file)
    job_name = parse_pbs_file(pbs_file)[0]
    job_nums = set()
    for f in os.listdir(work_dir):
        m = re.match(r'{}\.[oe](\d+)-{}'.format(job_name, array_idx), f)
        if m:
            job_nums.add(int(m.group(1)))
    return [format_job_id(j, array_idx) for j in sorted(job_nums)]


def format_job_id(job_num, array_idx):
    return '{:d}[{:d}].n198.dcb.private.net'.format(job_num, array_idx)


def parse_job_id(job_id):
    m = re.match(r'^(\d+)\[(\d+)\]\.n198\.dcb\.private\.net$', job_id)
    return int(m.group(1)), int(m.group(2))


def write_expt_file(expt_file, df):
    columns = ['pbs_file', 'array_idx', 'job_id', 'node_id', 'job_state', 'iteration', 'time_modified', 'error']
    df.to_csv(expt_file, sep=' ', index=False, header=False, columns=columns)


def read_expt_file(expt_file):
    names = ['pbs_file', 'array_idx', 'job_id', 'node_id', 'job_state', 'iteration', 'time_modified', 'error']
    return pd.read_csv(expt_file, sep=' ', names=names).apply(fix_job_fields, axis=1)


def parse_args(argv):
    parser = argparse.ArgumentParser(description='check status of GAN experiment')
    parser.add_argument('expt_file', help='file specifying experiment pbs scripts and job IDs')
    parser.add_argument('-u', '--update', default=False, action='store_true')
    parser.add_argument('-s', '--submit', default=False, action='store_true', help='submit jobs that aren\'t in queue or have errors')
    parser.add_argument('-o', '--out_file', help='output file to write updated experiment status')
    parser.add_argument('-m', '--metric', default='L2')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    expt = read_expt_file(args.expt_file)

    if args.update:
        qstat = torque_util.get_qstat_data()
        expt = expt.apply(update_job_fields, axis=1, qstat=qstat, metric=args.metric)

    if args.submit:
        expt.apply(submit_incomplete_jobs, axis=1)
        qstat = torque_util.get_qstat_data()
        expt = expt.apply(update_job_fields, axis=1, qstat=qstat, metric=args.metric)
        qstat = torque_util.get_qstat_data()

    if not args.out_file:
        args.out_file = args.expt_file

    write_expt_file(args.out_file, expt)

    pd.set_option('display.max_rows', 2000)
    pd.set_option('display.max_colwidth', 120)
    pd.set_option('display.width', torque_util.get_terminal_size()[1])
    print(expt)


if __name__ == '__main__':
    main(sys.argv[1:])
