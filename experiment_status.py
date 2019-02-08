from __future__ import print_function

import sys, os, re, glob, argparse
import pandas as pd

import torque_util


def read_file(file_):
    with open(file_, 'r') as f:
        return f.read()


def parse_pbs_file(pbs_file):
    buf = read_file(pbs_file)
    job_name = re.search(r'#PBS -N (.+)', buf).group(1)
    data_name = re.search(r'.*train\.py.*-p ([^\s]+)', buf).group(1)
    return job_name, data_name


def parse_output_file(out_file):
    lines = filter(len, read_file(out_file).split('\n'))
    return int(lines[-1]) if lines else None


def parse_stderr_file(stderr_file):
    buf = read_file(stderr_file)
    m = re.search(r'(([^\s]+(Error|Exception|Interrupt|Exit).*)|Segmentation fault|(Check failed.*))', buf)
    return m.group(0) if m else None


def get_job_status(pbs_file, job_id, qstat_data):

    error = None

    work_dir = os.path.dirname(pbs_file)
    try:
        job_name, data_name = parse_pbs_file(pbs_file)
    except IOError as e:
        return None, None, str(e).replace(repr(pbs_file), 'pbs_file')

    job_num, array_idx = map(int, job_id.split(']')[0].split('['))
    seed, fold = array_idx//4, array_idx%4
    out_base = '{}.{}.{}.{}.training_output'.format(job_name, data_name, seed, fold)

    try:
        job_state = qstat_data.loc[job_id, 'job_state']
    except KeyError:
        job_state = None

    if job_state == 'R': # track training progress from scr_dir while running
        gpu = qstat_data.loc[job_id, 'gpus_reserved'].split('.')[0]
        scr_dir = '/net/{}/scr/{}'.format(gpu, job_id)
        out_file = os.path.join(scr_dir, out_base)

    else: # check training output in work_dir, though it could be outdated
        out_file = os.path.join(work_dir, out_base)

    try:
        iter_ = parse_output_file(out_file)
    except IOError as e:
        iter_ = None

    stderr_file = os.path.join(work_dir, '{}.e{}-{}'.format(job_name, job_num, array_idx))
    try:
        error = parse_stderr_file(stderr_file)
    except IOError as e:
        error = str(e).replace(repr(stderr_file), 'stderr_file')

    return job_state, iter_, error


def parse_args(argv):
    parser = argparse.ArgumentParser(description='check status of GAN experiment')
    parser.add_argument('expt_file', help='file specifying experiment pbs scripts and job IDs')
    return parser.parse_args(argv)


def format_row(fields, cw):
    return '\t'.join(str(f).ljust(cw[i]) for i, f in enumerate(fields))


def main(argv):
    args = parse_args(argv)

    pbs_files = []
    job_ids = []
    with open(args.expt_file, 'r') as f:
        for line in f:
            pbs_file, job_id = line.split()[:2]
            pbs_files.append(pbs_file)
            job_ids.append(job_id.rstrip())

    qstat_data = torque_util.get_qstat_data()

    cw = [max(len(f) for f in pbs_files), 9, 9, 5]

    print(format_row(['pbs_file', 'job_state', 'iteration', 'error'], cw))
    for pbs_file, job_id in zip(pbs_files, job_ids):
        job_state, iter_, error = get_job_status(pbs_file, job_id, qstat_data)
        print(format_row([pbs_file, job_state, iter_, error], cw))


if __name__ == '__main__':
    main(sys.argv[1:])

