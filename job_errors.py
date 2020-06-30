import sys, os, re, shutil, argparse
from collections import defaultdict
import pandas as pd

from job_queue import SlurmQueue


def read_stderr_file(stderr_file):
    warning_pat = re.compile(r'Warning.*')
    error_pat = re.compile(r'.*(Error|Exception|error|fault|failed).*')
    error = None
    with open(stderr_file) as f:
        for line in f:
            if not warning_pat.match(line) and error_pat.match(line):
                error = line.rstrip()
    return error


def print_array_indices(idx_set):
    s = get_array_indices_string(idx_set)
    print(s)


def get_array_indices_string(idx_set):
    s = ''
    last_idx = None
    skipping = False
    for idx in sorted(idx_set):
        if last_idx is None:
            s += str(idx)
        elif idx == last_idx + 1:
            skipping = True
        else: # gap
            if skipping:
                skipping = False
                s += '-' + str(last_idx)
            s += ',' + str(idx)
        last_idx = idx
    if skipping:
        s += '-' + str(last_idx)
    return s


def match_files_in_dir(dir, pattern):
    '''
    Iterate through files in dir that match pattern.
    '''
    for file in os.listdir(dir):
        m = pattern.match(file)
        if m is not None:
            yield m


def find_submitted_array_indices(job_dir, stderr_pat):
    '''
    Find array indices and job ids that have been
    submitted by parsing stderr files in job_dir.
    '''
    submitted = set()
    job_ids = []
    for m in match_files_in_dir(job_dir, stderr_pat):
        stderr_file = m.group(0)
        job_id = int(m.group(1))
        array_idx = int(m.group(2))
        job_ids.append(job_id)
        submitted.add(array_idx)

    return submitted, job_ids


def copy_back_from_scr_dir(job_dir, scr_dir, output_pat):
    '''
    Copy back output files from scr_dir to job_dir.
    '''
    copied = []
    for m in match_files_in_dir(scr_dir, output_pat):
        output_file = m.group(0)
        src_file = os.path.join(scr_dir, output_file)
        dst_file = os.path.join(job_dir, output_file)
        shutil.copyfile(src_file, dst_file)
        copied.append(dst_file)

    return copied


def find_completed_array_indices(job_dir, output_pat, read=False):
    '''
    Find array_indices that have completed by parsing output
    files in job_dir, also optionally read and return job dfs.
    '''
    job_dfs = []
    completed = set()
    for m in match_files_in_dir(job_dir, output_pat):
        array_idx = int(m.group(2))
        completed.add(array_idx)
        if read:
            output_file = os.path.join(job_dir, m.group(0))
            job_df = pd.read_csv(output_file, sep=' ')
            job_df['job_name']  = os.path.split(job_dir)[-1]
            job_df['array_idx'] = array_idx
            job_dfs.append(job_df)

    return completed, job_dfs


def print_errors_for_array_indices(job_dir, stderr_pat, indices):

    stderr_files = defaultdict(list)
    for m in match_files_in_dir(job_dir, stderr_pat):
        stderr_file = m.group(0)
        job_id = int(m.group(1))
        array_idx = int(m.group(2))
        if array_idx in indices:
            stderr_files[array_idx].append((job_id, stderr_file))

    for array_idx in sorted(indices):
        if not stderr_files[array_idx]:
            print('no error file for array_idx {}'.format(array_idx))
            continue
        job_id, stderr_file = sorted(stderr_files[array_idx])[-1]
        stderr_file = os.path.join(job_dir, stderr_file)
        error = read_stderr_file(stderr_file)
        print(stderr_file + '\t' + str(error))


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('job_dirs', nargs='+')
    parser.add_argument('--copy_back', '-c', default=False, action='store_true')
    parser.add_argument('--print_indices', '-i', default=False, action='store_true')
    parser.add_argument('--print_errors', '-e', default=False, action='store_true')
    parser.add_argument('--resub_errors', '-r', default=False, action='store_true')
    parser.add_argument('--output_file', '-o')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    for job_dir in args.job_dirs:
        assert os.path.isdir(job_dir), job_dir + ' is not a directory'

    output_pat = re.compile(r'(.*)_(\d+)\.gen_metrics')
    stderr_pat = re.compile(r'slurm-(\d+)_(\d+)\.err')
    inf = float('inf')

    job_dfs = []
    for job_dir in args.job_dirs:
        print(job_dir)

        submitted, job_ids = find_submitted_array_indices(job_dir, stderr_pat)

        n_submitted = len(submitted)
        print('n_submitted = {}'.format(n_submitted))

        if n_submitted == 0:
            continue

        if args.print_indices:
            print_array_indices(submitted)

        if args.copy_back:
            last_job_id = sorted(job_ids)[-1]
            scr_dir = job_dir + '/' + str(last_job_id)

            copied = copy_back_from_scr_dir(job_dir, scr_dir, output_pat)

            n_copied = len(copied)
            print('copied {} files from {}'.format(n_copied, last_job_id))

        completed, job_dfs = find_completed_array_indices(job_dir, output_pat, read=args.output_file)

        n_completed = len(completed)
        print('n_completed = {}'.format(n_completed))

        if args.print_indices:
            print_array_indices(completed)

        incomplete = submitted - completed

        n_incomplete = len(incomplete)
        print('n_incomplete = {}'.format(n_incomplete))

        if args.print_indices:
            print_array_indices(incomplete)

        if args.print_errors: # parse stderr files for incomplete indices
            print_errors_for_array_indices(job_dir, stderr_pat, indices=incomplete)

        if args.resub_errors: # resubmit incomplete jobs

            for m in match_files_in_dir(job_dir):
                job_script = os.path.join(job_dir, m.group(0))
                SlurmQueue.submit_job(
                    job_file,
                    work_dir=job_dir,
                    array_idx=get_array_indices_string(incomplete)
                )

    if args.output_file:
        if job_dfs:
            pd.concat(job_dfs).to_csv(args.output_file, sep=' ')
            print('concatenated metrics to {}'.format(args.output_file))
        else:
            print('nothing to concatenate')


if __name__ == '__main__':
    main(sys.argv[1:])

