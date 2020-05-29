import sys, os, re, shutil, argparse
from collections import defaultdict
import pandas as pd

from job_queue import SlurmQueue


metric_file_pat = re.compile(r'(.*)_(\d+)\.gen_metrics')
error_file_pat  = re.compile(r'slurm-(\d+)_(\d+)\.err')
job_script_pat  = re.compile(r'(csb|crc|bridges)_fit.sh')
inf = float('inf')


def read_err_file(err_file):
    wrn_pat = re.compile(r'Warning.*')
    err_pat = re.compile(r'.*(Error|Exception|error|fault|failed).*')
    error = None
    with open(err_file) as f:
        for line in f:
            if not wrn_pat.match(line) and err_pat.match(line):
                error = line.rstrip()
    return error


files_cache = dict()
def get_files(dir):
    global files_cache
    if dir not in files_cache:
        files_cache[dir] = os.listdir(dir)
    return files_cache[dir]


def print_index_set(idx_set):
    s = get_index_set_str(idx_set)
    print(s)


def get_index_set_str(idx_set):
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


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('job_dirs', nargs='+')
    parser.add_argument('--copy_metrics', '-c', default=False, action='store_true')
    parser.add_argument('--print_idxs',   '-i', default=False, action='store_true')
    parser.add_argument('--print_errors', '-e', default=False, action='store_true')
    parser.add_argument('--resub_errors', '-r', default=False, action='store_true')
    parser.add_argument('--output_file',  '-o')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    for job_dir in args.job_dirs:
        assert os.path.isdir(job_dir), job_dir + ' is not a directory'

    job_dfs = []
    for job_dir in args.job_dirs:

        print(job_dir)

        # find set of submitted array_idxs from error files
        job_ids = []
        submitted_idxs = set()
        for job_file in get_files(job_dir):
            m = error_file_pat.match(job_file)
            if not m: continue
            job_id    = int(m.group(1))
            array_idx = int(m.group(2))
            job_ids.append(job_id)
            submitted_idxs.add(array_idx)

        n_submitted = len(submitted_idxs)
        print('n_submitted = {}'.format(n_submitted))
        if n_submitted == 0:
            continue
        elif args.print_idxs:
            print_index_set(submitted_idxs)

        if args.copy_metrics: # copy back metric files from latest scr_dir
            last_job_id = sorted(job_ids)[-1]
            scr_dir = job_dir + '/' + str(last_job_id)
            n_copied = 0
            for scr_file in get_files(scr_dir):
                m = metric_file_pat.match(scr_file)
                if not m: continue
                job_name  = m.group(1)
                array_idx = int(m.group(2))
                src_file = scr_dir + '/' + scr_file
                dst_file = job_dir + '/' + scr_file
                shutil.copyfile(src_file, dst_file)
                n_copied += 1
            print('copied {} metrics files from {}'.format(n_copied, last_job_id))

        # find set of completed array_idxs from metrics files
        complete_idxs = set()
        for job_file in get_files(job_dir):
            m = metric_file_pat.match(job_file)
            if not m: continue
            array_idx = int(m.group(2))
            complete_idxs.add(array_idx)
            if args.output_file:
                job_file = job_dir + '/' + job_file
                job_df = pd.read_csv(job_file, sep=' ')
                job_df['job_name']  = os.path.split(job_dir)[-1]
                job_df['array_idx'] = array_idx
                job_dfs.append(job_df)

        n_complete = len(complete_idxs)
        print('n_complete = {}'.format(n_complete))
        if args.print_idxs:
            print_index_set(complete_idxs)

        # determine set of incomplete array_idxs
        incomplete_idxs = submitted_idxs - complete_idxs
        n_incomplete = len(incomplete_idxs)
        print('n_incomplete = {}'.format(n_incomplete))
        if args.print_idxs:
            print_index_set(incomplete_idxs)

        print()

        if args.print_errors: # parse stderr files for incomplete indices

            error_files = defaultdict(list)
            for job_file in get_files(job_dir):
                m = error_file_pat.match(job_file)
                if not m: continue
                job_id    = int(m.group(1))
                array_idx = int(m.group(2))
                if array_idx in incomplete_idxs:
                    error_files[array_idx].append((job_id, job_file))

            for array_idx in sorted(incomplete_idxs):
                if not error_files[array_idx]:
                    print('no error file for array_idx {}'.format(array_idx))
                    continue
                job_id, error_file = sorted(error_files[array_idx])[-1]
                error_file = os.path.join(job_dir, error_file)
                error = read_err_file(error_file)
                print(error_file + '\t' + str(error))

            print()

        if args.resub_errors: # resubmit incomplete jobs

            for job_file in get_files(job_dir):
                m = job_script_pat.match(job_file)
                if not m: continue
                job_file = job_dir + '/' + job_file
                SlurmQueue.submit_job(job_file, work_dir=job_dir, array_idx=get_index_set_str(incomplete_idxs))

    if args.output_file:
        if job_dfs:
            pd.concat(job_dfs).to_csv(args.output_file, sep=' ')
            print('concatenated metrics to {}'.format(args.output_file))
        else:
            print('nothing to concatenate')

