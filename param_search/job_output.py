import sys, os, re, shutil, argparse
from collections import defaultdict
import pandas as pd

from .job_queues import SlurmQueue


def read_stderr_file(stderr_file):
    warning_pat = re.compile(r'Warning.*')
    error_pat = re.compile(
        r'.*(Error|Exception|error|fault|failed|Errno).*'
    )
    error = None
    with open(stderr_file) as f:
        for line in f:
            if not warning_pat.match(line) and error_pat.match(line):
                error = line.rstrip()
    return error


def get_job_error(job_file, stderr_pat):
    '''
    Parse the latest error for job_file.
    '''
    job_dir = os.path.dirname(job_file)
    stderr_files = []
    for m in match_files_in_dir(job_dir, stderr_pat):
        stderr_file = m.group(0)
        job_id = int(m.group(1))
        stderr_files.append((job_id, stderr_file))

    job_id, stderr_file = sorted(stderr_files)[-1]
    stderr_file = os.path.join(job_dir, stderr_file)
    error = read_stderr_file(stderr_file)
    return error


def get_job_errors(job_files, stderr_pat=re.compile(r'(\d+).stderr')):
    '''
    Parse the latest errors for a set of job_files.
    '''
    errors = []
    for job_file in job_files:
        error = get_job_error(job_file, stderr_pat)
        errors.append(error)

    return errors


def get_job_output(job_file, output_pat):
    '''
    Read the latest output for job_file.
    '''
    job_dir = os.path.dirname(job_file)
    job_name = os.path.basename(job_dir)

    output_files = []
    for m in match_files_in_dir(job_dir, output_pat):
        output_file = m.group(0)
        job_id = int(m.group(1))
        output_files.append((job_id, output_file))

    job_id, output_file = sorted(output_files)[-1]
    output_file = os.path.join(job_dir, output_file)
    df = pd.read_csv(output_file, sep=' ')
    df['job_name'] = job_name
    return df


def get_job_outputs(job_files, output_pat=re.compile(r'(\d+).output')):
    '''
    Read the latest output for a set of job_files.
    '''
    dfs = []
    for job_file in job_files:
        df = get_job_output(job_file, output_pat)
        dfs.append(df)

    return pd.concat(dfs)


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


def parse_array_indices_str(s):
    idx_pat = re.compile(r'^(\d+)(-(\d+))?$')
    indices = []
    for field in s.split(','):
        m = idx_pat.match(field)
        idx_start = int(m.group(1))
        if m.group(2):
            idx_end = int(m.group(3))
            indices.extend(range(idx_start, idx_end+1))
        else:
            indices.append(idx_start)
    return set(indices)


def match_files_in_dir(dir, pattern):
    '''
    Iterate through files in dir that match pattern.
    '''
    for file in os.listdir(dir):
        m = pattern.match(file)
        if m is not None:
            yield m


def find_job_ids(job_dir, stderr_pat):
    '''
    Find job ids that have been submitted by
    parsing stderr file names in job_dir.
    '''
    job_ids = []
    for f in os.listdir(job_dir):
        if os.path.isdir(os.path.join(job_dir, f)):
            m = re.match(r'^(\d+)', f)
        else:
            m = re.match(stderr_pat, f)
        if m:
            job_id = int(m.group(1))
            job_ids.append(job_id)

    return job_ids


def read_job_output(job_dir, output_pat):
    '''
    Find job ids that have been submitted by
    parsing stderr file names in job_dir.
    '''
    job_dfs = []
    for m in match_files_in_dir(job_dir, output_pat):
        output_file = os.path.join(job_dir, m.group(0))
        print(output_file)
        job_df = pd.read_csv(output_file, sep=' ', error_bad_lines=False)
        job_df['job_name']  = os.path.split(job_dir)[-1]
        try:
            array_idx = int(m.group(2))
            job_df['array_idx'] = array_idx
        except:
            pass
        job_dfs.append(job_df)

    return job_dfs


def print_last_error(job_dir, stderr_pat):

    last_job_id = -1
    last_stderr_file = None
    for m in match_files_in_dir(job_dir, stderr_pat):
        stderr_file = os.path.join(job_dir, m.group(0))
        job_id = int(m.group(1))
        if job_id > last_job_id:
            last_job_id = job_id
            last_stderr_file = stderr_file

    if last_stderr_file is None:
        print('no error file')
        return

    error = read_stderr_file(last_stderr_file)
    print(last_stderr_file + '\t' + str(error))


def find_submitted_array_indices(job_dir, stderr_pat):
    '''
    Find array indices and job ids that have been
    submitted by parsing stderr file names in job_dir.
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


def copy_back_from_scr_dir(job_dir, scr_dir, copy_back_pat):
    '''
    Copy back output files from scr_dir to job_dir.
    '''
    copied = []
    for m in match_files_in_dir(scr_dir, copy_back_pat):
        copy_back_file = m.group(0)
        src_file = os.path.join(scr_dir, copy_back_file)
        dst_file = os.path.join(job_dir, copy_back_file)
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
            job_df['job_name'] = os.path.split(job_dir)[-1]
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
    parser.add_argument('job_scripts', nargs='+')
    parser.add_argument('--job_type')
    parser.add_argument('--array_job', default=False, action='store_true')
    parser.add_argument('--submitted', default=None)
    parser.add_argument('--copy_back', '-c', default=False, action='store_true')
    parser.add_argument('--print_indices', '-i', default=False, action='store_true')
    parser.add_argument('--print_errors', '-e', default=False, action='store_true')
    parser.add_argument('--resub_errors', '-r', default=False, action='store_true')
    parser.add_argument('--output_file', '-o')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    all_job_dfs = []
    for job_script in args.job_scripts:

        assert os.path.isfile(job_script), 'file ' + job_script + ' does not exist'

        if args.job_type is None: # infer from file name
            if 'fit' in job_script:
                args.job_type = 'fit'
            elif 'train' in job_script:
                args.job_type = 'train'

        # determine relevant files based on job type
        if args.job_type == 'train':
            job_script_pat = re.compile(r'(.*)_train.sh')
            output_ext = 'training_output'
            copy_back_exts = [
                'model', 'solver','caffemodel', 'solverstate', 'training_output', 'png', 'pdf'
            ]
        elif args.job_type == 'fit':
            job_script_pat = re.compile(r'(.*)_fit.sh')
            output_ext = 'gen_metrics'
            copy_back_exts = [
                'types', 'model', 'caffemodel', 'dx', 'sdf', 'channels', 'latent', 'pymol', 'gen_metrics'
            ]

        # for array jobs, get output for any array indices present
        if args.array_job:
            stderr_pat = re.compile(r'slurm-(\d+)_(\d+)\.err$')
            output_pat = re.compile(r'(.*)_(\d+)\.' + output_ext + '$')
            copy_back_pat = re.compile(r'(.*)_(\d+)\.' + '({})$'.format('|'.join(copy_back_exts)))
        else:
            stderr_pat = re.compile(r'slurm-(\d+)\.err$')
            output_pat = re.compile(r'(.*)\.' + output_ext + '$')
            copy_back_pat = re.compile(r'(.*)\.' + '({})$'.format('|'.join(copy_back_exts)))

        print(job_script)
        job_dir = os.path.dirname(job_script)

        if args.array_job:

            if args.submitted is not None:
                submitted = parse_array_indices_str(args.submitted)
                job_ids = find_job_ids(job_dir, stderr_pat)
            else:
                submitted, job_ids = find_submitted_array_indices(job_dir, stderr_pat)
            n_submitted = len(submitted)

            if n_submitted == 0:
                print('none submitted')
                continue

            completed, job_dfs = find_completed_array_indices(job_dir, output_pat, read=args.output_file)
            n_completed = len(completed)

            if args.output_file:
                all_job_dfs.extend(job_dfs)

            incomplete = submitted - completed
            n_incomplete = len(incomplete)

            if args.print_indices:
                print('n_submitted = {} ({})'.format(n_submitted, get_array_indices_string(submitted)))
                print('n_completed = {} ({})'.format(n_completed, get_array_indices_string(completed)))
                print('n_incomplete = {} ({})'.format(n_incomplete, get_array_indices_string(incomplete)))
            else:
                print('n_submitted = {}'.format(n_submitted))
                print('n_completed = {}'.format(n_completed))
                print('n_incomplete = {}'.format(n_incomplete))

            if args.print_errors:
                print_errors_for_array_indices(job_dir, stderr_pat, indices=incomplete)

            if args.copy_back:

                last_job_id = sorted(job_ids)[-1]
                scr_dir = os.path.join(job_dir, str(last_job_id))

                copied = copy_back_from_scr_dir(job_dir, scr_dir, copy_back_pat)
                n_copied = len(copied)
                print('copied {} files from {}'.format(n_copied, last_job_id))

            if args.resub_errors: # resubmit incomplete jobs

                for m in match_files_in_dir(job_dir, job_script_pat):
                    job_script = os.path.join(job_dir, m.group(0))
                    SlurmQueue.submit_job(
                        job_script,
                        work_dir=job_dir,
                        array_idx=get_array_indices_string(incomplete)
                    )

        else:
            job_ids = find_job_ids(job_dir, stderr_pat)

            if args.output_file:
                job_dfs = read_job_output(job_dir, output_pat)
                all_job_dfs.extend(job_dfs)

            if args.print_errors:
                print_last_error(job_dir, stderr_pat)

            if args.copy_back:

                for last_job_id in sorted(job_ids):
                    scr_dir = os.path.join(job_dir, str(last_job_id))
                    copied = copy_back_from_scr_dir(job_dir, scr_dir, copy_back_pat)
                    n_copied = len(copied)
                    print('copied {} files from {}'.format(n_copied, last_job_id))

    if args.output_file:
        if all_job_dfs:
            job_df = pd.concat(all_job_dfs)
            pd.set_option('display.max_columns', 100)
            print(job_df.groupby('job_name').mean())
            job_df.to_csv(args.output_file, sep=' ')
            print('concatenated metrics to {}'.format(args.output_file))
        else:
            print('nothing to concatenate')


if __name__ == '__main__':
    main(sys.argv[1:])

