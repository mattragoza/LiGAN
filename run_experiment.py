import sys, os, glob, time, re
import itertools
import numpy as np
import shlex
import subprocess as sp
import multiprocessing as mp

from parse_qstat import parse_qstat


def write_pbs_file(pbs_file, pbs_template_file, job_name, **kwargs):
    with open(pbs_template_file) as f:
        buf = f.read()
    defs = []
    buf = re.sub(r'#PBS -N JOB_NAME', '#PBS -N {}'.format(job_name), buf)
    for key, val in kwargs.items():
        var = key.upper()
        buf = re.sub(r'{}=.*'.format(var), '{}="{}"'.format(var, val), buf)
    with open(pbs_file, 'w') as f:
        f.write(buf)


def run_subprocess(cmd, stdin=None):
    args = shlex.split(cmd)
    proc = sp.Popen(args, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
    return proc.communicate(stdin)


def submit_job(pbs_file, array_idx):
    cmd = 'qsub {} -t {}'.format(pbs_file, array_idx)
    stdout, stderr = run_subprocess(cmd)
    if stderr:
        raise Exception(stderr)
    return stdout.strip().replace('[]', '[{}]'.format(array_idx))


def get_job_state(job_id):
    cmd = 'qstat -f -t {}'.format(job_id)
    stdout, stderr = run_subprocess(cmd)
    if stderr:
        return 'C' #ignore unknown job id
    return parse_qstat(stdout).loc[job_id]['job_state']


def submit_job_and_wait_to_complete(args):
    pbs_file, array_idx = args
    work_dir, pbs_file = os.path.split(pbs_file)
    if work_dir:
        orig_dir = os.getcwd()
        os.chdir(work_dir)
    job_id = submit_job(pbs_file, array_idx)
    time.sleep(5)
    while get_job_state(job_id) != 'C':
        time.sleep(5)
    if work_dir:
        os.chdir(orig_dir)


def get_n_gpus_free(queue):
    cmd = 'pbsnodes'
    stdout, stderr = run_subprocess(cmd)
    if stderr:
        raise Exception(stderr)
    df = parse_qstat(stdout)
    df = df[df['properties'].map(lambda qs: queue in qs.split(','))]
    df = df[df['state'] == 'free']
    n_gpus = df['gpus'].astype(int)
    cmd = 'qstat -f -t'
    stdout, stderr = run_subprocess(cmd)
    if stderr:
        raise Exception(stderr)
    df = parse_qstat(stdout)
    df = df[df['job_state'] == 'R']
    df['Node Id'] = df['gpus_reserved'].map(lambda g: g.split(':', 1)[0])
    df['gpu'] = df['gpus_reserved'].map(lambda g: g.split(':', 1)[1])
    df = df[df['Node Id'].map(lambda n: n in n_gpus)]
    n_gpus_used = df.groupby('Node Id')['gpu'].nunique().astype(int)
    n_gpus_free = n_gpus.subtract(n_gpus_used, fill_value=0).astype(int)
    return n_gpus_free.sum()


def wait_for_free_gpus_and_submit_job(args):
    pbs_file, array_idx = args
    work_dir, pbs_file = os.path.split(pbs_file)
    if work_dir:
        orig_dir = os.getcwd()
        os.chdir(work_dir)
    while get_n_gpus_free(queue='dept_gpu') < 5:
        time.sleep(5)
    job_id = submit_job(pbs_file, array_idx)
    print(job_id)
    time.sleep(5)
    if work_dir:
        os.chdir(orig_dir)


if __name__ == '__main__':
    pbs_template = 'train2.pbs'
    #model_files = [line.rstrip() for line in open('memory_error_models')]
    model_files = glob.glob('models/vce13_*_e.model')[:1]
    #df = parse_qstat(open('qjobs').read())
    #model_names = df[(df['euser'] == 'mtr22') & (df['job_state'] == 'Q')]['Job_Name']
    #model_names = [m for m in model_names if not len(glob.glob(m + '/' + m + '_iter_20000.caffemodel')) == 4]
    #model_files = ['models/' + m + '.model' for m in model_names]
    for model_file in model_files:
        assert os.path.isfile(model_file), 'file {} does not exist'.format(model_file)

    data_name = 'lowrmsd' #'genlowrmsd'
    data_root = '/net/pulsar/home/koes/dkoes/PDBbind/refined-set/' #general-set-with-refined/'
    max_iter = 20000
    seeds = [0]
    folds = [0, 1, 2, 3]

    args = []
    #for line in open('v13_cuda_errors').readlines()[:1]:
    #    m = re.match(r'(.+)/(.+)\.(.+)\.(\d+)\.(\d+|all)_iter_20000.caffemodel', line)
    #    model_name = m.group(1)
    #    print(model_name)
    #    seed = int(m.group(4))
    #    fold = 3 if m.group(5) == 'all' else int(m.group(5))
    for model_file, seed, fold in itertools.product(model_files, seeds, folds):
        model_name = os.path.splitext(os.path.split(model_file)[1])[0]
        print(model_name)
        if not os.path.isdir(model_name):
            os.makedirs(model_name)
        pbs_file = os.path.join(model_name, pbs_template)
        write_pbs_file(pbs_file, pbs_template, model_name,
                       model_name=model_name,
                       data_name=data_name,
                       data_root=data_root,
                       max_iter=max_iter)
        args.append((pbs_file, 4*seed+fold))

    map(wait_for_free_gpus_and_submit_job, args)

