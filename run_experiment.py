import sys, os, glob, time, re
import itertools
import numpy as np
import shlex
import subprocess as sp
import multiprocessing as mp

from parse_qstat import parse_qstat

MIN_GPUS_FREE = 4


def write_pbs_file(pbs_file, pbs_template_file, job_name, **kwargs):
    with open(pbs_template_file) as f:
        buf = f.read()
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
    try:
        df['Node Id'] = df['gpus_reserved'].map(lambda g: g.split(':', 1)[0])
    except AttributeError:
        print(df['gpus_reserved'][df['gpus_reserved'].map(lambda g: type(g) == float)])
        raise
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
    while get_n_gpus_free(queue='dept_gpu') <= MIN_GPUS_FREE:
        time.sleep(5)
    job_id = submit_job(pbs_file, array_idx)
    print(job_id)
    time.sleep(5)
    if work_dir:
        os.chdir(orig_dir)


if __name__ == '__main__':

    _, models_file = sys.argv
    pbs_templates = ['bgan.pbs', 'agan.pbs', 'abgan.pbs']

    model_names = [line.rstrip() for line in open(models_file)]
    model_files = ['models/' + m + '.model' for m in model_names]
    for model_file in model_files:
        assert os.path.isfile(model_file), 'file {} does not exist'.format(model_file)

    data_name = 'lowrmsd' #'genlowrmsd'
    data_root = '/net/pulsar/home/koes/dkoes/PDBbind/refined-set/' #general-set-with-refined/'
    max_iter = 20000
    cont_iter = 0
    seeds = [0]
    folds = [0, 1, 2, 3]

    args = []
    for pbs_template, model_file, seed, fold in itertools.product(pbs_templates, model_files, seeds, folds):
        model_name = os.path.splitext(os.path.split(model_file)[1])[0]
        if 'gan' in pbs_template:
            gan_type = os.path.splitext(os.path.basename(pbs_template))[0]
            resolution = model_name.split('_')[3]
            gen_model_name = model_name
            data_model_name = 'data_24_{}'.format(resolution)
            disc_model_name = 'disc' #'info_2048' if '1024' in model_name else 'info_1024'
            solver_name = 'adam0'
            gen_warmup_name = model_name.lstrip('_')
            gan_name = '{}{}_{}'.format(gan_type, gen_model_name, disc_model_name)
            if not os.path.isdir(gan_name):
                os.makedirs(gan_name)
            pbs_file = os.path.join(gan_name, pbs_template)
            write_pbs_file(pbs_file, pbs_template, gan_name,
                           gan_name=gan_name,
                           data_model_name=data_model_name,
                           gen_model_name=gen_model_name,
                           disc_model_name=disc_model_name,
                           data_name=data_name,
                           data_root=data_root,
                           solver_name=solver_name,
                           max_iter=max_iter,
                           cont_iter=cont_iter,
                           gen_warmup_name=gen_warmup_name)
        else:
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

