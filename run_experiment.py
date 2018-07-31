import sys, os, glob, time, re
import itertools
import numpy as np
import shlex
import subprocess as sp
import multiprocessing as mp

from parse_qstat import parse_qstat

MIN_GPUS_FREE = 4


def write_pbs_file(pbs_file, pbs_template_file, job_name, **kwargs):
    '''
    Make a copy of a pbs job script file that has -n JOB_NAME replaced
    with job_name and any variable definitions whose names are all-
    caps versions of keys in kwargs redefined as the arg value.
    '''
    with open(pbs_template_file) as f:
        buf = f.read()
    buf = re.sub(r'#PBS -N JOB_NAME', '#PBS -N {}'.format(job_name), buf)
    for key, val in kwargs.items():
        var = key.upper()
        buf = re.sub(r'{}=.*'.format(var), '{}="{}"'.format(var, val), buf)
    with open(pbs_file, 'w') as f:
        f.write(buf)


def run_subprocess(cmd, stdin=None):
    '''
    Run a subprocess with the given stdin and return (stdout, stderr).
    '''
    args = shlex.split(cmd)
    proc = sp.Popen(args, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
    return proc.communicate(stdin)


def submit_job(pbs_file, array_idx):
    '''
    Submit a job script with an array index using qsub.
    '''
    cmd = 'qsub {} -t {}'.format(pbs_file, array_idx)
    stdout, stderr = run_subprocess(cmd)
    if stderr:
        raise Exception(stderr)
    return stdout.strip().replace('[]', '[{}]'.format(array_idx))


def get_job_state(job_id):
    '''
    Query the state of a job with the given job_id using qstat.
    '''
    cmd = 'qstat -f -t {}'.format(job_id)
    stdout, stderr = run_subprocess(cmd)
    if stderr:
        return 'C' #ignore unknown job id
    return parse_qstat(stdout).loc[job_id]['job_state']


def submit_job_and_wait_to_complete(args):
    '''
    Submit a job with the args (pbs_file, array_idx) and then wait
    for it to complete, querying its state every 5 seconds. The job
    will be submit from the dir containing the job script.
    '''
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
    '''
    Compute the number of gpus in a particular job queue that are
    not currently being used by any jobs, using pbsnodes and qstat.
    '''
    # get total n_gpus in each currently available node in the queue
    cmd = 'pbsnodes'
    stdout, stderr = run_subprocess(cmd)
    if stderr:
        raise Exception(stderr)
    df = parse_qstat(stdout)
    df = df[df['properties'].map(lambda qs: queue in qs.split(','))]
    df = df[df['state'] == 'free']
    n_gpus = df['gpus'].astype(int)
    # get jobs running on those nodes that are using gpus
    cmd = 'qstat -f -t'
    stdout, stderr = run_subprocess(cmd)
    if stderr:
        raise Exception(stderr)
    df = parse_qstat(stdout)
    df = df[df['job_state'] == 'R']
    df = df[df['gpus_reserved'].notnull()]
    df['Node Id'] = df['gpus_reserved'].map(lambda g: g.split(':', 1)[0])
    df['gpu']     = df['gpus_reserved'].map(lambda g: g.split(':', 1)[1])
    df = df[df['Node Id'].map(lambda n: n in n_gpus)]
    # get num unique gpus being used by those jobs on each node
    n_gpus_used = df.groupby('Node Id')['gpu'].nunique().astype(int)
    # return total free gpus as sum of total gpus minus gpus in use, per node
    n_gpus_free = n_gpus.subtract(n_gpus_used, fill_value=0).astype(int)
    return n_gpus_free.sum()


def wait_for_free_gpus_and_submit_job(args):
    '''
    Wait for a certain number of gpus to be free and then
    submit a job with args (pbs_file, array_idx). The job
    will be submit from the dir containing the job script.
    '''
    pbs_file, array_idx = args
    work_dir, pbs_file = os.path.split(pbs_file)
    if work_dir:
        orig_dir = os.getcwd()
        os.chdir(work_dir)
    queue = 'dept_gpu'
    last_n_gpus_free = -1
    n_gpus_free = get_n_gpus_free(queue)
    while n_gpus_free <= MIN_GPUS_FREE:
        if n_gpus_free != last_n_gpus_free:
            print(n_gpus_free)
        time.sleep(5)
        n_gpus_free = get_n_gpus_free(queue)
    job_id = submit_job(pbs_file, array_idx)
    print(job_id)
    time.sleep(5)
    if work_dir:
        os.chdir(orig_dir)


if __name__ == '__main__':

    _, params_file = sys.argv
    pbs_templates = ['bgan.pbs', 'agan.pbs', 'abgan.pbs']
    params = [line.rstrip().split() for line in open(params_file)]

    data_name = 'lowrmsd' #'genlowrmsd'
    data_root = '/net/pulsar/home/koes/dkoes/PDBbind/refined-set/' #general-set-with-refined/'
    max_iter = 20000
    cont_iter = 0

    args = []
    for pbs_template, gen_model_file, disc_model_file, seed, fold in params:
        seed, fold = int(seed), int(fold)
        gen_model_name = os.path.splitext(os.path.split(gen_model_file)[1])[0]
        if 'gan' in pbs_template:
            gan_type = os.path.splitext(os.path.basename(pbs_template))[0]
            resolution = gen_model_name.split('_')[3]
            data_model_name = 'data_24_{}'.format(resolution)
            disc_model_name = os.path.splitext(os.path.split(disc_model_file)[1])[0]
            solver_name = 'adam0'
            gen_warmup_name = gen_model_name.lstrip('_')
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
            if not os.path.isdir(gen_model_name):
                os.makedirs(gen_model_name)
            pbs_file = os.path.join(gen_model_name, pbs_template)
            write_pbs_file(pbs_file, pbs_template, gen_model_name,
                           model_name=gen_model_name,
                           data_name=data_name,
                           data_root=data_root,
                           max_iter=max_iter)
        args.append((pbs_file, 4*seed+fold))

    map(wait_for_free_gpus_and_submit_job, args)

