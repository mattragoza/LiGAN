import sys, os, glob, time
import shlex
import subprocess as sp
import multiprocessing as mp

from parse_qstat import parse_qstat


def write_pbs_file(pbs_file, pbs_template_file, model_name, data_name, root_name, iterations):
    with open(pbs_template_file) as f:
        buf = f.read()
    buf = buf.replace('JOB_NAME', model_name)
    buf = buf.replace('MODEL_NAME', model_name)
    buf = buf.replace('DATA_NAME', data_name)
    buf = buf.replace('ROOT_NAME', root_name)
    buf = buf.replace('ITERATIONS', str(iterations))
    with open(pbs_file, 'w') as f:
        f.write(buf)


def run_subprocess(cmd, stdin=None):
    args = shlex.split(cmd)
    proc = sp.Popen(args, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
    return proc.communicate(stdin)


def submit_job(pbs_file, array_idx):
    cmd = 'qsub -t {} {}'.format(array_idx, pbs_file)
    stdout, stderr = run_subprocess(cmd)
    if stderr:
        raise Exception(stderr)
    return stdout.strip().replace('[]', '[{}]'.format(array_idx))


def get_job_state(job_id):
    cmd = 'qstat -f -t {}'.format(job_id)
    stdout, stderr = run_subprocess(cmd)
    if stderr:
        raise Exception(stderr)
    return parse_qstat(stdout).loc[job_id]['job_state']


def run_queue_job(args):
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


if __name__ == '__main__':
    pbs_template = 'train2.pbs'
    #model_files = [line.rstrip() for line in open('memory_error_models')]
    #model_files = glob.glob('models/*e13*.model')
    df = parse_qstat(open('qjobs').read())
    model_names = df[(df['euser'] == 'mtr22') & (df['job_state'] == 'Q')]['Job_Name']
    model_files = ['models/' + m + '.model' for m in model_names]
    for model_file in model_files:
        assert os.path.isfile(model_file), 'file {} does not exist'.format(model_file)

    data_name = 'lowrmsd' #'genlowrmsd'
    data_root = '/net/pulsar/home/koes/dkoes/PDBbind/refined-set/' #general-set-with-refined/'
    iterations = 20000
    seeds = [0]
    folds = [0, 1, 2, 3]
    n_processes = 20

    args = []
    for model_file in model_files:
        for seed in seeds:
            for fold in folds:
                model_name = os.path.splitext(os.path.split(model_file)[1])[0]
                if not os.path.isdir(model_name):
                    os.makedirs(model_name)
                pbs_file = os.path.join(model_name, pbs_template)
                write_pbs_file(pbs_file, pbs_template, model_name, data_name, data_root, iterations)
                args.append((pbs_file, 4*seed + fold))

    pool = mp.Pool(n_processes)
    pool.map(run_queue_job, args)
    pool.terminate()

