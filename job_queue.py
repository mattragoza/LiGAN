from __future__ import print_function
import sys, os, glob, time, re
import shlex
from subprocess import Popen, PIPE
import pandas as pd
from io import StringIO


def run_subprocess(cmd, stdin=None):
    '''
    Run a subprocess with the given stdin and return (stdout, stderr).
    '''
    args = shlex.split(cmd)
    proc = Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    return proc.communicate(stdin)


def read_file(file_):
    with open(file_, 'r') as f:
        return f.read()


def get_job_queue(job_file):
    buf = read_file(job_file)
    if re.search(r'^#PBS ', buf, re.MULTILINE):
        return TorqueQueue
    elif re.search(r'^#SBATCH ', buf, re.MULTILINE):
        return SlurmQueue
    else:
        raise ValueError('unknown job queue type')


class JobQueue(object):

    @staticmethod
    def _status_cmd(job_names):
        raise NotImplementedError

    @staticmethod
    def _submit_cmd(job_file, array_idx):
        raise NotImplementedError

    @staticmethod
    def _parse_status(stdout):
        raise NotImplementedError

    @staticmethod
    def _parse_submit(stdout):
        raise NotImplementedError

    @classmethod
    def get_status(cls, job_names):
        cmd = cls._status_cmd(job_names)
        stdout, stderr = run_subprocess(cmd)
        if stderr:
            raise Exception(stderr)
        else:
            return cls._parse_status(stdout)

    @classmethod
    def submit_job(cls, job_file, array_idx):
        cmd = cls._submit_cmd(job_file, array_idx)
        stdout, stderr = run_subprocess(cmd)
        if stderr:
            raise Exception(stderr)
        else:
            return cls._parse_submit(stdout)


class TorqueQueue(JobQueue):
    
    pass #TODO


class SlurmQueue(JobQueue):

    @staticmethod
    def _status_cmd(job_names):
        out_format = r'%i %P %j %u %t %M %l %R %Z'
        return 'squeue --cluster=gpu --name={} --format="{}"'.format(','.join(job_names), out_format)

    @staticmethod
    def _submit_cmd(job_file, array_idx):
        return 'sbatch {} --array={}'.format(job_file, array_idx)

    @staticmethod
    def _parse_status(stdout):

        lines = stdout.split('\n')
        columns = lines[1].split(' ')
        col_data = {c: [] for c in columns}
        for line in filter(len, lines[2:]):
            fields = line.split(' ')
            for i, field in enumerate(fields):
                col_data[columns[i]].append(field)

        return pd.DataFrame(col_data).rename(columns={
            'JOBID': 'job_id',
            'PARTITION': 'queue',
            'NAME': 'job_name',
            'USER': 'user',
            'ST': 'job_state',
            'TIME': 'runtime',
            'TIME_LIMIT': 'walltime',
            'NODELIST(REASON)': 'node_id',
            'WORK_DIR': 'work_dir'
        })

    @staticmethod
    def _parse_submit(stdout):
        return int(re.match(r'^Submitted batch job (\d+) on cluster .+\n$', stdout).group(1))

    @staticmethod
    def get_scr_dir(self, job_id):
        return '/scratch/slumr-{}'.format(job_id)


def get_terminal_size():
    with os.popen('stty size') as p:
        return map(int, p.read().split())


def parse_qstat(buf, job_delim='\n\n', field_delim='\n    ', index_name=None):
    '''
    Parse the stdout of either qstat -f or pbsnodes and return it in a
    data frame indexed either by job ID or node ID, respectively.
    '''
    assert buf, 'nothing to parse'
    all_job_data = []
    for job_buf in filter(len, buf.split(job_delim)):
        job_data = dict()
        for field_buf in filter(len, job_buf.split(field_delim)):
            if not job_data:
                if index_name is None:
                    name, value = field_buf.split(': ', 1)
                    index_name = name
                else:
                    name, value = index_name, field_buf.split(': ', 1)[-1]
            else:
                name, value = field_buf.split(' = ', 1)
            job_data[name] = value.replace('\n\t', '')
        all_job_data.append(job_data)
    return pd.DataFrame(all_job_data).set_index(index_name)


def parse_pbsnodes(buf):
    return parse_qstat(buf, field_delim='\n     ', index_name='Node Id')


def get_qstat_data():
    '''
    Query the status of all jobs in the queue.
    '''
    cmd = 'qstat -f -t'
    stdout, stderr = run_subprocess(cmd)
    if stderr:
        raise Exception(stderr)
    else:
        return parse_qstat(stdout)


def get_pbsnodes_data():
    '''
    Query the status of all nodes in the queue.
    '''
    cmd = 'pbsnodes'
    stdout, stderr = run_subprocess(cmd)
    if stderr:
        raise Exception(stderr)
    else:
        return parse_pbsnodes(stdout)


def get_job_state(job_id):
    '''
    Query the state of a job with the given job_id using qstat.
    '''
    try:
        return get_qstat_data().loc[job_id, 'job_state']
    except KeyError:
        return None


def qsub_job(pbs_file, array_idx=None):
    '''
    Submit a job script with an optional array index using qsub,
    and return the job ID (including the array index).
    '''
    cmd = 'qsub {}'.format(pbs_file)
    if array_idx is not None:
        cmd += ' -t {}'.format(array_idx)
    stdout, stderr = run_subprocess(cmd)
    if stderr:
        raise Exception(stderr)
    job_id = stdout.strip()
    if array_idx is not None:
        job_id = job_id.replace('[]', '[{}]'.format(array_idx))
    return job_id


def submit_job(args):
    '''
    Submit a job with the args (pbs_file, array_idx). The job
    will be submit from the dir containing the job script.
    '''
    pbs_file, array_idx = args
    work_dir, pbs_file = os.path.split(pbs_file)
    if work_dir:
        orig_dir = os.getcwd()
        os.chdir(work_dir)
    job_id = qsub_job(pbs_file, array_idx)
    if work_dir:
        os.chdir(orig_dir)
    print('{}/{} {}'.format(work_dir, pbs_file, job_id))
    return job_id


def submit_job_and_wait_to_complete(args, poll_every='5'):
    '''
    Submit a job with the args (pbs_file, array_idx) and then wait
    for it to complete, querying its state every 5 seconds. The job
    will be submit from the dir containing the job script.
    '''
    job_id = submit_job(args)
    time.sleep(poll_every)
    while get_job_state(job_id) in {'Q', 'R'}:
        time.sleep(poll_every)
    return job_id


def get_n_gpus_free(queue):
    '''
    Compute the number of gpus in a particular job queue that are
    not currently being used by any jobs, using pbsnodes and qstat.
    '''
    # get total n_gpus in each currently available node in the queue
    df = get_pbsnodes_data()
    df = df[df['properties'].map(lambda qs: queue in qs.split(','))]
    df = df[df['state'] == 'free']
    n_gpus = df['gpus'].astype(int)

    # get jobs running on those nodes that are using gpus
    df = get_qstat_data()
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


def wait_for_free_gpus_and_submit_job(args, n_gpus_free=5, poll_every=5, queue='dept_gpu'):
    '''
    Wait for a certain number of gpus to be free and then
    submit a job with args (pbs_file, array_idx). The job
    will be submit from the dir containing the job script.
    ''' 
    while get_n_gpus_free(queue) < n_gpus_free:
        time.sleep(poll_every)
    job_id = submit_job(args)
    time.sleep(poll_every)
    return job_id
