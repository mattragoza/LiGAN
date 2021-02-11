import sys, os, re, shlex
from contextlib import contextmanager
import pandas as pd
from subprocess import Popen, PIPE

from .common import read_file, write_file


class SubprocessError(RuntimeError):
    '''
    Raised when a subprocess fails, and contains stderr.
    '''
    pass


class JobQueue(object):
    '''
    An abstract interface for communicating with a job
    scheduling system such as Slurm or PBS Torque.
    '''
    @classmethod
    def get_submit_cmd(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_status_cmd(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def parse_submit_out(cls, stdout):
        raise NotImplementedError

    @classmethod
    def parse_status_out(cls, stdout):
        raise NotImplementedError

    @classmethod
    def submit_job_script(cls, job_file, *args, **kwargs):
        job_file = os.path.abspath(job_file)
        work_dir = os.path.dirname(job_file)
        submit_cmd = cls.get_submit_cmd(job_file, *args, **kwargs)
        submit_out = call_subprocess(submit_cmd, work_dir=work_dir)
        return cls.parse_submit_out(submit_out)

    @classmethod
    def submit_job_scripts(cls, job_files, *args, **kwargs):
        job_ids = []
        for job_file in job_files:
            job_id = cls.submit_job_script(job_file, *args, **kwargs)
            job_ids.append(job_id)
        return job_ids

    @classmethod
    def get_job_status(cls, *args, **kwargs):
        status_cmd = cls.get_status_cmd(**kwargs)
        status_out = call_subprocess(status_cmd)
        return cls.parse_status_out(status_out)


class SlurmQueue(JobQueue):

    @classmethod
    def get_submit_cmd(cls, job_file, array_idx=None):
        cmd = 'sbatch '
        if array_idx is not None:
            cmd += '--array={} '.format(array_idx)
        return cmd + job_file

    @classmethod
    def get_status_cmd(cls, *args, **kwargs):
        out_format = r'%i %P %j %u %t %M %l %R %Z'
        status_cmd = 'squeue --format="{}"'.format(out_format)
        for a in args:
            status_cmd += ' ' + a
        for k, v in kwargs.items():
            status_cmd += ' --' + k + ' ' + v
        return status_cmd

    @classmethod
    def parse_submit_out(cls, stdout):
        return int(re.match(
            r'^Submitted batch job (\d+)( on cluster .+)?\n$',
            stdout
        ).group(1))

    @classmethod
    def parse_status_out(cls, stdout):

        stdout = stdout[stdout.index('JOBID'):]
        lines = stdout.split('\n')
        columns = lines[0].split(' ')
        col_data = {c: [] for c in columns}
        for line in filter(len, lines[1:]):
            fields = paren_split(line, sep=' ')
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


class TorqueQueue(JobQueue):

    @classmethod
    def get_submit_cmd(cls, job_file, array_idx=None):
        cmd = 'qsub ' + job_file
        if array_idx is not None:
            cmd += ' -t {}'.format(array_idx)
        return cmd

    @classmethod
    def get_status_cmd(cls, job_names):
        return 'qstat'

    @classmethod
    def parse_submit_out(cls, stdout):
        try:
            return int(re.match(
                r'^(\d+)\.n198\.dcb\.private\.net\n$',
                stdout
            ).group(1))
        except Exception as e:
            print(stdout)
            raise

    @classmethod
    def parse_status_out(cls, stdout):
        raise NotImplementedError('TODO')


class DummyQueue(JobQueue):

    @classmethod
    def get_submit_cmd(cls, job_file, array_idx=None):
        return 'echo hello, world'

    @classmethod
    def get_status_cmd(cls, job_names):
        return 'OK'


def run_subprocess(cmd, stdin=None, work_dir=None):
    '''
    Run cmd as a subprocess with the given stdin,
    from the given work_dir, and return (stdout, stderr).
    '''
    if sys.platform == 'win32':
        args = cmd
    else:
        args = shlex.split(cmd)

    proc = Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=work_dir)
    stdout, stderr = proc.communicate(stdin)

    if isinstance(stdout, bytes):
        stdout = stdout.decode()

    if isinstance(stderr, bytes):
        stderr = stderr.decode()

    return stdout, stderr


def call_subprocess(cmd, stdin=None, work_dir=None):
    '''
    Run cmd as a subprocess and raise an exc-
    eption if there is any stderr.
    '''
    stdout, stderr = run_subprocess(cmd, stdin, work_dir)
    if stderr:
        raise SubprocessError(stderr)
    return stdout



def paren_split(string, sep):
    '''
    Split string by instances of sep character that are
    outside of balanced parentheses.
    '''
    fields = []
    last_sep = -1
    esc_level = 0
    for i, char in enumerate(string):
        if char in sep and esc_level == 0:
            fields.append(string[last_sep+1:i])
            last_sep = i
        elif char == '(':
            esc_level += 1
        elif char == ')':
            if esc_level > 0:
                esc_level -= 1
            else:
                raise ValueError('missing open parentheses')
    if esc_level == 0:
        fields.append(string[last_sep+1:])
    else:
        raise ValueError('missing close parentheses')
    return fields


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
