from __future__ import print_function

import sys, os, re, glob, argparse, string
import datetime as dt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def get_terminal_size():
    with os.popen('stty size') as p:
        return map(int, p.read().split())

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', get_terminal_size()[1])

import params
import models
import solvers
import job_templates
import job_queue


class Experiment(object):
    '''
    An object for managing a collection of jobs based on job_template_file
    with placeholder values filled in from job_params.
    '''
    _status_cols = ['job_name', 'job_id', 'job_state', 'error']

    def __init__(self, expt_name, expt_dir, job_template_file, job_params):

        self.name = expt_name
        self.dir = os.path.abspath(expt_dir)

        self.job_queue = job_queue.get_job_queue(job_template_file)
        self.job_template = read_file(job_template_file)
        self.job_base = os.path.basename(job_template_file)
        self.job_params = params.ParamSpace(job_params)

        self.df = pd.DataFrame(list(job_params.flatten(scope='job_params')))
        self.df['work_dir'] = self.df.apply(self._get_work_dir, axis=1)
        self.df['job_file'] = self.df.apply(self._get_job_file, axis=1)
        self.df['array_idx'] = self.df.apply(self._get_array_idx, axis=1)

    @classmethod
    def from_file(cls, expt_file):
        raise NotImplementedError('TODO')

    def _get_work_dir(self, job):
        return os.path.join(self.dir, job['job_name'])

    def _get_job_file(self, job):
        return os.path.join(job['work_dir'], self.job_base)

    def _get_array_idx(self, job):
        return job['job_params.array_idx']

    def _find_job_id(self, job):
        job_ids = set()
        for file_ in os.listdir(job['work_dir']):
            m = re.match(r'slurm-(\d+)\.(out|err)', file_)
            if m:
                job_ids.add(int(m.group(1)))
        try:
            return sorted(job_ids)[-1]
        except IndexError:
            return -1

    def _parse_error(self, job):
        stderr_file = os.path.join(job['curr_dir'], 'slurm-{}.err'.format(job['job_id']))
        return parse_stderr_file(stderr_file)

    def _update_job_status(self, job, qstat):
       
        if job['job_name'] in qstat['job_name'].values:
            job_qstat = qstat.loc[qstat['job_name'] == job['job_name']].iloc[-1]
            job['job_state'] = job_qstat['job_state']
            job['job_id'] = job_qstat['job_id']
        else:
            job['job_state'] = '-'
            job['job_id'] = self._find_job_id(job)

        if job['job_id'] != -1:
            job['scr_dir'] = self.job_queue.get_scr_dir(job['job_id'])

        if job['job_state'] == 'R':
            job['curr_dir'] = job['scr_dir']
        else:
            job['curr_dir'] = job['work_dir']

        try:
            job['error'] = self._parse_error(job)
        except IOError:
            job['error'] = None

        return job

    def status(self):
        qstat = self.job_queue.get_status(self.df['job_name'])
        self.df = self.df.apply(self._update_job_status, axis=1, qstat=qstat)
        print(self.df[self._status_cols])

    def _setup_job(self, job):
        
        if not os.path.isdir(job['work_dir']):
            os.makedirs(job['work_dir'])

        job_params = params.Params((p.replace('job_params.', ''), v) for p, v in job.items() if p.startswith('job_params'))
        job_params.name = job['job_name']

        job_templates.write_job_script(job['job_file'], self.job_template, job_params)
        print(job['job_file'])

    def setup(self):
        self.df.apply(self._setup_job, axis=1)

    def _submit_job(self, job):
        os.chdir(job['work_dir'])
        job_id = self.job_queue.submit_job(job['job_file'], job['array_idx'])
        print(job_id)
        return job_id

    def run(self):
        self.df['job_id'] = self.df.apply(self._submit_job, axis=1)

    def _sync_job(self, job):
        pass

    def sync(self):
        self.df.apply(self._sync_job, axis=1)

    def main(self):
        parser = argparse.ArgumentParser('Manage the experiment')
        parser.add_argument('command', help='one of {setup, status, run, sync}')
        args = parser.parse_args()
        getattr(self, args.command)()


class TrainExperiment(Experiment):

    _status_cols = ['job_name', 'job_id', 'job_state', 'test_iter', 'save_iter', 'error']

    def _get_array_idx(self, job):
        return 4*job['job_params.seed'] + job['job_params.fold']

    def _setup_job(self, job):
        model_dir = os.path.join(self.dir, 'models')
        solver_dir = os.path.join(self.dir, 'solvers')
        models.write_models(model_dir, self.job_params['data_model_params'])
        models.write_models(model_dir, self.job_params['gen_model_params'])
        models.write_models(model_dir, self.job_params['disc_model_params'])
        solvers.write_solvers(solver_dir, self.job_params['solver_params'])
        Experiment._setup_job(self, job)

    def _parse_test_iter(self, job):
        out_file = os.path.join(job['work_dir'], '{}.{}.{}.{}.training_output' \
            .format(job['job_name'], job['job_params.data_prefix'], job['job_params.seed'], job['job_params.fold']))
        return parse_output_file(out_file)

    def _find_save_iter(self, job):
        return find_save_iter(job['work_dir'])

    def _update_job_status(self, job, qstat):
        job = Experiment._update_job_status(self, job, qstat)     
        job['test_iter'] = self._parse_test_iter(job)
        job['save_iter'] = self._find_save_iter(job)
        return job

    def _sync_job(self):
        pass


def read_file(file_):
    with open(file_, 'r') as f:
        return f.read()


def write_file(file_, buf):
    with open(file_, 'w') as f:
        f.write(buf)


def parse_pbs_file(pbs_file):
    buf = read_file(pbs_file)
    job_name = re.search(r'#PBS -N (.+)', buf).group(1)
    try:
        data_name = re.search(r'DATA_NAME="([^\s]+)"', buf).group(1)
    except AttributeError:
        data_name = re.search(r'.*train\.py.*-p ([^\s]+)', buf).group(1)
    return job_name, data_name


def parse_output_file(out_file):
    try:
        buf = read_file(out_file)
        return int(re.findall(r'^(\d+)\s+', buf, re.MULTILINE)[-1])
    except IOError: # file does not exist
        return -1
    except IndexError: # file exists but is empty
        return -1


def parse_stderr_file(stderr_file):
    buf = read_file(stderr_file)
    m = re.search(r'(([^\s]+(Error|Exception|Interrupt|Exit).*)|Segmentation fault|(Check failed.*))', buf)
    return m.group(0) if m else None


def find_save_iter(dir_):
    save_iter = 0
    states = glob.glob(os.path.join(dir_, '*.solverstate'))
    for s in states:
        m = re.match(dir_ + r'.*_iter_(\d+)\.solverstate', s)
        if m:
            iter_ = int(m.group(1))
            save_iter = max(save_iter, iter_)
    return save_iter


def submit_incomplete_jobs(job):

    if job['job_state'] not in ['Q', 'R'] and job['error'] is not None:
        torque_util.submit_job((job['pbs_file'], job['array_idx']))


def update_job_fields(job, qstat):
    '''
    Update experiment status by parsing qstat command, training
    output, and stderr files. Expects pbs_file and array_idx fields
    to be present and accurate, while job_id is resolved by first
    checking latest in queue, then latest in work_dir.
    '''
    work_dir = os.path.dirname(job['pbs_file'])
    job_name, data_name = parse_pbs_file(job['pbs_file'])
    full_job_name = '{}-{}'.format(job_name, job['array_idx'])

    try: # find latest job_id in queue
        job_qstat = qstat[qstat['Job_Name'] == full_job_name].iloc[-1]
        job['job_id'] = job_qstat.name
        job['job_state'] = job_qstat['job_state']

    except IndexError: # not in queue, find in work_dir
        job['job_state'] = np.nan
        try:
            job['job_id'] = find_job_ids(job['pbs_file'], job['array_idx'])[-1]

        except IndexError: # could not find job_id
            pass

    # find latest training output file
    seed, fold = job['array_idx']//4, job['array_idx']%4
    if fold == 3:
        fold = 'all'
    out_base = '{}.{}.{}.{}.training_output'.format(job_name, data_name, seed, fold)

    if job['job_state'] == 'R': # if running, check scr_dir
        job['node_id'] = job_qstat['gpus_reserved'].split('.')[0]
        scr_dir = '/net/{}/scr/{}'.format(job['node_id'], job['job_id'])
        out_file = os.path.join(scr_dir, out_base)

    else: # otherwise check work_dir
        out_file = os.path.join(work_dir, out_base)

    try: # get iteration from training output file
        job['time_modified'] = dt.datetime.fromtimestamp(os.path.getmtime(out_file))
        job['iteration'] = parse_output_file(out_file)

    except IndexError: # training output file is empty
        job['iteration'] = None

    except (OSError, IOError): # couldn't find or read training output file
        pass

    try: # get error type from the stderr_file
        job_num, _ = parse_job_id(job['job_id'])
        stderr_file = os.path.join(work_dir, '{}.e{}-{}'.format(job_name, job_num, job['array_idx']))
        job['error'] = parse_stderr_file(stderr_file)

    except AttributeError:
        job['error'] = None

    except TypeError: # job_id is nan = job was never submitted
        pass

    except IOError: # couldn't find or read stderr_file = job not finished or stderr not copied
        pass

    return job


def fix_job_fields(job):
    '''
    Attempt to reconcile different experiment file formats. At minimum
    the correct pbs_file and array_idx are required to update job status.
    '''
    if not is_pbs_file(job['pbs_file']):
        try: # it's the job_id
            job_num, job['array_idx'] = parse_job_id(job['pbs_file'])
            job['job_id'] = job['pbs_file']
            job['pbs_file'] = find_pbs_file(job_num)
        except AttributeError: # it's the work_dir
            job['pbs_file'] = glob.glob(os.path.join(job['pbs_file'], '*.pbs'))[-1]

    if not isinstance(job['array_idx'], int):
        try: # it's the job_id
            job_num, job['array_idx'] = parse_job_id(job['array_idx'])
            job['job_id'] = job['array_idx']
        except TypeError: # it's null
            job['array_idx'] = 3 # TODO this is a hack

    return job


def is_pbs_file(path):
    '''
    Test whether a path is a file with .pbs extension.
    '''
    return os.path.isfile(path) and path.endswith('.pbs')


def find_pbs_file(job_num):
    '''
    Find the pbs_file associated with a job_num.
    '''
    raise NotImplementedError('TODO')


def find_job_ids(pbs_file, array_idx):
    '''
    Find job_ids associated with a pbs_file and array_idx, sorted by
    job_num, by finding stdout and stderr files in work_dir.
    '''
    work_dir = os.path.dirname(pbs_file)
    job_name, _ = parse_pbs_file(pbs_file)
    job_nums = set()
    for f in os.listdir(work_dir):
        m = re.match(r'{}\.[oe](\d+)-{}'.format(job_name, array_idx), f)
        if m:
            job_nums.add(int(m.group(1)))
    return [format_job_id(j, array_idx) for j in sorted(job_nums)]


def format_job_id(job_num, array_idx):
    return '{:d}[{:d}].n198.dcb.private.net'.format(job_num, array_idx)


def parse_job_id(job_id):
    m = re.match(r'^(\d+)\[(\d+)\]\.n198\.dcb\.private\.net$', job_id)
    return int(m.group(1)), int(m.group(2))


def write_expt_file(expt_file, df):
    df.to_csv(expt_file, sep=' ', index=False, header=False)


def read_expt_file(expt_file):
    names = ['pbs_file', 'array_idx', 'job_id', 'node_id', 'job_state', 'iteration', 'time_modified', 'error']
    return pd.read_csv(expt_file, sep=' ', names=names).apply(fix_job_fields, axis=1)


def parse_args(argv):
    parser = argparse.ArgumentParser(description='check status of GAN experiment')
    sub_parsers = parser.add_subparsers()

    setup_parser = sub_parsers.add_parser('setup', help='setup new experiment')
    setup_parser.add_argument()

    status_parser = sub_parsers.add_parser('status', help='check experiment status')

    run_parser = sub_parsers.add_parser('run', help='run experiment jobs')
    #parser.add_argument('expt_file', help='file specifying experiment pbs scripts and job IDs')
    #parser.add_argument('-s', '--submit', default=False, action='store_true', help='submit jobs that aren\'t in queue or have errors')
    #parser.add_argument('-o', '--out_file', help='output file to write updated experiment status')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    print(args)

if False:

    expt = read_expt_file(args.expt_file)
    qstat = torque_util.get_qstat_data()
    expt = expt.apply(update_job_fields, axis=1, qstat=qstat)

    if args.submit:
        expt.apply(submit_incomplete_jobs, axis=1)
        qstat = torque_util.get_qstat_data()
        expt = expt.apply(update_job_fields, axis=1, qstat=qstat)
        qstat = torque_util.get_qstat_data()

    if args.out_file:
        write_expt_file(args.out_file, expt)

    print(expt)


if __name__ == '__main__':
    main(sys.argv[1:])
