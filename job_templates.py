import sys, os, re, argparse
import itertools
import params
import job_queue


SOLVER_NAME_FORMAT = '{solver_name}_{gen_train_iter:d}_{disc_train_iter:d}_{train_options}_{instance_noise:g}_{loss_weight:g}_{loss_weight_decay:g}'

SOLVER_SEARCH_SPACE = dict(
    solver_name=['adam0'],
    gen_train_iter=[2],
    disc_train_iter=[2, 4],
    train_options=['', 'b'],
    instance_noise=[0.0],
    loss_weight=[1.0, 0.5],
    loss_weight_decay=[0.0, 1e-5, 2e-5],
    memory=['24gb'],
    walltime=['672:00:00'],
    queue=['dept_gpu']
)


def keyword_product(**kwargs):
    for values in itertools.product(*kwargs.itervalues()):
        yield dict(itertools.izip(kwargs.iterkeys(), values))


def read_file(file_):
    with open(file_, 'r') as f:
        return f.read()


def write_file(file_, buf):
    with open(file_, 'w') as f:
        f.write(buf)


def write_job_scripts(expt_dir, job_template_file, param_space):
    '''
    Write a job script in a separate sub dir of expt_dir for every
    set of params in param_space, by filling in job_template_file.
    '''
    job_template = read_file(job_template_file)
    job_base = os.path.basename(job_template_file)

    for job_params in param_space:

        job_params['job_name'] = str(job_params)
        job_dir = os.path.join(expt_dir, str(job_params))
        if not os.path.isdir(job_dir):
            os.makedirs(job_dir)

        job_file = os.path.join(job_dir, job_base)
        write_job_script(job_file, job_template, job_params)


def write_job_script(job_file, job_template, job_params):
    '''
    Write a job script to job_file by filling in job_template with job_params.
    '''
    buf = params.format_params(job_params, '# ')
    buf = fill_job_template(job_template, dict(job_name=job_params.name, job_params=buf))
    buf = fill_job_template(buf, job_params)
    write_file(job_file, buf)


def fill_job_template(job_template, job_params):
    '''
    Return a copy of job_template string with uppercase instances
    of keys from job_params replaced with their values.
    '''
    job = job_template
    for param, value in job_params.items():
        if param == 'train_options':
            value = expand_train_options(value)
        job = re.sub('<'+param.upper()+'>', str(value), job)
    return job


def expand_train_options(args):
    return ' '.join(dict(
        a='--alternate',
        b='--balance',
        g='--disc_grad_norm',
        s='--disc_spectral_norm',
    )[a] for a in args)


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Create pbs script templates with GAN solver parameters')
    parser.add_argument('-o', '--out_prefix', default='pbs_templates', help='common prefix for pbs output files')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    with open('pbs_templates/template.pbs', 'r') as f:
        pbs_template = f.read()

    for kwargs in keyword_product(**SOLVER_SEARCH_SPACE):

        pbs_name = SOLVER_NAME_FORMAT.format(**kwargs)

        pbs_file = os.path.join(args.out_prefix, pbs_name + '.pbs')
        kwargs['train_options'] = expand_train_options(kwargs['train_options'])
        pbs_filled = fill_template(pbs_template, **kwargs)

        with open(pbs_file, 'w') as f:
            f.write(pbs_filled)
            print(pbs_file)


if __name__ == '__main__':
    main(sys.argv[1:])
