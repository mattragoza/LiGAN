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
    for values in itertools.product(*kwargs.values()):
        yield dict(zip(kwargs.keys(), values))


def read_file(file_):
    with open(file_, 'r') as f:
        return f.read()


def write_file(file_, buf):
    with open(file_, 'w') as f:
        f.write(buf)


def write_job_scripts(expt_dir, job_template_file, param_space, print_=False):
    '''
    Write a job script in a separate sub dir of expt_dir for every
    set of params in param_space, by filling in job_template_file.
    '''
    job_template = read_file(job_template_file)
    job_base = os.path.basename(job_template_file)

    for job_params in param_space:

        job_dir = os.path.join(expt_dir, job_params.name)
        if not os.path.isdir(job_dir):
            os.makedirs(job_dir)

        job_file = os.path.join(job_dir, job_base)
        write_job_script(job_file, job_template, job_params)
        if print_:
            print(job_file)


def write_job_script(job_file, job_template, job_params):
    '''
    Write a job script to job_file by filling in job_template with job_params.
    '''
    buf = params.format_params(job_params, line_start='# ')
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
        if param == 'gen_options':
            value = expand_gen_options(value)
        job = re.sub('<'+param.upper()+'>', str(value), job)
    return job


def expand_train_options(args):
    return ' '.join(dict(
        a='--alternate',
        b='--balance',
        g='--disc_grad_norm',
        s='--disc_spectral_norm',
    )[a] for a in args)


def expand_gen_options(args):
    return ' '.join(dict(
        p='--prior',
        m='--mean',
        i='--interpolate',
        s='--spherical',
        r='--random_rotation',
        M='--multi_atom',
        c='--apply_conv',
        t='--constrain_types',
        f='--constrain_frags',
    )[a] for a in args)


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Create job scripts from a template and job params')
    parser.add_argument('params_file', help='file defining job params or dimensions of param space')
    parser.add_argument('-b', '--job_template', required=True, help='job script template file')
    parser.add_argument('-o', '--out_dir', required=True, help='common directory for job working directories')
    parser.add_argument('-n', '--job_name', required=True, help='job name format')
    parser.add_argument('-e', '--expt_name', help='experiment name, for status file')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    param_space = params.ParamSpace(args.params_file, format=args.job_name.format)
    write_job_scripts(args.out_dir, args.job_template, param_space, print_=True)


if __name__ == '__main__':
    main(sys.argv[1:])
