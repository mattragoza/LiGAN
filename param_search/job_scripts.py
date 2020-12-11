import sys, os, re, argparse
import itertools

from . import params
from .common import read_file, write_file


def setup_job_scripts(
    expt_dir,
    name_format,
    template_file,
    param_space,
):
    '''
    Write a job script in a separate sub dir of expt_dir
    for every set of params in param_space, by formatting
    template_file with the params. Name the created dirs
    by formatting name_format with the params.
    '''
    template = read_file(template_file)
    job_base = os.path.basename(template_file)

    job_files = []
    for job_params in param_space:

        job_name = name_format.format(**job_params)
        job_params['job_name'] = job_name
        job_dir = os.path.join(expt_dir, job_name)

        if not os.path.isdir(job_dir):
            os.makedirs(job_dir)

        job_file = os.path.join(job_dir, job_base)
        write_job_script(job_file, template, job_params)
        job_files.append(job_file)

    return job_files


def write_job_script(job_file, template, job_params):
    '''
    Write a job script to job_file by filling in
    template with job_params.
    '''
    params_str = params.format_params(job_params, line_start='# ')
    write_file(job_file, template.format(job_params=params_str, **job_params))


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
        if param == 'val_options':
            value = expand_val_options(value)
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
        l='--fit_L1_loss',
        t='--constrain_types',
        f='--constrain_frags',
        e='--estimate_types',
        a='--alt_bond_adding',
        d='--dkoes_simple_fit',
        D='--dkoes_make_mol',
    )[a] for a in args)


def expand_val_options(args):
    return ' '.join(dict(
        D='--dkoes_make_mol',
        o='--use_openbabel',
    )[a] for a in args)


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Create job scripts from a template and job params')
    parser.add_argument('params_file', help='file defining job params or dimensions of param space')
    parser.add_argument('-o', '--expt_dir', default='.', help='common directory for job working directories')
    parser.add_argument('-n', '--name_format', required=True, help='job name format')
    parser.add_argument('-t', '--template_file', required=True, help='job script template file')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    param_space = params.ParamSpace.from_file(args.params_file)
    setup_job_scripts(
        expt_dir=args.expt_dir,
        name_format=args.name_format,
        template_file=args.template_file,
        param_space=param_space
    )


if __name__ == '__main__':
    main(sys.argv[1:])
