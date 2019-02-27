import sys, os, re, argparse
import itertools


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


def fill_template(template, **kwargs):
    '''
    Return a copy of template string with uppercase instance of keys 
    from kwargs replaced with their values.
    '''
    for key, val in kwargs.items():
        template = re.sub(key.upper(), str(val), template)
    return template


def expand_train_options(*args):
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
        kwargs['train_options'] = expand_train_options(*kwargs['train_options'])
        pbs_filled = fill_template(pbs_template, **kwargs)

        with open(pbs_file, 'w') as f:
            f.write(pbs_filled)
            print(pbs_file)


if __name__ == '__main__':
    main(sys.argv[1:])
