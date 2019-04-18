import sys, os, argparse
import params
import caffe_util


def write_file(file_, buf):
    with open(file_, 'w') as f:
        f.write(buf)


def write_solvers(solver_dir, param_space, print_=False):
    '''
    Write a solver in solver_dir for every param assignment in param_space.
    '''
    if not os.path.isdir(solver_dir):
        os.makedirs(solver_dir)

    for solver_params in param_space:
        solver_file = os.path.join(solver_dir, '{}.solver'.format(solver_params))
        write_solver(solver_file, solver_params)
        if print_:
            print(solver_file)


def write_solver(solver_file, solver_params):
    '''
    Write a dict of solver_params to solver_file.
    '''
    buf = ''
    if solver_params:
        buf += params.format_params(solver_params, '# ')
    buf += str(make_solver(**solver_params))
    write_file(solver_file, buf)


def make_solver(**solver_params):
    '''
    Create a SolverParameter from a dict of solver_params.
    '''
    solver = caffe_util.SolverParameter()
    for param, value in solver_params.items():
        setattr(solver, param, value)
    return solver


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Create solver prototxt files from solver params')
    parser.add_argument('params_file', help='file defining solver params or dimensions of param space')
    parser.add_argument('-n', '--solver_name', required=True, help='solver name format')
    parser.add_argument('-o', '--out_dir', required=True, help='common output directory for solver files')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    param_space = params.ParamSpace(args.params_file, format=args.solver_name.format)
    write_solvers(args.out_dir, param_space, print_=True)


if __name__ == '__main__':
    main(sys.argv[1:])
