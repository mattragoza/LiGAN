import os
import params
import caffe_util


def write_file(file_, buf):
    with open(file_, 'w') as f:
        f.write(buf)


def write_solvers(solver_dir, param_space):
	'''
	Write a solver in solver_dir for every set of params in param_space.
	'''
	if not os.path.isdir(solver_dir):
		os.makedirs(solver_dir)

	for solver_params in param_space:
		solver_file = os.path.join(solver_dir, '{}.solver'.format(solver_params))
		write_solver(solver_file, solver_params)


def write_solver(solver_file, solver_params):
	'''
	Write a set of solver_params to solver_file.
	'''
	buf = ''
	if solver_params:
		buf += params.format_params(solver_params, '# ')
	buf += str(make_solver(**solver_params))
	write_file(solver_file, buf)


def make_solver(**solver_params):
	'''
	Create a SolverParameter from a set of solver_params.
	'''
	solver = caffe_util.SolverParameter()
	for param, value in solver_params.items():
		setattr(solver, param, value)
	return solver
