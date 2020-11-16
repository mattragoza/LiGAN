import sys, os, pytest
import numpy as numpy
from numpy import isclose
from numpy.linalg import norm

os.environ['GLOB_minloglevel'] = '1'
import caffe

sys.path.insert(0, '.')
import caffe_util as cu


def get_net_param():
	net_param = cu.NetParameter()
	s = '''\
layer {
	name: "input"
	type: "Input"
	top: "input"
	input_param {
		shape {
			dim: 1
			dim: 2
			dim: 3
		}
	}
}
	'''
	cu.text_format.Merge(s, net_param)
	return net_param


class TestCaffeSolver(object):

	def test_init(self):
		s = cu.CaffeSolver()

	def test_init_kwargs(self):
		net_param = get_net_param()
		s = cu.CaffeSolver(type='SGD', net_param=net_param)
		assert s.param_.type == 'SGD'
		assert s.param_.net_param == net_param

	def test_scaffold_type_known(self):
		net_param = get_net_param()
		s = cu.CaffeSolver(type='SGD', net_param=net_param)
		s.scaffold()
		assert isinstance(s, cu.CaffeSolver)
		assert isinstance(s, caffe.SGDSolver)

	def test_scaffold_type_unknown(self):
		net_param = get_net_param()
		s = cu.CaffeSolver(type='ASDF', net_param=net_param)
		with pytest.raises(AttributeError):
			s.scaffold()

	def test_scaffold_type_multiple(self):
		net_param = get_net_param()
		s1 = cu.CaffeSolver(type='SGD', net_param=net_param)
		s2 = cu.CaffeSolver(type='Adam', net_param=net_param)
		s1.scaffold()
		s2.scaffold()
		assert isinstance(s1, caffe.SGDSolver)
		assert isinstance(s2, caffe.AdamSolver)
