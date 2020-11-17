import sys, os, pytest
import numpy as numpy
from numpy import isclose
from numpy.linalg import norm

os.environ['GLOB_minloglevel'] = '1'
import caffe

sys.path.insert(0, '.')
import caffe_util as cu
import models


def get_input_blob(*shape):
	return cu.Input(shape=dict(dim=shape))()


class TestConvReLU(object):

	def test_init(self):
		x = get_input_blob(10, 5, 3, 3, 3)
		f = models.ConvReLU(5, 6, 3, 0.1)
		y = f(x)
		assert len(y.bottoms) == 2
		assert isinstance(y.bottoms[0], cu.Convolution)
		assert isinstance(y.bottoms[1], cu.ReLU)
		assert y.bottoms[0].bottoms == [x]
		assert y.bottoms[1].bottoms == [y]
		assert x.tops == y.bottoms[:1]
		assert y.tops == y.bottoms[1:]

	def test_scaffold(self):
		in_shape = (10, 5, 3, 3, 3)
		out_shape = (10, 6, 3, 3,3)
		x = get_input_blob(*in_shape)
		f = models.ConvReLU(5, 6, 3, 0.1)
		y = f(x)
		y.scaffold()
		assert y.has_scaffold()
		assert x.has_scaffold()
		assert x.shape == in_shape
		assert y.shape == out_shape
		assert isclose(0, norm(x.data))
		assert isclose(0, norm(y.data))

	def test_forward_zero(self):
		x = get_input_blob(10, 5, 3, 3, 3)
		f = models.ConvReLU(5, 6, 3, 0.1)
		y = f(x)
		y.scaffold()
		f.forward()
		assert isclose(0, norm(x.data))
		assert isclose(0, norm(y.data))
		assert isclose(0, norm(y.diff))
		assert isclose(0, norm(x.diff))

	def test_forward_ones(self):
		x = get_input_blob(10, 5, 3, 3, 3)
		f = models.ConvReLU(5, 6, 3, 0.1)
		y = f(x)
		y.scaffold()
		x.data[...] = 1
		f.forward()
		assert not isclose(0, norm(x.data))
		assert not isclose(0, norm(y.data))
		assert isclose(0, norm(y.diff))
		assert isclose(0, norm(x.diff))

	def test_backward_zero(self):
		x = get_input_blob(10, 5, 3, 3, 3)
		f = models.ConvReLU(5, 6, 3, 0.1)
		y = f(x)
		y.scaffold()
		f.forward()
		f.backward()
		assert isclose(0, norm(x.data))
		assert isclose(0, norm(y.data))
		assert isclose(0, norm(y.diff))
		assert isclose(0, norm(x.diff))

	def test_backward_ones(self):
		x = get_input_blob(10, 5, 3, 3, 3)
		f = models.ConvReLU(5, 6, 3, 0.1)
		y = f(x)
		y.scaffold()
		f.forward()
		y.diff[...] = 1
		f.backward()
		print(y.net.to_param())
		assert isclose(0, norm(x.data))
		assert isclose(0, norm(y.data))
		assert not isclose(0, norm(y.diff))
		assert not isclose(0, norm(x.diff))
