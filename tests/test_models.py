import sys, os, pytest
import numpy as numpy
from numpy import isclose
from numpy.linalg import norm

os.environ['GLOB_minloglevel'] = '1'
import caffe

sys.path.insert(0, '.')
import caffe_util as cu
import models


def get_input_blob(shape):
	return cu.Input(shape=dict(dim=shape))()


class TestConvReLU(object):

	def test_init(self):
		n_input = 5
		n_output = 6
		shape = (10, n_input, 3, 3, 3)
		x = get_input_blob(shape)
		f = models.ConvReLU(n_input, n_output, 3, 0.1)
		y = f(x)
		assert len(y.bottoms) == 2
		assert isinstance(y.bottoms[0], cu.Convolution)
		assert isinstance(y.bottoms[1], cu.ReLU)
		assert y.bottoms[0].bottoms == [x]
		assert y.bottoms[1].bottoms == [y]
		assert x.tops == y.bottoms[:1]
		assert y.tops == y.bottoms[1:]

	def test_to_param(self):
		n_input = 5
		n_output = 6
		shape = (10, n_input, 3, 3, 3)
		x = get_input_blob(shape)
		f = models.ConvReLU(n_input, n_output, 3, 0.1)
		y = f(x)
		y.scaffold()


# y = CaffeBlob
#	Convolution, ReLU
#		x = CaffeBlob
#			Input