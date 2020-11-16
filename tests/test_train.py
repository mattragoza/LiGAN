import sys, os, pytest
import numpy as numpy
from numpy import isclose
from numpy.linalg import norm

os.environ['GLOB_minloglevel'] = '1'
import caffe

sys.path.insert(0, '.')
import train


def get_molgrid_data():
	return train.MolGridData(
			data_root='data/molport',
			batch_size=10,
			rec_map_file='my_rec_map',
			lig_map_file='my_lig_map',
			resolution=1.0,
			dimension=10,
			shuffle=False,
	)


class TestMolGridData(object):

	def test_init(self):
		d = get_molgrid_data()

	def test_populate_size(self):
		d = get_molgrid_data()
		d.populate('data/molportFULL_rand_test0_1000.types')
		assert d.size() == 1000

	def test_forward(self):
		d = get_molgrid_data()
		d.populate('data/molportFULL_rand_test0_1000.types')
		r, l = d.forward()
