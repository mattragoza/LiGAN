import sys, os, pytest
from numpy import isclose
from caffe.proto import caffe_pb2
os.environ['GLOG_minloglevel'] = '1'

sys.path.insert(0, '.')
from liGAN.data import AtomGridData


class TestAtomGridData(object):

	@pytest.fixture
	def data(self):
		return AtomGridData(
			data_root='data/molport',
			batch_size=10,
			rec_map_file='data/my_rec_map',
			lig_map_file='data/my_lig_map',
			resolution=0.5,
			dimension=23.5,
			shuffle=False,
		)

	@pytest.fixture
	def param(self):
		param = caffe_pb2.MolGridDataParameter()
		param.root_folder = 'data/molport'
		param.batch_size = 10
		param.recmap = 'data/my_rec_map'
		param.ligmap = 'data/my_lig_map'
		param.resolution = 0.5
		param.dimension = 23.5
		return param

	def test_init(self, data):
		assert data.rec_lig_split == 16
		assert data.ex_provider
		assert data.grid_maker
		assert data.grid.shape == (10, 16+19, 48, 48, 48)
		assert isclose(0, data.grid.norm().cpu())
		assert data.size() == 0

	def test_from_param(self, param):
		data = AtomGridData.from_param(param)
		assert data.rec_lig_split == 16
		assert data.ex_provider
		assert data.grid_maker
		assert data.grid.shape == (10, 16+19, 48, 48, 48)
		assert isclose(0, data.grid.norm().cpu())
		assert data.size() == 0

	def test_populate(self, data):
		data.populate('data/molportFULL_rand_test0_1000.types')
		assert data.size() == 1000

	def test_populate2(self, data):
		data.populate('data/molportFULL_rand_test0_1000.types')
		data.populate('data/molportFULL_rand_test0_1000.types')
		assert data.size() == 2000

	def test_forward(self, data):
		data.populate('data/molportFULL_rand_test0_1000.types')
		rec, lig = data.forward()
		assert rec.shape == (10, 16, 48, 48, 48)
		assert lig.shape == (10, 19, 48, 48, 48)
		assert not isclose(0, rec.norm().cpu())
		assert not isclose(0, lig.norm().cpu())
