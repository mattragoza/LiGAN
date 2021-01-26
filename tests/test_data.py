import sys, os, pytest
from numpy import isclose
from caffe.proto import caffe_pb2
os.environ['GLOG_minloglevel'] = '1'

sys.path.insert(0, '.')
from liGAN.data import AtomGridData


class TestAtomGridData(object):

    @pytest.fixture
    def lig_data(self):
        return AtomGridData(
            data_root='data/molport',
            batch_size=10,
            rec_map_file=None,
            lig_map_file='data/my_lig_map',
            resolution=0.5,
            dimension=23.5,
            shuffle=False,
        )

    @pytest.fixture
    def rec_lig_data(self):
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
    def rec_lig_param(self):
        param = caffe_pb2.MolGridDataParameter()
        param.root_folder = 'data/molport'
        param.batch_size = 10
        param.recmap = 'data/my_rec_map'
        param.ligmap = 'data/my_lig_map'
        param.resolution = 0.5
        param.dimension = 23.5
        return param

    @pytest.fixture
    def lig_param(self):
        param = caffe_pb2.MolGridDataParameter()
        param.root_folder = 'data/molport'
        param.batch_size = 10
        param.recmap = ''
        param.ligmap = 'data/my_lig_map'
        param.resolution = 0.5
        param.dimension = 23.5
        return param

    def test_lig_data_init(self, lig_data):
        assert lig_data.n_rec_channels == 0
        assert lig_data.n_lig_channels == 19
        assert lig_data.ex_provider
        assert lig_data.grid_maker
        assert lig_data.grids.shape == (10, 19, 48, 48, 48)
        assert isclose(0, lig_data.grids.norm().cpu())
        assert lig_data.size == 0

    def test_rec_lig_data_init(self, rec_lig_data):
        assert rec_lig_data.n_rec_channels == 16
        assert rec_lig_data.n_lig_channels == 19
        assert rec_lig_data.ex_provider
        assert rec_lig_data.grid_maker
        assert rec_lig_data.grids.shape == (10, 16+19, 48, 48, 48)
        assert isclose(0, rec_lig_data.grids.norm().cpu())
        assert rec_lig_data.size == 0

    def test_lig_data_from_param(self, lig_param):
        lig_data = AtomGridData.from_param(lig_param)
        assert lig_data.n_rec_channels == 0
        assert lig_data.n_lig_channels == 19
        assert lig_data.ex_provider
        assert lig_data.grid_maker
        assert lig_data.grids.shape == (10, 19, 48, 48, 48)
        assert isclose(0, lig_data.grids.norm().cpu())
        assert lig_data.size == 0

    def test_rec_lig_data_from_param(self, rec_lig_param):
        rec_lig_data = AtomGridData.from_param(rec_lig_param)
        assert rec_lig_data.n_rec_channels == 16
        assert rec_lig_data.n_lig_channels == 19
        assert rec_lig_data.ex_provider
        assert rec_lig_data.grid_maker
        assert rec_lig_data.grids.shape == (10, 16+19, 48, 48, 48)
        assert isclose(0, rec_lig_data.grids.norm().cpu())
        assert rec_lig_data.size == 0

    def test_rec_lig_data_populate(self, rec_lig_data):
        rec_lig_data.populate('data/molportFULL_rand_test0_1000.types')
        assert rec_lig_data.size == 1000

    def test_rec_lig_data_populate2(self, rec_lig_data):
        rec_lig_data.populate('data/molportFULL_rand_test0_1000.types')
        rec_lig_data.populate('data/molportFULL_rand_test0_1000.types')
        assert rec_lig_data.size == 2000

    def test_lig_data_forward_ok(self, lig_data):
        lig_data.populate('data/molportFULL_rand_test0_1000.types')
        lig_grids, labels = lig_data.forward()
        assert lig_grids.shape == (10, 19, 48, 48, 48)
        assert not isclose(0, lig_grids.norm().cpu())

    def test_rec_lig_data_forward_ok(self, rec_lig_data):
        rec_lig_data.populate('data/molportFULL_rand_test0_1000.types')
        (rec_grids, lig_grids), labels = rec_lig_data.forward()
        assert rec_grids.shape == (10, 16, 48, 48, 48)
        assert lig_grids.shape == (10, 19, 48, 48, 48)
        assert not isclose(0, rec_grids.norm().cpu())
        assert not isclose(0, lig_grids.norm().cpu())

    def test_rec_lig_data_forward_empty(self, rec_lig_data):
        with pytest.raises(AssertionError):
            rec_lig_data.forward()
