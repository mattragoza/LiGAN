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
    def lig_data(self):
        return AtomGridData(
            data_root='data/molport',
            batch_size=10,
            rec_map_file='data/my_rec_map',
            lig_map_file='data/my_lig_map',
            resolution=0.5,
            dimension=23.5,
            shuffle=False,
            ligand_only=True,
        )

    @pytest.fixture
    def split_data(self):
        return AtomGridData(
            data_root='data/molport',
            batch_size=10,
            rec_map_file='data/my_rec_map',
            lig_map_file='data/my_lig_map',
            resolution=0.5,
            dimension=23.5,
            shuffle=False,
            split_rec_lig=True,
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

    def test_data_init(self, data):
        assert data.n_rec_channels == 16
        assert data.n_lig_channels == 19
        assert data.ex_provider
        assert data.grid_maker
        assert data.grids.shape == (10, 16+19, 48, 48, 48)
        assert isclose(0, data.grids.norm().cpu())
        assert len(data) == 0

    def test_data_from_param(self, param):
        data = AtomGridData.from_param(param)
        assert data.n_rec_channels == 16
        assert data.n_lig_channels == 19
        assert data.ex_provider
        assert data.grid_maker
        assert data.grids.shape == (10, 16+19, 48, 48, 48)
        assert isclose(0, data.grids.norm().cpu())
        assert len(data) == 0

    def test_data_populate(self, data):
        data.populate('data/molportFULL_rand_test0_1000.types')
        assert len(data) == 1000

    def test_data_populate2(self, data):
        data.populate('data/molportFULL_rand_test0_1000.types')
        data.populate('data/molportFULL_rand_test0_1000.types')
        assert len(data) == 2000

    def test_data_forward_empty(self, data):
        with pytest.raises(AssertionError):
            data.forward()

    def test_data_forward_ok(self, data):
        data.populate('data/molportFULL_rand_test0_1000.types')
        grids, labels = data.forward()
        assert grids.shape == (10, 16+19, 48, 48, 48)
        assert labels.shape == (10,)
        assert not isclose(0, grids.norm().cpu())
        assert all(labels == 1)

    def test_lig_data_forward_ok(self, lig_data):
        data = lig_data
        data.populate('data/molportFULL_rand_test0_1000.types')
        lig_grids, labels = data.forward()
        assert lig_grids.shape == (10, 19, 48, 48, 48)
        assert labels.shape == (10,)
        assert not isclose(0, lig_grids.norm().cpu())
        assert all(labels == 1)

    def test_split_data_forward_ok(self, split_data):
        data = split_data
        data.populate('data/molportFULL_rand_test0_1000.types')
        (rec_grids, lig_grids), labels = data.forward()
        assert rec_grids.shape == (10, 16, 48, 48, 48)
        assert lig_grids.shape == (10, 19, 48, 48, 48)
        assert labels.shape == (10,)
        assert not isclose(0, rec_grids.norm().cpu())
        assert not isclose(0, lig_grids.norm().cpu())
        assert all(labels == 1)

