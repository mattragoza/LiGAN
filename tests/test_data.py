import sys, os, pytest
from numpy import isclose
from caffe.proto import caffe_pb2
os.environ['GLOG_minloglevel'] = '1'

sys.path.insert(0, '.')
from liGAN.data import AtomGridData
from liGAN.atom_types import AtomTyper


class TestAtomGridData(object):

    @pytest.fixture
    def data(self):
        return AtomGridData(
            data_root='data/molport',
            batch_size=10,
            rec_typer='on-c',
            lig_typer='on-c',
            resolution=0.5,
            dimension=23.5,
            shuffle=False,
        )

    @pytest.fixture
    def data_param(self):
        param = caffe_pb2.MolGridDataParameter()
        param.root_folder = 'data/molport'
        param.batch_size = 10
        param.recmap = 'on-c'
        param.ligmap = 'on-c'
        param.resolution = 0.5
        param.dimension = 23.5
        return param

    @pytest.fixture
    def data_file(self):
        return 'data/molportFULL_rand_test0_1000.types'

    def test_data_init(self, data):
        assert data.n_rec_channels == (data.rec_typer.n_types if data.rec_typer else 0)
        assert data.n_lig_channels == data.lig_typer.n_types
        assert data.ex_provider
        assert data.grid_maker
        assert data.grids.shape == (10, data.n_channels) + (data.grid_size,)*3
        assert isclose(0, data.grids.norm().cpu())
        assert len(data) == 0

    def test_data_from_param(self, data_param):
        data = AtomGridData.from_param(data_param)
        assert data.n_rec_channels == (data.rec_typer.n_types if data.rec_typer else 0)
        assert data.n_lig_channels == data.lig_typer.n_types
        assert data.ex_provider
        assert data.grid_maker
        assert data.grids.shape == (10, data.n_channels) + (data.grid_size,)*3
        assert isclose(0, data.grids.norm().cpu())
        assert len(data) == 0

    def test_data_populate(self, data, data_file):
        data.populate(data_file)
        assert len(data) == 1000

    def test_data_populate2(self, data, data_file):
        data.populate(data_file)
        data.populate(data_file)
        assert len(data) == 2000

    def test_data_forward_empty(self, data):
        with pytest.raises(AssertionError):
            data.forward()

    def test_data_forward_ok(self, data, data_file):
        data.populate(data_file)
        grids, structs, labels = data.forward()
        rec_structs, lig_structs = structs
        assert grids.shape == (10, data.n_channels, 48, 48, 48)
        assert len(lig_structs) == 10
        assert labels.shape == (10,)
        assert not isclose(0, grids.norm().cpu())
        assert all(labels == 1)

    def test_data_forward_ligs(self, data, data_file):
        data.populate(data_file)
        lig_grids, lig_structs, labels = data.forward(ligand_only=True)
        assert lig_grids.shape == (10, data.n_lig_channels, 48, 48, 48)
        assert len(lig_structs) == 10
        assert labels.shape == (10,)
        assert not isclose(0, lig_grids.norm().cpu())
        assert all(labels == 1)

    def test_data_forward_split(self, data, data_file):
        data.populate(data_file)
        grids, structs, labels = data.forward(split_rec_lig=True)
        rec_grids, lig_grids = grids
        rec_structs, lig_structs = structs
        assert rec_grids.shape == (10, data.n_lig_channels, 48, 48, 48)
        assert lig_grids.shape == (10, data.n_rec_channels, 48, 48, 48)
        assert len(lig_structs) == 10
        assert labels.shape == (10,)
        assert not isclose(0, rec_grids.norm().cpu())
        assert not isclose(0, lig_grids.norm().cpu())
        assert all(labels == 1)
