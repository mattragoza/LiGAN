import sys, os, pytest, time, torch
from numpy import isclose
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
            debug=True,
        )

    @pytest.fixture
    def data2(self):
        return AtomGridData(
            data_root='data',
            batch_size=2,
            rec_typer='on-c',
            lig_typer='on-c',
            resolution=0.5,
            dimension=23.5,
            shuffle=False,
            debug=False,
        )

    @pytest.fixture
    def data_file(self):
        return 'data/molportFULL_rand_test0_1000.types'

    @pytest.fixture
    def data2_file(self):
        return 'data/two_atoms.types'

    def test_data_init(self, data):
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

    def test_data_benchmark(self, data, data_file):
        data.populate(data_file)
        t0 = time.time()
        for i in range(10):
            data.forward()
        t_delta = time.time() - t0
        assert t_delta < 1, 'too slow ({:.2f}s)'.format(t_delta)

    def test_data_no_transform(self, data2, data2_file):
        data2.populate(data2_file)
        data2.random_rotation = False
        data2.random_translation = 0.0
        n_trials = 100

        diff = 0.0
        for i in range(n_trials):
            grids, _, _ = data2.forward()
            assert grids.norm() > 0, 'initial grids are empty'
            diff += (grids[0] - grids[1]).abs().max()

        diff /= n_trials
        assert diff < 0.1, \
            'no-transform grids are different ({:.2f})'.format(diff)

    def test_data_rand_rotate(self, data2, data2_file):
        data2.populate(data2_file)
        data2.random_rotation = True
        data2.random_translation = 0.0
        n_trials = 100

        diff = 0.0
        for i in range(n_trials):
            grids, _, _ = data2.forward()
            assert grids.norm() > 0, 'rotated grids are empty'
            diff += (grids[0] - grids[1]).abs().max()

        diff /= n_trials
        assert diff > 0.5, \
            'rotated grids are the same ({:.2f})'.format(diff)

    def test_data_rand_translate(self, data2, data2_file):
        data2.populate(data2_file)
        data2.random_rotation = False
        data2.random_translation = 2.0
        n_trials = 100

        diff = 0.0
        for i in range(n_trials):
            grids, _, _ = data2.forward()
            assert grids.norm() > 0, 'translated grids are empty'
            diff += (grids[0] - grids[1]).abs().max()

        diff /= n_trials
        assert diff > 0.5, \
            'translated grids are the same ({:.2f})'.format(diff)
