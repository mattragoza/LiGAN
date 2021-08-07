import sys, os, pytest, time, torch
from numpy import isclose
os.environ['GLOG_minloglevel'] = '1'

sys.path.insert(0, '.')
from liGAN.data import molgrid, MolDataset, AtomGridData
from liGAN.atom_types import AtomTyper
from liGAN.atom_structs import AtomStruct


batch_size = 10


class TestAtomGridData(object):

    @pytest.fixture
    def data(self):
        return AtomGridData(
            data_root=os.environ['CROSSDOCK_ROOT'],
            batch_size=batch_size,
            rec_typer='oadc-1.0',
            lig_typer='oadc-1.0',
            resolution=0.5,
            dimension=23.5,
            shuffle=True,
            debug=True,
            cache_structs=True,
        )

    @pytest.fixture
    def data2(self):
        return AtomGridData(
            data_root='tests/input',
            batch_size=2,
            rec_typer='oadc-1.0',
            lig_typer='oadc-1.0',
            resolution=0.5,
            dimension=23.5,
            shuffle=False,
            debug=False,
            diff_cond_transform=True,
        )

    @pytest.fixture
    def data_file(self):
        return 'data/it2_tt_0_lowrmsd_valid_mols_test0_1000.types'

    @pytest.fixture
    def data2_file(self):
        return 'data/two_atoms.types'

    def test_data_init(self, data):
        assert data.n_rec_channels == (data.rec_typer.n_types if data.rec_typer else 0)
        assert data.n_lig_channels == data.lig_typer.n_types
        assert data.ex_provider
        assert data.grid_maker
        assert len(data) == 0

    def test_data_populate(self, data, data_file):
        data.populate(data_file)
        assert len(data) == 1000

    def test_data_populate2(self, data, data_file):
        data.populate(data_file)
        data.populate(data_file)
        assert len(data) == 2000

    def test_data_find_real_mol(self, data, data_file):
        data.populate(data_file)
        return
        for ex in data.ex_provider.next_batch(16):
            rec_src = ex.coord_sets[0].src
            lig_src = ex.coord_sets[1].src
            rec_file, rec_name, rec_idx = data.find_real_mol(rec_src, '.pdb')
            lig_file, lig_name, lig_idx = data.find_real_mol(lig_src, '.sdf')
            assert os.path.isfile(rec_file), rec_file
            assert os.path.isfile(lig_file), lig_file

    def test_data_forward_empty(self, data):
        with pytest.raises(AssertionError):
            data.forward()

    def test_data_forward_ok(self, data, data_file):
        data.populate(data_file)
        input_grids, cond_grids, input_structs, cond_structs, transforms, labels = data.forward()
        rec_structs, lig_structs = input_structs
        assert input_grids.shape == \
            (batch_size, data.n_channels) + (data.grid_size,)*3
        assert len(lig_structs) == batch_size
        assert labels.shape == (batch_size,)
        assert not isclose(0, input_grids.norm().cpu())
        assert all(labels == 1)

    def test_data_forward_split(self, data, data_file):
        data.populate(data_file)
        input_grids, cond_grids, input_structs, cond_structs, transforms, labels = data.forward()
        rec_grids, lig_grids = data.split_channels(input_grids)
        rec_structs, lig_structs = input_structs
        assert rec_grids.shape == (batch_size, data.n_lig_channels) + (data.grid_size,)*3
        assert lig_grids.shape == (batch_size, data.n_rec_channels) + (data.grid_size,)*3
        assert len(lig_structs) == batch_size
        assert labels.shape == (batch_size,)
        assert not isclose(0, rec_grids.norm().cpu())
        assert not isclose(0, lig_grids.norm().cpu())
        assert all(labels == 1)

    def test_data_no_transform(self, data2, data2_file):
        data2.populate(data2_file)
        data2.random_rotation = False
        data2.random_translation = 0.0
        n_trials = 100

        diff = 0.0
        for i in range(n_trials):
            input_grids, cond_grids = data2.forward()[:2]
            assert input_grids.norm() > 0, 'initial grids are empty'
            diff += (input_grids[0] - input_grids[1]).abs().max()

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
            input_grids, cond_grids = data2.forward()[:2]
            assert input_grids.norm() > 0, 'rotated grids are empty'
            diff += (input_grids[0] - input_grids[1]).abs().max()

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
            input_grids, cond_grids = data2.forward()[:2]
            assert input_grids.norm() > 0, 'translated grids are empty'
            diff += (input_grids[0] - input_grids[1]).abs().max()

        diff /= n_trials
        assert diff > 0.5, \
            'translated grids are the same ({:.2f})'.format(diff)

    def test_data_diff_cond_transform(self, data2, data2_file):
        data2.populate(data2_file)
        data2.random_rotation = True
        data2.random_translation = 2.0
        n_trials = 100

        diff = 0.0
        for i in range(n_trials):
            input_grids, cond_grids = data2.forward()[:2]
            assert input_grids.norm() > 0, 'input grids are empty'
            assert cond_grids.norm() > 0, 'conditional grids are empty'
            diff += (input_grids[0] - cond_grids[0]).abs().max()
            diff += (input_grids[1] - cond_grids[1]).abs().max()

        diff /= (2*n_trials)
        assert diff > 0.5, \
            'input and conditional grids are the same ({:.2f})'.format(diff)

    def test_data_consecutive(self, data2, data2_file):
        data2.populate(data2_file)
        data2.random_rotation = True
        data2.random_translation = 2.0
        n_trials = 100

        diff = 0.0
        for i in range(n_trials):
            input_grids = data2.forward()[0]
            assert input_grids.norm() > 0, 'input grids are empty'
            if i > 0:
                diff += (input_grids[0] - last_input_grids[0]).abs().max()
                diff += (input_grids[1] - last_input_grids[1]).abs().max()
            last_input_grids = input_grids

        diff /= (2*(n_trials-1))
        assert diff > 0.5, \
            'consecutive input grids are the same ({:.2f})'.format(diff)

    def test_data_benchmark(self, data, data_file):
        data.populate(data_file)
        n_trials = 100

        t0 = time.time()
        for i in range(n_trials):
            data.forward()

        t_delta = time.time() - t0
        t_delta /= n_trials
        assert t_delta < 1, 'too slow ({:.2f}s / batch)'.format(t_delta)


class TestMolDataset(object):

    @pytest.fixture(params=[False, True])
    def data(self, request):
        use_dataset = request.param

        data_root = os.environ['CROSSDOCK_ROOT']
        data_file = 'data/it2_tt_0_lowrmsd_valid_mols_test0_100.types'

        lig_typer = AtomTyper.get_typer('oadc', 1.0, rec=False)
        rec_typer = AtomTyper.get_typer('oadc', 1.0, rec=True)

        if use_dataset:
            data = MolDataset(
                rec_typer, lig_typer,
                data_root=data_root,
                data_file=data_file,
                verbose=True,
            )
            #data = torch.utils.data.DataLoader(
            #    data, batch_size=batch_size, collate_fn=list, num_workers=0
            #)
            #data.rec_typer = rec_typer
            #data.lig_typer = lig_typer
            return data
        else:
            data = molgrid.ExampleProvider(
                rec_typer, lig_typer,
                data_root=data_root,
                cache_structs=True,
            )
            data.root_dir = data_root
            data.populate(data_file)
            data.rec_typer = rec_typer
            data.lig_typer = lig_typer
            return data

    def test_benchmark(self, data):
        
        t_start = time.time()
        if isinstance(data, molgrid.ExampleProvider):
            n_rows = data.size()
            i = 0
            while i < n_rows:
                examples = data.next_batch(batch_size)
                for ex in examples:
                    rec_coord_set, lig_coord_set = ex.coord_sets
                    rec_struct = AtomStruct.from_coord_set(
                        rec_coord_set, data.rec_typer, data.root_dir
                    )
                    lig_struct = AtomStruct.from_coord_set(
                        lig_coord_set, data.lig_typer, data.root_dir
                    )
                i += batch_size

        elif isinstance(data, MolDataset):
            n_rows = len(data)
            for rec_mol, lig_mol in data:
                rec_struct = data.rec_typer.make_struct(rec_mol)
                lig_struct = data.lig_typer.make_struct(lig_mol)
                print(rec_mol.GetTitle(), '\t', lig_mol.GetTitle())

        else: # data loader
            n_rows = len(data)
            i = 0
            for batch in data:
                for rec_mol, lig_mol in batch:
                    rec_struct = data.rec_typer.make_struct(rec_mol)
                    lig_struct = data.lig_typer.make_struct(lig_mol)
                    print(rec_mol.GetTitle(), '\t', lig_mol.GetTitle())
                i += batch_size
                if i > n_rows:
                    break

        t_delta = time.time() - t_start
        t_delta /= n_rows
        assert t_delta < 0.001, 'too slow ({:.4f}s / row)'.format(t_delta)
