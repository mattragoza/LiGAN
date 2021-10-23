import sys, os, pytest, time
from numpy import isclose, isnan
import pandas as pd
from torch import optim
pd.set_option('display.width', 180)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

sys.path.insert(0, '.')
import liGAN
from liGAN import models, generating


@pytest.fixture(params=[
    'data/test_pockets/AROK_MYCTU_1_176_0/fixed_input_1zyu_A_rec_mutants.types',
    'data/test_pockets/AROK_MYCTU_1_176_0/fixed_cond_1zyu_A_rec_mutants.types',
    #'data/test_pockets/AROK_MYCTU_1_176_0/fixed_input_no_rec_mutants.types',
    #'data/test_pockets/AROK_MYCTU_1_176_0/fixed_cond_no_rec_mutants.types',
])
def generator(request):
    data_file = request.param
    generator_name, diff_cond_transform, diff_cond_structs = 'CVAE', 0, 1
    generator_type = getattr(liGAN.generating, generator_name + 'Generator')
    return generator_type(
        data_kws=dict(
            data_file=data_file,
            data_root='data/crossdock2020',
            batch_size=1,
            rec_typer='oadc-1.0',
            lig_typer='oadc-1.0',
            resolution=1.0,
            grid_size=16,
            shuffle=False,
            random_rotation=True,
            random_translation=0,
            diff_cond_transform=diff_cond_transform,
            diff_cond_structs=diff_cond_structs,
            cache_structs=False,
        ),
        gen_model_kws=dict(
            n_filters=8,
            n_levels=4,
            conv_per_level=1,
            spectral_norm=1,
            n_latent=128,
            init_conv_pool=False,
            skip_connect=generator_type.gen_model_type.has_conditional_encoder,
        ),
        prior_model_kws=dict(
            n_h_layers=1,
            n_h_units=96,
            n_latent=64,
        ),
        atom_fitting_kws=dict(),
        bond_adding_kws=dict(),
        output_kws=dict(batch_metrics=False),
        n_samples=5,
        fit_atoms=True,
        out_prefix=f'tests/output/TEST_{generator_name}',
        device='cuda',
        debug=True
    )

class TestGenerator(object):

    ### TEST INITIALIZE

    def test_generator_init(self, generator):

        assert type(generator.gen_model) == type(generator).gen_model_type
        generator_name = type(generator).__name__
        assert generator.has_complex_input == ('CVAE' in generator_name)

        assert isinstance(generator.data, liGAN.data.AtomGridData)
        assert isinstance(generator.atom_fitter, liGAN.atom_fitting.AtomFitter)
        assert isinstance(generator.bond_adder, liGAN.bond_adding.BondAdder)
        assert isinstance(generator.out_writer, liGAN.generating.OutputWriter)
        assert isinstance(generator.out_writer.metrics, pd.DataFrame)
        assert len(generator.out_writer.metrics) == 0
        if generator.data.diff_cond_structs or generator.data.diff_cond_transform:
            assert generator.out_writer.grid_types > {'cond_rec', 'cond_lig'}

    ### TEST FORWARD

    def test_gen_forward(self, generator):
        (
            input_grids, cond_grids, input_structs, cond_structs,
            latents, lig_gen_grids, transforms
        ) = generator.forward(prior=False, stage2=False)
        input_rec_grids, input_lig_grids = input_grids
        cond_rec_grids, cond_lig_grids = cond_grids
        input_rec_structs, input_lig_structs = input_structs
        cond_rec_structs, cond_lig_structs = cond_structs
        input_transforms, cond_transforms = transforms

        if generator.data.diff_cond_structs:
            assert cond_rec_structs != input_rec_structs
            assert cond_lig_structs != input_lig_structs
        else:
            assert cond_rec_structs == input_rec_structs
            assert cond_lig_structs == input_lig_structs

        if generator.data.diff_cond_transform:
            assert cond_transforms != input_transforms
        else:
            assert cond_transforms == input_transforms

        assert input_rec_grids.norm() > 0, 'input rec grids are zero'
        assert input_lig_grids.norm() > 0, 'input lig grids are zero'
        assert cond_rec_grids.norm() > 0, 'cond rec grids are zero'
        assert cond_lig_grids.norm() > 0, 'cond lig grids are zero'
        assert latents.norm() > 0, 'latent vecs are zero'
        assert lig_gen_grids.norm() > 0, 'lig gen grids are zero'

    def test_gen_forward2(self, generator):
        input_grids, cond_grids, input_structs, cond_structs, _, _, _ = \
            generator.forward(prior=False, stage2=False)
        input_rec_grids0, input_lig_grids0 = input_grids
        cond_rec_grids0, cond_lig_grids0 = cond_grids
        input_rec_structs0, input_lig_structs0 = input_structs
        cond_rec_structs0, cond_lig_structs0 = cond_structs

        input_grids, cond_grids, input_structs, cond_structs, _, _, _ = \
            generator.forward(prior=False, stage2=False)
        input_rec_grids1, input_lig_grids1 = input_grids
        cond_rec_grids1, cond_lig_grids1 = cond_grids
        input_rec_structs1, input_lig_structs1 = input_structs
        cond_rec_structs1, cond_lig_structs1 = cond_structs

        assert input_rec_structs1 != input_rec_structs0, \
            'input rec structs are the same'
        assert input_lig_structs1 != input_lig_structs0, \
            'input lig structs are the same'
        assert cond_rec_structs1 != cond_rec_structs0, \
            'cond rec structs are the same'
        assert cond_lig_structs1 != cond_lig_structs0, \
            'cond lig structs are the same'
        assert (input_rec_grids1 != input_rec_grids0).any(), \
            'input rec grids are the same'
        assert (input_lig_grids1 != input_lig_grids0).any(), \
            'input lig grids are the same'
        assert (cond_rec_grids1 != cond_rec_grids0).any(), \
            'cond rec grids are the same'
        assert (cond_lig_grids1 != cond_lig_grids0).any(), \
            'cond lig grids are the same'

    def test_generate(self, generator):
        m = generator.generate(n_examples=1, n_samples=5)
        print(generator.out_writer.grid_types)
        assert len(m) > 0, 'empty metrics'
        m = m.reset_index().set_index(['example_idx', 'sample_idx'])
        print(m)
        if generator.data.diff_cond_structs:
            rec_prod = m['lig_gen_rec_prod']
            cond_rec_prod = m['lig_gen_cond_rec_prod']
            print((cond_rec_prod-rec_prod).describe())
            assert False, 'OK'
