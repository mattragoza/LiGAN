import sys, os, pytest, time
from numpy import isclose, isnan
import pandas as pd
from torch import optim
pd.set_option('display.width', 180)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

sys.path.insert(0, '.')
import liGAN
from liGAN import models, inference


@pytest.fixture(params=[
    'CVAE',
    #'CVAE2',
    #'AE', 'CE', 'VAE', 'CVAE', 'GAN', 'CGAN',
    #'VAEGAN', 'CVAEGAN', 'VAE2', 'CVAE2'
])
def generator(request):
    generator_type = getattr(liGAN.inference, request.param + 'Generator')
    return generator_type(
        data_kws=dict(
            data_file='data/it2_tt_0_lowrmsd_valid_mols_head1.types',
            data_root='data/crossdock2020',
            batch_size=1,
            rec_typer='on-1',
            lig_typer='on-1',
            resolution=1.0,
            grid_size=16,
            shuffle=False,
            random_rotation=True,
            random_translation=0,
            diff_cond_transform=False,
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
        output_kws=dict(),
        n_samples=1,
        fit_atoms=True,
        diff_cond_rec=False,
        out_prefix='tests/output/TEST_' + request.param,
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
        assert isinstance(generator.out_writer, liGAN.inference.OutputWriter)
        assert isinstance(generator.out_writer.metrics, pd.DataFrame)
        assert len(generator.out_writer.metrics) == 0

    ### TEST FORWARD

    def test_gen_forward(self, generator):
        input_grids, cond_grids, input_structs, cond_structs, _, _, _ = \
            generator.forward(
                prior=False,
                stage2=False,
                fixed_input=False,
                fixed_condition=False,
            )
        input_rec_grids, input_lig_grids = input_grids
        cond_rec_grids, cond_lig_grids = cond_grids
        input_rec_structs, input_lig_structs = input_structs
        cond_rec_structs, cond_lig_structs = cond_structs

        assert input_rec_grids.norm() > 0, 'rec grids are zero'
        assert input_lig_grids.norm() > 0, 'lig grids are zero'
        assert input_rec_structs == cond_rec_structs, \
            'input and cond rec structs are different'
        assert input_lig_structs == cond_lig_structs, \
            'input and cond lig structs are different'
        assert (input_rec_grids == cond_rec_grids).all(), \
            'input and cond rec grids are different'
        assert (input_lig_grids == cond_lig_grids).all(), \
            'input and cond lig grids are different'

    def test_gen_forward_not_fixed(self, generator):
        input_grids, cond_grids, input_structs, cond_structs, _, _, _ = \
            generator.forward(
                prior=False,
                stage2=False,
                fixed_input=False,
                fixed_condition=False,
            )
        input_rec_grids0, input_lig_grids0 = input_grids
        cond_rec_grids0, cond_lig_grids0 = cond_grids
        input_rec_structs0, input_lig_structs0 = input_structs
        cond_rec_structs0, cond_lig_structs0 = cond_structs

        input_grids, cond_grids, input_structs, cond_structs, _, _, _ = \
            generator.forward(
                prior=False,
                stage2=False,
                fixed_input=False,
                fixed_condition=False,
            )
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

    def test_gen_forward_fixed_input(self, generator):
        input_grids, cond_grids, input_structs, cond_structs, _, _, _ = \
            generator.forward(
                prior=False,
                stage2=False,
                fixed_input=True,
                fixed_condition=False,
            )
        input_rec_grids0, input_lig_grids0 = input_grids
        cond_rec_grids0, cond_lig_grids0 = cond_grids
        input_rec_structs0, input_lig_structs0 = input_structs
        cond_rec_structs0, cond_lig_structs0 = cond_structs

        input_grids, cond_grids, input_structs, cond_structs, _, _, _ = \
            generator.forward(
                prior=False,
                stage2=False,
                fixed_input=True,
                fixed_condition=False,
            )
        input_rec_grids1, input_lig_grids1 = input_grids
        cond_rec_grids1, cond_lig_grids1 = cond_grids
        input_rec_structs1, input_lig_structs1 = input_structs
        cond_rec_structs1, cond_lig_structs1 = cond_structs

        assert input_rec_structs1 == input_rec_structs0, \
            'input rec structs are different'
        assert input_lig_structs1 == input_lig_structs0, \
            'input lig structs are different'
        assert cond_rec_structs1 != cond_rec_structs0, \
            'cond rec structs are the same'
        assert cond_lig_structs1 != cond_lig_structs0, \
            'cond lig structs are the same'
        assert (input_rec_grids1 == input_rec_grids0).all(), \
            'input rec grids are different'
        assert (input_lig_grids1 == input_lig_grids0).all(), \
            'input lig grids are different'
        assert (cond_rec_grids1 != cond_rec_grids0).any(), \
            'cond rec grids are the same'
        assert (cond_lig_grids1 != cond_lig_grids0).any(), \
            'cond lig grids are the same'

    def test_gen_forward_fixed_condition(self, generator):
        input_grids, cond_grids, input_structs, cond_structs, _, _, _ = \
            generator.forward(
                prior=False,
                stage2=False,
                fixed_input=False,
                fixed_condition=True,
            )
        input_rec_grids0, input_lig_grids0 = input_grids
        cond_rec_grids0, cond_lig_grids0 = cond_grids
        input_rec_structs0, input_lig_structs0 = input_structs
        cond_rec_structs0, cond_lig_structs0 = cond_structs

        input_grids, cond_grids, input_structs, cond_structs, _, _, _ = \
            generator.forward(
                prior=False,
                stage2=False,
                fixed_input=False,
                fixed_condition=True,
            )
        input_rec_grids1, input_lig_grids1 = input_grids
        cond_rec_grids1, cond_lig_grids1 = cond_grids
        input_rec_structs1, input_lig_structs1 = input_structs
        cond_rec_structs1, cond_lig_structs1 = cond_structs

        assert input_rec_structs1 != input_rec_structs0, \
            'input rec structs are the same'
        assert input_lig_structs1 != input_lig_structs0, \
            'input lig structs are the same'
        assert cond_rec_structs1 == cond_rec_structs0, \
            'cond rec structs are different'
        assert cond_lig_structs1 == cond_lig_structs0, \
            'cond lig structs are different'
        assert (input_rec_grids1 != input_rec_grids0).any(), \
            'input rec grids are the same'
        assert (input_lig_grids1 != input_lig_grids0).any(), \
            'input lig grids are different'
        assert (cond_rec_grids1 == cond_rec_grids0).all(), \
            'cond rec grids are different'
        assert (cond_lig_grids1 == cond_lig_grids0).all

    def test_gen_forward_fixed_both(self, generator):
        input_grids, cond_grids, input_structs, cond_structs, _, _, _ = \
            generator.forward(
                prior=False,
                stage2=False,
                fixed_input=True,
                fixed_condition=True,
            )
        input_rec_grids0, input_lig_grids0 = input_grids
        cond_rec_grids0, cond_lig_grids0 = cond_grids
        input_rec_structs0, input_lig_structs0 = input_structs
        cond_rec_structs0, cond_lig_structs0 = cond_structs

        input_grids, cond_grids, input_structs, cond_structs, _, _, _ = \
            generator.forward(
                prior=False,
                stage2=False,
                fixed_input=True,
                fixed_condition=True,
            )
        input_rec_grids1, input_lig_grids1 = input_grids
        cond_rec_grids1, cond_lig_grids1 = cond_grids
        input_rec_structs1, input_lig_structs1 = input_structs
        cond_rec_structs1, cond_lig_structs1 = cond_structs

        assert input_rec_structs1 == input_rec_structs0, \
            'input rec structs are different'
        assert input_lig_structs1 == input_lig_structs0, \
            'input lig structs are different'
        assert cond_rec_structs1 == cond_rec_structs0, \
            'cond rec structs are different'
        assert cond_lig_structs1 == cond_lig_structs0, \
            'cond lig structs are different'
        assert (input_rec_grids1 == input_rec_grids0).all(), \
            'input rec grids are different'
        assert (input_lig_grids1 == input_lig_grids0).all(), \
            'input lig grids are different'
        assert (cond_rec_grids1 == cond_rec_grids0).all(), \
            'cond rec grids are different'
        assert (cond_lig_grids1 == cond_lig_grids0).all(), \
            'cond lig grids are different'
