import sys, os, pytest, time
from numpy import isclose, isnan
import pandas as pd
from torch import optim
pd.set_option('display.width', 180)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

sys.path.insert(0, '.')
import liGAN
from liGAN import models, training
from liGAN.models import compute_grad_norm as param_grad_norm


def param_norm(model):
    '''
    Compute the L2 norm of model parameters.
    '''
    norm2 = 0
    for p in model.parameters():
        norm2 += (p.data**2).sum().item()
    return norm2**(1/2)


@pytest.fixture
def train_params():
    return dict(
        max_iter=10,
        test_interval=10,
        n_test_batches=1,
        fit_interval=0,
        norm_interval=10,
        save_interval=10,
    )


@pytest.fixture(params=[
    #'CVAE', 'CVAE2',
    'AE', 'CE', 'VAE', 'CVAE', 'GAN', 'CGAN',
    'VAEGAN', 'CVAEGAN', 'VAE2', 'CVAE2'
])
def solver(request):
    solver_type = getattr(liGAN.training, request.param + 'Solver')
    return solver_type(
        data_kws=dict(
            train_file='data/it2_tt_0_lowrmsd_valid_mols_head1.types',
            test_file='data/it2_tt_0_lowrmsd_valid_mols_head1.types',
            data_root='data/crossdock2020',
            batch_size=1,
            rec_typer='oadc-1.0',
            lig_typer='oadc-1.0',
            resolution=1.0,
            grid_size=16,
            shuffle=False,
            random_rotation=True,
            random_translation=0,
            diff_cond_transform=True,
            cache_structs=False,
        ),
        gen_model_kws=dict(
            n_filters=8,
            n_levels=4,
            conv_per_level=1,
            spectral_norm=1,
            n_latent=128,
            init_conv_pool=False,
            skip_connect=solver_type.gen_model_type.has_conditional_encoder,
        ),
        disc_model_kws=dict(
            n_filters=8,
            n_levels=4,
            conv_per_level=1,
            spectral_norm=1,
            n_output=1,
        ),
        prior_model_kws=dict(
            n_h_layers=1,
            n_h_units=96,
            n_latent=64,
        ),
        loss_fn_kws=dict(
            types=dict(recon_loss='2', gan_loss='w'),
            weights=dict(
                kldiv_loss=1.0,
                recon_loss=1.0,
                gan_loss=1.0 * solver_type.has_disc_model,
                steric_loss=1.0 * solver_type.gen_model_type.has_conditional_encoder,
                kldiv2_loss=1.0 * solver_type.has_prior_model,
                recon2_loss=1.0 * solver_type.has_prior_model,
            ),
            learn_recon_var=True,
        ),
        gen_optim_kws=dict(
            type='RMSprop',
            lr=1e-5,
            n_train_iters=1,
        ),
        disc_optim_kws=dict(
            type='RMSprop',
            lr=5e-5,
            n_train_iters=2,
        ),
        prior_optim_kws=dict(
            type='RMSprop',
            lr=1e-5,
        ),
        atom_fitting_kws=dict(),
        bond_adding_kws=dict(),
        out_prefix='tests/output/TEST_' + request.param,
        device='cuda',
        debug=True
    )


def check_grad(model, expect_grad, name):
    if expect_grad:
        assert param_grad_norm(model) > 0, name + ' grad is zero'
    else:
        assert param_grad_norm(model) == 0, name + ' grad not zero'


def check_solver_grad(
    solver,
    expect_disc_grad,
    expect_dec_grad,
    expect_inp_enc_grad,
    expect_cond_enc_grad,
    expect_prior_grad,
):
    if solver.has_disc_model:
        check_grad(solver.disc_model, expect_disc_grad, 'disc')

    check_grad(solver.gen_model.decoder, expect_dec_grad, 'decoder')

    if solver.gen_model.has_input_encoder:
        check_grad(
            solver.gen_model.input_encoder,
            expect_inp_enc_grad,
            name='input encoder'
        )

    if solver.gen_model.has_conditional_encoder:
        check_grad(
            solver.gen_model.conditional_encoder,
            expect_cond_enc_grad,
            name='cond encoder'
        )

    if solver.has_prior_model:
        check_grad(solver.prior_model, expect_prior_grad, 'prior')


class TestGenerativeSolver(object):

    ### TEST INITIALIZE

    def test_solver_init(self, solver):

        assert type(solver.gen_model) == type(solver).gen_model_type
        assert solver.gen_iter == 0

        solver_name = type(solver).__name__
        assert solver.has_complex_input == ('CVAE' in solver_name)

        assert solver.has_disc_model == ('GAN' in solver_name)
        if solver_name.endswith('GAN'):
            assert type(solver.disc_model) == models.Discriminator
            assert solver.disc_iter == 0

        assert isinstance(solver.train_data, liGAN.data.AtomGridData)
        assert isinstance(solver.test_data, liGAN.data.AtomGridData)
        assert isinstance(solver.loss_fn, liGAN.loss_fns.LossFunction)
        assert isinstance(solver.atom_fitter, liGAN.atom_fitting.AtomFitter)
        assert isinstance(solver.bond_adder, liGAN.bond_adding.BondAdder)
        assert isinstance(solver.metrics, pd.DataFrame)
        assert len(solver.metrics) == 0

        for name, params in solver.named_parameters():
            if name.endswith('weight'):
                assert params.detach().norm().cpu() > 0, 'weights are zero'
            elif name.endswith('bias'):
                pass # ok for bias to be zero

    ### TEST GRID PHASE

    def test_solver_gen_phases(self, solver):
        phases = [solver.get_gen_grid_phase(i) for i in range(6)]
        assert phases, phases

    def test_solver_disc_phases(self, solver):
        if solver.has_disc_model:
            phases = [solver.get_disc_grid_phase(i) for i in range(6)]
            assert phases, phases

    ### TEST FORWARD PASS

    def test_solver_gen_forward_poster(self, solver):

        if solver.has_posterior_phase:
            data = solver.train_data
            loss, metrics = solver.gen_forward(data, grid_type='poster')
            for k, v in metrics.items():
                print(k, v)
            assert loss.item() != 0, 'loss is zero'

    def test_solver_gen_forward_prior(self, solver):

        if solver.has_prior_phase:
            data = solver.train_data
            loss, metrics = solver.gen_forward(data, grid_type='prior')
            for k, v in metrics.items():
                print(k, v)
            print(solver.loss_fn.has_prior_loss)
            assert loss.item() != 0 or not solver.loss_fn.has_prior_loss, \
                'loss is zero'

    def test_solver_gen_forward_poster2(self, solver):

        if solver.has_prior_model:
            data = solver.train_data
            loss, metrics = solver.gen_forward(data, grid_type='poster2')
            for k, v in metrics.items():
                print(k, v)
            assert loss.item() != 0, 'loss is zero'

    def test_solver_gen_forward_prior2(self, solver):

        if solver.has_prior_model:
            data = solver.train_data
            loss, metrics = solver.gen_forward(data, grid_type='prior2')
            for k, v in metrics.items():
                print(k, v)
            print(solver.loss_fn.has_prior_loss)
            assert loss.item() != 0 or not solver.loss_fn.has_prior_loss, \
                'loss is zero'

    def test_solver_disc_forward_real(self, solver):

        if solver.has_disc_model:
            data = solver.train_data
            loss, metrics = solver.disc_forward(data, grid_type='real')
            for k, v in metrics.items():
                print(k, v)
            assert loss.item() != 0, 'loss is zero'

    def test_solver_disc_forward_poster(self, solver):

        if solver.has_disc_model and solver.has_posterior_phase:
            data = solver.train_data
            loss, metrics = solver.disc_forward(data, grid_type='poster')
            for k, v in metrics.items():
                print(k, v)
            assert loss.item() != 0, 'loss is zero'

    def test_solver_disc_forward_prior(self, solver):

        if solver.has_disc_model and solver.has_prior_phase:
            data = solver.train_data
            loss, metrics = solver.disc_forward(data, grid_type='prior')
            for k, v in metrics.items():
                print(k, v)
            assert loss.item() != 0, 'loss is zero'

    ### TEST BACKWARD PASS

    def test_solver_gen_backward_poster(self, solver):

        if solver.has_posterior_phase:
            data = solver.train_data
            loss, metrics = solver.gen_forward(data, grid_type='poster')
            metrics = solver.gen_backward(loss)
            for k, v in metrics.items():
                print(k, v)
            check_solver_grad(solver, True, True, True, True, True)

            if solver.learn_recon_var:
                assert (
                    solver.gen_model.log_recon_var.grad.data**2
                ).sum() > 0, 'no gen log_recon_var gradient'
                if solver.has_prior_model:
                    assert (
                        solver.prior_model.log_recon_var.grad.data**2
                    ).sum() > 0, 'no prior log_recon_var gradient'
            else:
                assert solver.gen_model.log_recon_var.grad is None
                if solver.has_prior_model:
                    assert solver.prior_model.log_recon_var.grad is None

    def test_solver_gen_backward_prior(self, solver):

        if solver.has_prior_phase and solver.loss_fn.has_prior_loss:
            data = solver.train_data
            loss, metrics = solver.gen_forward(data, grid_type='prior')
            metrics = solver.gen_backward(loss)
            for k, v in metrics.items():
                print(k, v)
            check_solver_grad(solver, True, True, False, True, False)

    def test_solver_gen_backward_poster2(self, solver):

        if solver.has_prior_model:
            data = solver.train_data
            loss, metrics = solver.gen_forward(data, grid_type='poster2')
            metrics = solver.gen_backward(loss)
            for k, v in metrics.items():
                print(k, v)
            check_solver_grad(solver, True, True, True, True, True)

    def test_solver_gen_backward_prior2(self, solver):

        if solver.has_prior_model and solver.loss_fn.has_prior_loss:
            data = solver.train_data
            loss, metrics = solver.gen_forward(data, grid_type='prior2')
            metrics = solver.gen_backward(loss)
            for k, v in metrics.items():
                print(k, v)
            check_solver_grad(solver, True, True, False, True, True)

    def test_solver_disc_backward_real(self, solver):

        if solver.has_disc_model:
            data = solver.train_data
            loss, metrics = solver.disc_forward(data, grid_type='real')
            metrics = solver.gen_backward(loss)
            for k, v in metrics.items():
                print(k, v)
            check_solver_grad(solver, True, False, False, False, False)

    def test_solver_disc_backward_poster(self, solver):

        if solver.has_disc_model and solver.has_posterior_phase:
            data = solver.train_data
            loss, metrics = solver.disc_forward(data, grid_type='poster')
            metrics = solver.gen_backward(loss)
            for k, v in metrics.items():
                print(k, v)
            check_solver_grad(solver, True, False, False, False, False)

    def test_solver_disc_backward_prior(self, solver):

        if solver.has_disc_model and solver.has_prior_phase:
            data = solver.train_data
            loss, metrics = solver.disc_forward(data, grid_type='prior')
            metrics = solver.gen_backward(loss)
            for k, v in metrics.items():
                print(k, v)
            check_solver_grad(solver, True, False, False, False, False)

    ### TEST TRAINING STEP

    def test_solver_gen_step_poster(self, solver):
        if solver.has_posterior_phase:
            data = solver.train_data
            liGAN.set_random_seed(0)

            if solver.learn_recon_var:
                gen_log_var0 = solver.gen_model.log_recon_var.item()
                if solver.has_prior_model:
                    prior_log_var0 = solver.prior_model.log_recon_var.item()

            metrics0 = solver.gen_step(grid_type='poster')
            liGAN.set_random_seed(0)
            _, metrics1 = solver.gen_forward(data, grid_type='poster')
            assert metrics1['loss'] < metrics0['loss'], 'loss did not decrease'

            if solver.has_prior_model:
                assert (
                    metrics1['recon2_loss'] + metrics1['kldiv2_loss'] < \
                    metrics0['recon2_loss'] + metrics0['kldiv2_loss']
                ), 'prior model loss did not decrease'

            if solver.learn_recon_var:
                assert (
                    solver.gen_model.log_recon_var != gen_log_var0
                ), 'gen log_recon_var did not change'
                if solver.has_prior_model:
                    assert (
                        solver.prior_model.log_recon_var != prior_log_var0
                    ), 'prior log_recon_var did not change'

    def test_solver_gen_step_prior(self, solver):
        if solver.has_prior_phase and solver.loss_fn.has_prior_loss:
            data = solver.train_data
            liGAN.set_random_seed(0)
            metrics0 = solver.gen_step(grid_type='prior')
            liGAN.set_random_seed(0)
            _, metrics1 = solver.gen_forward(data, grid_type='prior')
            assert metrics1['loss'] < metrics0['loss'], 'loss did not decrease'

    def test_solver_disc_step_real(self, solver):
        if solver.has_disc_model:
            data = solver.train_data
            liGAN.set_random_seed(0)
            metrics0 = solver.disc_step(grid_type='real')
            liGAN.set_random_seed(0)
            _, metrics1 = solver.disc_forward(data, grid_type='real')
            assert metrics1['loss'] < metrics0['loss'], 'loss did not decrease'

    def test_solver_disc_step_poster(self, solver):
        if solver.has_disc_model and solver.has_posterior_phase:
            data = solver.train_data
            liGAN.set_random_seed(0)
            metrics0 = solver.disc_step(grid_type='poster')
            liGAN.set_random_seed(0)
            _, metrics1 = solver.disc_forward(data, grid_type='poster')
            assert metrics1['loss'] < metrics0['loss'], 'loss did not decrease'

    def test_solver_disc_step_prior(self, solver):
        if solver.has_disc_model and solver.has_prior_phase:
            data = solver.train_data
            liGAN.set_random_seed(0)
            metrics0 = solver.disc_step(grid_type='prior')
            liGAN.set_random_seed(0)
            _, metrics1 = solver.disc_forward(data, grid_type='prior')
            assert metrics1['loss'] < metrics0['loss'], 'loss did not decrease'

    ### TEST SOLVER SAVE/LOAD STATE

    def test_solver_state(self, solver):

        assert solver.gen_iter == 0
        assert solver.disc_iter == 0
        assert solver.prior_iter == 0
        init_norm = param_norm(solver)
        solver.save_state()

        if solver.has_posterior_phase:
            solver.gen_step(grid_type='poster')
            if solver.has_disc_model:
                solver.disc_step(grid_type='poster')

        elif solver.has_prior_phase:
            solver.gen_step(grid_type='prior')
            if solver.has_disc_model:
                solver.disc_step(grid_type='prior')

        assert solver.gen_iter == 1
        assert solver.disc_iter == int(solver.has_disc_model)
        assert solver.prior_iter == int(solver.has_prior_model)

        norm = param_norm(solver)
        norm_diff = (norm - init_norm)
        assert abs(norm_diff) > 1e-4, \
            'same params after update ({:.4f})'.format(norm_diff)

        solver.load_state(cont_iter=0)
        assert solver.gen_iter == 0
        assert solver.disc_iter == 0
        assert solver.prior_iter == 0
        norm = param_norm(solver)
        norm_diff = (norm - init_norm)
        assert isclose(norm, init_norm), \
            'different params after load ({:.2f})'.format(norm_diff)

        if solver.learn_recon_var:
            state_dict = solver.state_dict()
            assert 'gen_model.log_recon_var' in state_dict
            if solver.has_prior_model:
                assert 'prior_model.log_recon_var' in state_dict

    ### TEST TESTING MODELS

    def test_solver_test_disc(self, solver):
        assert len(solver.metrics) == 0
        if solver.has_disc_model:
            solver.test_model(n_batches=10, model_type='disc', fit_atoms=False)
            assert solver.disc_iter == 0
            assert len(solver.metrics) == 10

    def test_solver_test_gen(self, solver):
        assert len(solver.metrics) == 0
        solver.test_model(n_batches=10, model_type='gen', fit_atoms=False)
        assert solver.gen_iter == 0
        assert len(solver.metrics) == 10

    def test_solver_test_gen_fit(self, solver):
        assert len(solver.metrics) == 0
        solver.test_model(n_batches=1, model_type='gen', fit_atoms=True)
        assert solver.gen_iter == 0
        assert len(solver.metrics) == 1
        print(solver.metrics.transpose())
        assert 'lig_gen_fit_n_atoms' in solver.metrics.columns
        if solver.has_posterior_phase:
            assert 'lig_gen_fit_type_diff' in solver.metrics.columns

    def test_solver_test_models(self, solver):
        assert len(solver.metrics) == 0
        solver.test_models(n_batches=10, fit_atoms=False)
        assert solver.gen_iter == 0
        assert solver.disc_iter == 0
        assert len(solver.metrics) == 10 + solver.has_disc_model * 10

    ### TEST TRAINING MODELS

    def test_solver_train_gen(self, solver):
        solver.train_model(n_iters=10, model_type='gen')
        assert solver.gen_iter == len(solver.metrics) == 10

    def test_solver_train_disc(self, solver):
        if solver.has_disc_model:
            solver.train_model(n_iters=10, model_type='disc')
            assert solver.disc_iter == len(solver.metrics) == 10

    def test_solver_train_models(self, solver):
        solver.train_models(update=True)
        assert not solver.balance
        assert solver.gen_iter == solver.n_gen_train_iters
        if solver.has_disc_model:
            assert solver.disc_iter == solver.n_disc_train_iters
        assert len(solver.metrics) == (
            solver.n_gen_train_iters + 
            (solver.n_disc_train_iters if solver.has_disc_model else 0)
        )

    def test_solver_train_models_noup(self, solver):
        solver.train_models(update=False)
        assert not solver.balance
        assert solver.gen_iter == 0
        if solver.has_disc_model:
            assert solver.disc_iter == 0
        print(solver.metrics)
        assert len(solver.metrics) == (
            solver.n_gen_train_iters + 
            (solver.n_disc_train_iters if solver.has_disc_model else 0)
        )

    def test_solver_train_and_test(self, solver, train_params):

        max_iter = train_params['max_iter']
        test_interval = train_params['test_interval']
        n_test_batches = train_params['n_test_batches']

        t0 = time.time()
        solver.train_and_test(**train_params)
        t_delta = time.time() - t0

        print(solver.metrics)
        assert solver.gen_iter == max_iter

        m = solver.metrics.reset_index()
        n_train_rows = (max_iter + 1) * (
            solver.n_gen_train_iters + (
                solver.n_disc_train_iters if solver.has_disc_model else 0
            )
        )
        n_test_rows = (max_iter//test_interval + 1) * (
            n_test_batches + (n_test_batches if solver.has_disc_model else 0)
        )
        assert len(m[m['data_phase'] == 'train']) == n_train_rows, 'unexpected num trains'
        assert len(m[m['data_phase'] == 'test']) == n_test_rows, 'unexpected num tests'

        assert os.path.isfile(solver.metrics_file), solver.metrics_file
        assert os.path.isfile(solver.gen_model_state_file), \
            solver.gen_model_state_file
        assert os.path.isfile(solver.gen_solver_state_file), \
            solver.gen_solver_state_file
        if solver.has_disc_model:
            assert os.path.isfile(solver.disc_model_state_file), \
                solver.disc_model_state_file
            assert os.path.isfile(solver.disc_solver_state_file), \
                solver.disc_solver_state_file

        loss_i = m[m['iteration'] == 0]['loss'].mean()
        loss_f = m[m['iteration'] == max_iter]['loss'].mean()
        assert loss_f < loss_i, 'loss did not decrease'

        t_per_iter = t_delta / max_iter
        iters_per_day = (24*60*60) / t_per_iter
        k_iters_per_day = int(iters_per_day//1000)
        assert k_iters_per_day >= 100, 'too slow ({:d}k iters/day)'.format(
            k_iters_per_day
        )
