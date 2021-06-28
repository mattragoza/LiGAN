import sys, os, time, torch
from contextlib import redirect_stderr
import liGAN


if __name__ == '__main__':

    _, model_type, init_conv_pool, block_type, growth_rate, bn_factor = sys.argv
    init_conv_pool = int(init_conv_pool)
    growth_rate = int(growth_rate)
    bn_factor = int(bn_factor)

    model_name = '{}_{}_{}_{}_{}'.format(
        model_type, init_conv_pool, block_type, growth_rate, bn_factor
    )

    batch_size = 16
    n_rec_channels = 16
    n_lig_channels = 16
    grid_size = 48

    model_type = getattr(liGAN.models, model_type)
    model = model_type(
        n_channels_in=n_lig_channels if model_type.has_input_encoder else None,
        n_channels_cond=n_rec_channels if model_type.has_conditional_encoder else None,
        n_channels_out=n_lig_channels,
        grid_size=grid_size,
        n_filters=32,
        width_factor=2,
        n_levels=4 - bool(init_conv_pool),
        conv_per_level=3,
        kernel_size=3,
        relu_leak=0.1,
        batch_norm=0,
        spectral_norm=1,
        pool_type='a',
        unpool_type='n',
        n_latent=128,
        skip_connect=model_type.has_conditional_encoder,
        init_conv_pool=init_conv_pool,
        block_type=block_type,
        growth_rate=growth_rate,
        bottleneck_factor=bn_factor,
        device='cuda',
        debug=True,
    )
    optim = torch.optim.RMSprop(model.parameters())

    n_trials = 10
    t_data = 0.3 * batch_size # estimate of time getting data (no caching)

    inputs = torch.zeros(
        batch_size, n_lig_channels, grid_size, grid_size, grid_size
    ).cuda()

    conditions = torch.zeros(
        batch_size, n_rec_channels, grid_size, grid_size, grid_size
    ).cuda()

    t_start = time.time()
    torch.cuda.reset_max_memory_allocated()
    for i in range(n_trials):
        if i == 0:
            debug_file = 'tests/output/TEST_{}.model_debug'.format(model_name)
            with open(debug_file, 'w') as f:
                with redirect_stderr(f):
                    generated, latents, means, log_stds = model(
                        inputs, conditions, batch_size
                    )
        else:
            generated, latents, means, log_stds = model(
                inputs, conditions, batch_size
                )
            loss = ((inputs - generated)**2).sum() / 2
            optim.zero_grad()
            loss.backward()
            optim.step()
    t_delta = (time.time() - t_start + t_data) / n_trials / batch_size
    batch_days = int(24*60*60 / t_delta / 1000)
    gpu = torch.cuda.max_memory_allocated()

    n_params = liGAN.models.get_n_params(model)
    print('{}\t{:.1f}M params\t{:.2f}s / example\t{:d}k examples / day\t{:.1f} gb'.format(
        model_name, n_params / 1e6, t_delta, batch_days, gpu / 1024**3
    ))
