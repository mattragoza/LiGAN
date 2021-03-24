import sys, torch
assert torch.cuda.is_available()

sys.path.insert(0, '.')
import liGAN


use_gen = bool(int(sys.argv[1]))
print(use_gen)

torch.cuda.reset_max_memory_allocated()

print('creating data loader')
data = liGAN.data.AtomGridData(
    data_root='data/molport',
    batch_size=10,
    rec_map_file='data/my_rec_map',
    lig_map_file='data/my_lig_map',
    resolution=0.5,
    dimension=23.5,
    shuffle=False,
)

print('loading data')
data.populate('data/molportFULL_rand_test0_1000.types')

if use_gen:

    print('creating gen model')
    model = liGAN.models.GAN(
        n_channels_out=19,
        grid_size=48,
        n_filters=32,
        width_factor=2,
        n_levels=4,
        conv_per_level=3,
        kernel_size=3,
        relu_leak=0.1,
        pool_type='a',
        unpool_type='n',
        pool_factor=2,
        n_latent=1024,
    )

    print('loading gen weights')
    model.load_state_dict(torch.load(
        'torch_training/train_GAN_8_1024_x_1/train_GAN_8_1024_x_1_iter_2000.checkpoint'
    )['gen_model_state'])

print('creating atom fitter')
atom_fitter = liGAN.atom_fitting.AtomFitter(debug=True)

gpu = torch.cuda.max_memory_allocated() // int(1024**2)
print('GPU', gpu)

print('fitting atoms to grids')
n_batches = 10
for i in range(n_batches):
    if use_gen:
        grids, _ = model.forward(10)
    else:
        grids, _, _ = data.forward(ligand_only=True)
    structs, _ = atom_fitter.fit_batch(
        grids, data.lig_channels, (0,0,0), data.resolution
    )

print('done')
