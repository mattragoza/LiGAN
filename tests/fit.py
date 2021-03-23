import sys, torch
assert torch.cuda.is_available()

sys.path.insert(0, '.')
import liGAN


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

with open('data/molportFULL_rand_test0_1000.types') as f:
    lig_files = ('data/molport/' + line.split(' ')[3] for line in f)

print('creating atom fitter')
atom_fitter = liGAN.atom_fitting.AtomFitter(debug=True)

print('fitting atoms to grids')
n_batches = 10
for i in range(n_batches):
    lig_grids, lig_structs, _ = data.forward(ligand_only=True)
    atom_fitter.fit_batch(
        lig_grids, data.lig_channels, (0,0,0), data.resolution
    )

print('done')
