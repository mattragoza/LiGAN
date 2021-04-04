import sys, molgrid
import numpy as np
sys.path.insert(0, '.')
import liGAN

rec_typer = molgrid.FileMappedGninaTyper('data/my_rec_map')
lig_typer = molgrid.FileMappedGninaTyper('data/my_lig_map')
lig_channels = liGAN.atom_types.get_channels_from_map(lig_typer)

print('loading data')
ex_provider = molgrid.ExampleProvider(
	rec_typer,
	lig_typer,
	data_root='data/molport',
	recmolcache='data/molportFULL_rec.molcache2' and '',
	ligmolcache='data/molportFULL_lig.molcache2' and '',
	shuffle=True
)
ex_provider.populate('data/molportFULL_rand_test0_1000.types')

batch_size = 1000
n_examples = ex_provider.size()
n_batches = n_examples // batch_size

type_counts = np.zeros(lig_typer.num_types())
mol_count = 0

for i in range(n_batches):
	for ex in ex_provider.next_batch(batch_size):
		struct = liGAN.atom_structs.AtomStruct.from_coord_set(
			ex.coord_sets[1], lig_channels
		)
		type_counts += struct.type_counts
		mol_count += 1
	print('batch {}/{}'.format(i+1, n_batches))

for i, channel in enumerate(lig_channels):
    print('{:2d} {:10d} {:8.3f} {}'.format(
    	i, int(type_counts[i]), type_counts[i]/mol_count, channel.name
    ))
