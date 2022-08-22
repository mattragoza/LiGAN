import molgrid
from ligan.atom_types import AtomTyper

data_file = 'data/crossdock2020/selected_test_targets.types'
data_root = 'data/crossdock2020'
rec_typer = AtomTyper.get_typer('oadc', 1.0, rec=True)
lig_typer = AtomTyper.get_typer('oadc', 1.0, rec=False)
ex_provider = molgrid.ExampleProvider(rec_typer, lig_typer, data_root=data_root)
ex_provider = molgrid.ExampleProvider(data_root=data_root)
ex_provider.populate(data_file)
ex_provider.next_batch(100)
print('OK')

