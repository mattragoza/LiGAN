import sys, os, argparse
import pandas as pd
import ligan

parser = argparse.ArgumentParser()
parser.add_argument('data_root')
parser.add_argument('train_file')
parser.add_argument('metrics_file')
parser.add_argument('--groupby', default='all')
parser.add_argument('-o', '--out_file')
args = parser.parse_args()

print('Loading train set molecules')
train_mols = set()
with open(args.train_file) as f:
    for i, line in enumerate(f):
        mol_file = os.path.join(args.data_root, line.split()[4])
        mol = ligan.molecules.Molecule.from_sdf(mol_file)
        train_mols.add(mol.to_smi())

print('Loading generated molecules')
df = pd.read_csv(args.metrics_file, sep=' ')
df['all'] = 'all'

try:
    df = df.groupby(args.groupby)
except KeyError as e:
    print(f'{args.groupby} is not a column in {args.metrics_file}!')
    print('Available columns:')
    print(list(df.columns))
    raise

smiles_col = 'lig_gen_fit_add_SMILES'
valid_col = 'lig_gen_fit_add_valid'

unique = lambda x: x.nunique() / x.count()
novel = lambda x: len(set(x) - train_mols) / x.count()

df = pd.DataFrame({
    '% valid': df[valid_col].mean() * 100,
    '% novel': df[smiles_col].agg(novel) * 100,
    '% unique': df[smiles_col].agg(unique) * 100
})
print(df)

if args.out_file:
    df.to_csv(args.out_file, sep=' ')

