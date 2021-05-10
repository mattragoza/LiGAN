import sys, os, re
from functools import lru_cache
from rdkit import Chem
import molgrid
import liGAN
from liGAN import molecules as mols

# cache since we're not shuffling and each file has multiple poses
read_ob_mols_from_file = \
    lru_cache(maxsize=100)(mols.read_ob_mols_from_file)
read_rd_mols_from_sdf_file = \
    lru_cache(maxsize=100)(mols.read_rd_mols_from_sdf_file)


def find_real_lig_in_data_root(lig_src_no_ext, data_root):
    # this is based on the function in generate.py
    # but just for the cross-docked set and it
    # returns an ob_mol instead of rd_mol
    m = re.match(r'(.+)_(\d+)', lig_src_no_ext)
    lig_mol_base = m.group(1) + '.sdf'
    idx = int(m.group(2))
    lig_mol_file = os.path.join(data_root, lig_mol_base)
    try:
        return (
            read_ob_mols_from_file(lig_mol_file, 'sdf')[idx],
            read_rd_mols_from_sdf_file(lig_mol_file, sanitize=False)[idx]
        )
    except:
        print(lig_mol_file)
        raise


def make_mols(
    data_root,
    data_file,
    n_examples,
    typer_fns,
):
    atom_typer = liGAN.atom_types.AtomTyper.get_typer(typer_fns, 'c')
    bond_adder = liGAN.bond_adding.BondAdder()

    if not os.path.isdir('mols'):
        os.mkdir('mols')

    print(
        'example_idx lig_name pose_idx n_atoms_diff '
        'type_count_diff morgan_sim smi_match',
        flush=True
    )

    f = open(data_file)
    for i in range(n_examples):

        lig_src = next(f).rstrip().split(' ')[4]
        lig_src_no_ext = os.path.splitext(lig_src)[0]
        lig_name = os.path.basename(lig_src_no_ext)
        lig_name, pose_idx = lig_name.rsplit('_', 1)
        pose_idx = int(pose_idx)

        lig_ob_mol, lig_rd_mol = find_real_lig_in_data_root(
            lig_src_no_ext, data_root
        )
        try:
            Chem.SanitizeMol(lig_rd_mol)
            lig_rd_mol = mols.Molecule(Chem.AddHs(lig_rd_mol, addCoords=True))
        except Exception as e:
            print('SANITIZE {} {}'.format(lig_name, pose_idx), file=sys.stderr)
            print(e, file=sys.stderr)
            pass

        lig_ob_mol.AddHydrogens()
        lig_struct = atom_typer.make_struct(lig_ob_mol)
        lig_add_mol, lig_add_struct, visited_mols = bond_adder.make_mol(
            lig_struct
        )

        # difference in num atoms
        n_atoms_diff = lig_add_mol.n_atoms - lig_rd_mol.n_atoms

        # difference in type counts
        lig_type_counts = lig_struct.type_counts
        lig_add_type_counts = lig_add_struct.type_counts
        type_count_diff = (
            lig_type_counts - lig_add_type_counts
        ).norm(p=1).item()

        # fingerprint similarity
        morgan_sim = mols.get_rd_mol_similarity(
            lig_rd_mol, lig_add_mol, 'morgan'
        )

        # smiles string comparison
        lig_smi = lig_rd_mol.to_smi()
        lig_add_smi = lig_add_mol.to_smi()
        smi_match = (lig_smi == lig_add_smi)

        print('{} {} {} {} {} {:.4f} {}'.format(
            i,
            lig_name,
            pose_idx,
            n_atoms_diff,
            type_count_diff,
            morgan_sim,
            smi_match,
        ), flush=True)

        if not smi_match: # write out the mismatched molecules
            mols.write_rd_mols_to_sdf_file(
                'mols/{}_{}_add.sdf'.format(lig_name, pose_idx),
                visited_mols + [lig_rd_mol],
                kekulize=False
            )


if __name__ == '__main__':
    data_root, data_file, n_examples, typer_fns = sys.argv[1:]
    n_examples = int(n_examples)
    make_mols(data_root, data_file, n_examples, typer_fns)
