import sys, os, re, time
from functools import lru_cache
import numpy as np
import pandas as pd
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
    use_ob_mol,
    remove_h,
):
    atom_typer = liGAN.atom_types.AtomTyper.get_typer(
        typer_fns, 'c', device='cpu'
    )
    bond_adder = liGAN.bond_adding.BondAdder()

    if not os.path.isdir('mols'):
        os.mkdir('mols')

    columns = [
        'example_idx',
        'lig_name',
        'pose_idx',
        'elem_count_diff',
        'prop_count_diff',
        'n_atoms_diff',
        'lig_smi',
        'lig_valid',
        'lig_reason',
        'lig_add_smi',
        'lig_add_valid',
        'lig_add_reason',
        'smi_match',
        'rd_sim',
        'ob_sim',
        'add_time',
    ]
    print(' '.join(columns), flush=True)

    f = open(data_file)
    for example_idx in range(n_examples):

        lig_src = next(f).rstrip().split(' ')[4]
        lig_src_no_ext = lig_src.split('.', 1)[0]
        lig_name = os.path.basename(lig_src_no_ext)
        lig_name, pose_idx = lig_name.rsplit('_', 1)
        pose_idx = int(pose_idx)

        lig_ob_mol, lig_rd_mol = find_real_lig_in_data_root(
            lig_src_no_ext, data_root
        )
        
        # add hydrogens to ob_mol
        lig_ob_mol.AddHydrogens()

        if use_ob_mol: # use OB for real mol instead of RDkit
            lig_rd_mol = mols.Molecule.from_ob_mol(lig_ob_mol)

        else: # add hydrogens to rd_mol
            try: # which requires sanitize first
                lig_rd_mol.sanitize()
                lig_rd_mol = mols.Molecule(
                    Chem.AddHs(lig_rd_mol, addCoords=True)
                )
            except Chem.MolSanitizeException:
                pass # ignore, we'll find out why in validate()

        # validate real molecule
        lig_valid, lig_reason = lig_rd_mol.validate()

        # create typed struct from real ob_mol
        lig_struct = atom_typer.make_struct(lig_ob_mol)

        # reconstruct mol from struct by bond adding
        t_start = time.time()
        lig_add_mol, lig_add_struct, visited_mols = \
            bond_adder.make_mol(lig_struct)
        add_time = time.time() - t_start

        ### STRUCT-LEVEL METRICS ###

        type_count_diff = ( # difference in overall type counts 
            lig_struct.type_counts - lig_add_struct.type_counts
        ).abs().sum().item()

        elem_count_diff = ( # difference in element type counts 
            lig_struct.elem_counts - lig_add_struct.elem_counts
        ).abs().sum().item()

        prop_count_diff = ( # difference in property type counts
            lig_struct.prop_counts - lig_add_struct.prop_counts
        ).abs().sum().item()

        ### MOLECULE-LEVEL METRICS ###

        n_atoms_diff = ( # difference in num atoms
            lig_add_mol.n_atoms - lig_rd_mol.n_atoms
        )

        # validate output molecule
        lig_add_valid, lig_add_reason = lig_add_mol.validate()

        if remove_h: # remove hydrogen before computing similarity
            lig_rd_mol = lig_rd_mol.remove_hs()
            lig_add_mol = lig_add_mol.remove_hs()

        # smiles string comparison
        lig_smi = lig_rd_mol.to_smi()
        lig_add_smi = lig_add_mol.to_smi()
        smi_match = (lig_smi == lig_add_smi)

        if lig_valid and lig_add_valid:
            rd_sim = mols.get_rd_mol_similarity(lig_add_mol, lig_rd_mol)
            ob_sim = mols.get_ob_smi_similarity(lig_add_smi, lig_smi)
        else:
            rd_sim = ob_sim = np.nan

        print('{} {} {} {} {} {} "{}" {} "{}" "{}" {} "{}" {} {:.4f} {:.4f} {:.4f}'.format(
            example_idx,
            lig_name,
            pose_idx,
            elem_count_diff,
            prop_count_diff,
            n_atoms_diff,
            lig_smi,
            lig_valid,
            lig_reason,
            lig_add_smi,
            lig_add_valid,
            lig_add_reason,
            smi_match,
            rd_sim,
            ob_sim,
            add_time,
        ), flush=True)

        if not smi_match: # write out the mismatched molecules

            mols.write_rd_mols_to_sdf_file(
                'mols/{}_{}_add.sdf'.format(lig_name, pose_idx),
                visited_mols + [lig_rd_mol],
                kekulize=False
            )


if __name__ == '__main__':
    data_root, data_file, n_examples, typer_fns, use_ob_mol, remove_h = sys.argv[1:]
    make_mols(
        data_root,
        data_file,
        int(n_examples),
        typer_fns,
        use_ob_mol,
        remove_h
    )
