import sys, os, argparse, glob
from collections import defaultdict
import pandas as pd
from rdkit import Chem

import generate as g


class ValidMolMaker(g.AtomFitter, g.OutputWriter):

    def __init__(
        self,
        dkoes_make_mol,
        use_openbabel,
        out_prefix,
        output_sdf,
        output_visited,
        n_samples,
        verbose,
    ):
        self.dkoes_make_mol = dkoes_make_mol
        self.mtr22_make_mol = False
        self.use_openbabel = use_openbabel

        self.out_prefix = out_prefix
        self.output_sdf = output_sdf
        self.output_visited = output_visited
        self.n_samples = n_samples

        self.metric_file = '{}.gen_metrics'.format(out_prefix)
        columns = ['lig_name', 'sample_idx']
        self.metrics = pd.DataFrame(columns=columns).set_index(columns)

        self.pymol_file = '{}.pymol'.format(out_prefix)
        self.sdf_files = []
        self.centers = []

        self.verbose = verbose

        # organize structs by lig_name, sample_idx, struct_type
        self.structs = defaultdict(lambda: defaultdict(dict))

    def write(self, lig_name, struct_type, sample_idx, struct):
        '''
        Write output files and compute metrics for struct.
        '''
        struct_prefix = '{}_{}_{}'.format(self.out_prefix, lig_name, struct_type)
        src_sample_prefix = struct_prefix + '_src_' + str(sample_idx)
        add_sample_prefix = struct_prefix + '_add_' + str(sample_idx)
        is_fit_struct = struct_type.endswith('_fit')

        if self.output_sdf:

            if not is_fit_struct: # write real input molecule

                src_mol_file = src_sample_prefix + '.sdf'
                if self.verbose:
                    print('Writing ' + src_mol_file)

                src_mol = struct.info['src_mol']
                if self.output_visited:
                    rd_mols = src_mol.info['visited_mols']
                else:
                    rd_mols = [src_mol]
                g.write_rd_mols_to_sdf_file(src_mol_file, rd_mols)
                self.sdf_files.append(src_mol_file)
                self.centers.append(struct.center)

            add_mol_file = add_sample_prefix + '.sdf'
            if self.verbose:
                print('Writing ' + add_mol_file)

            add_mol = struct.info['add_mol']
            if self.output_visited:
                rd_mols = add_mol.info['visited_mols']
            else:
                rd_mols = [add_mol]
            g.write_rd_mols_to_sdf_file(add_mol_file, rd_mols)
            self.sdf_files.append(add_mol_file)
            self.centers.append(struct.center)

        # store struct until ready to compute output metrics
        self.structs[lig_name][sample_idx][struct_type] = struct
        lig_structs = self.structs[lig_name]

        # store until structs for all samples are ready
        has_all_samples = len(lig_structs) == self.n_samples
        has_all_structs = all(len(lig_structs[i]) == 3 for i in lig_structs)
        
        if has_all_samples and has_all_structs:

            if self.verbose:
                print('Computing metrics for all ' + lig_name + 'samples')

            self.compute_metrics(lig_name, range(self.n_samples))

            if self.verbose:
                print('Writing ' + self.metric_file)

            self.metrics.to_csv(self.metric_file, sep=' ')

            if self.verbose:
                print('Writing ' + self.pymol_file)

            g.write_pymol_script(
                self.pymol_file,
                self.out_prefix,
                [],
                self.sdf_files,
                self.centers,
            )
            del self.structs[lig_name]

    def compute_metrics(self, lig_name, sample_idxs):

        lig_structs = self.structs[lig_name]

        for sample_idx in sample_idxs:
            idx = (lig_name, sample_idx)

            lig_struct = lig_structs[sample_idx]['lig']
            lig_fit_struct = lig_structs[sample_idx]['lig_fit']
            lig_gen_fit_struct = lig_structs[sample_idx]['lig_gen_fit']

            lig_mol = lig_struct.info['src_mol']
            lig_add_mol = lig_struct.info['add_mol']
            lig_fit_add_mol = lig_fit_struct.info['add_mol']
            lig_gen_fit_add_mol = lig_gen_fit_struct.info['add_mol']

            self.compute_mol_metrics(idx, 'lig', lig_mol)
            self.compute_mol_metrics(idx, 'lig_add', lig_add_mol, lig_mol)
            self.compute_mol_metrics(idx, 'lig_fit_add', lig_fit_add_mol, lig_mol)
            self.compute_mol_metrics(idx, 'lig_gen_fit_add', lig_gen_fit_add_mol, lig_mol)

        if self.verbose:
            pd.set_option('display.max_columns', 100)
            print(self.metrics.loc[lig_name].loc[sample_idxs])

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', required=True)
    parser.add_argument('--data_file', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--n_examples', type=int)
    parser.add_argument('--n_samples', required=True, type=int)
    parser.add_argument('--lig_map', required=True)
    parser.add_argument('--dkoes_make_mol', default=False, action='store_true')
    parser.add_argument('--use_openbabel', default=False, action='store_true')
    parser.add_argument('--out_prefix', required=True)
    parser.add_argument('--output_sdf', default=False, action='store_true')
    parser.add_argument('--output_visited', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    return parser.parse_args(argv)
   

def main(argv):
    args = parse_args(argv)

    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_colwidth', 100)
    try:
        display_width = g.get_terminal_size()[1]
    except:
        display_width = 185
    pd.set_option('display.width', display_width)

    lig_channels = g.atom_types.get_channels_from_file(
        map_file=args.lig_map,
        name_prefix='Ligand',
    )

    print('Initializing valid molecule maker')
    mol_maker = ValidMolMaker(
        dkoes_make_mol=args.dkoes_make_mol,
        use_openbabel=args.use_openbabel,
        out_prefix=args.out_prefix,
        output_sdf=args.output_sdf,
        output_visited=args.output_visited,
        n_samples=args.n_samples,
        verbose=args.verbose,
    )

    print('Reading {} examples from {}'.format(args.n_examples, args.data_file))
    job_name = os.path.basename(args.in_dir)
    examples = g.read_examples_from_data_file(args.data_file, n=args.n_examples)

    for example_idx, example in enumerate(examples):
        progress = '[{}/{}] '.format(example_idx, args.n_examples)

        # get the lig_name from data example
        rec_src, lig_src = example
        lig_src_no_ext = os.path.splitext(lig_src)[0]
        lig_name = os.path.basename(lig_src_no_ext)

        # read the source molecule from data root
        lig_mol_file = os.path.join(args.data_root, lig_src_no_ext + '.sdf')
        print(progress + 'Reading ' + lig_mol_file)
        src_mol = g.read_rd_mols_from_sdf_file(lig_mol_file)[0]
        src_mol = Chem.RemoveHs(src_mol)
        
        # then get all of the derived atom type structs
        for struct_type in ['lig', 'lig_fit', 'lig_gen_fit']:

            # first, look for files that contain all samples
            # of a given (lig_name, struct_type)

            try: # if array idx is in filename, we need to glob
                struct_file = os.path.join(
                    args.in_dir,
                    '_'.join([
                        job_name,
                        '*', # array_idx
                        lig_name,
                        struct_type,
                    ])
                ) + '.sdf'
                print(progress + 'Globbing ' + struct_file)
                struct_file = glob.glob(struct_file)[0]
                print(progress + 'Reading ' + struct_file)
                xyz_mols = g.read_rd_mols_from_sdf_file(struct_file)
                found_structs = True

            except (IndexError, OSError):
                print(progress + 'No structs in glob')
                found_structs = False

            for sample_idx in range(args.n_samples):

                # get the atom coords for this sample_idx
                if found_structs:
                    xyz_mol = xyz_mols[sample_idx]

                else:
                    # otherwise look for files that contain a
                    # single (lig_name, struct_type, sample_idx)

                    # again, need to glob if the array_idx is in filename
                    struct_file = os.path.join(
                        args.in_dir,
                        '_'.join([
                            job_name,
                            '*', # array_idx
                            lig_name,
                            struct_type,
                            str(sample_idx),
                        ])
                    ) + '.sdf'

                    try:
                        print(progress + 'Globbing ' + struct_file)
                        struct_file = glob.glob(struct_file)[0]
                    except IndexError:
                        # read_mols will raise a better error message
                        pass

                    try:
                        print(progress + 'Reading ' + struct_file)
                        # these aren't validified molecules, just fit atom coords
                        xyz_mol = g.read_rd_mols_from_sdf_file(struct_file)[-1]
                    except OSError as e:
                        print(progress + 'Warning: ' + str(e))
                        continue           

                # get the atom types from channels file
                channels_file = struct_file.replace('.sdf', '_{}.channels'.format(sample_idx))
                print(progress + 'Reading ' + channels_file)
                c = g.read_channels_from_file(channels_file, lig_channels)

                # get the struct from mol coords and atom types
                struct = g.MolStruct.from_rd_mol(xyz_mol, c, lig_channels)

                print(progress + 'Validifying {} {} {} struct'.format(lig_name, struct_type, sample_idx))

                # validify the atom types and coords into a molecule
                mol_maker.validify(struct)

                # for real atom types, also align, minimize, and store the real molecule
                if struct_type == 'lig': 
                    print(progress + 'Minimizing {} {} {} molecule'.format(lig_name, struct_type, sample_idx))
                    src_mol_ = Chem.RWMol(src_mol)
                    try:
                        Chem.rdMolAlign.AlignMol(src_mol_, struct.info['add_mol'])
                    except RuntimeError as e:
                        print(progress + 'Warning: ' + str(e))
                    mol_maker.uff_minimize(src_mol_)
                    struct.info['src_mol'] = src_mol_

                mol_maker.write(lig_name, struct_type, sample_idx, struct)


if __name__ == '__main__':
    main(sys.argv[1:])
