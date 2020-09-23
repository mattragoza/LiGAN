import sys, os, argparse

import generate as g


class DummyGrid(g.MolGrid):
    '''
    Does not actually contain a density grid, just
    satisfies the OutputWriter.write() interface.
    '''
    def __init__(self, **info):
        self.info = info


class ValidMolMaker(g.AtomFitter, g.OutputWriter):

    def __init__(
        self,
        dkoes_make_mol,
        out_prefix,
        n_samples,
        verbose,
    ):
        # atom fitter settings
        self.dkoes_make_mol = dkoes_make_mol

        # output writer settings
        super(g.OutputWriter, self).__init__(
            out_prefix=out_prefix,
            output_dx=False,
            output_sdf=True,
            output_channels=True,
            output_visited=False,
            n_samples=n_samples,
            blob_names=['lig', 'lig_gen'],
            fit_atoms=True,
            batch_metrics=False,
            verbose=True,
        )

    def write(self, lig_name, grid_name, sample_idx, struct):
        '''
        Write output files and compute metrics for struct.
        '''
        self.validify(struct)
        grid = DummyGrid(src_struct=struct)
        super(g.OutputWriter, self).write(
            lig_name, grid_name, sample_idx, grid
        )

    def compute_grid_metrics(self, *args, **kwargs):
        pass

    def compute_fit_metrics(self, *args, **kwargs):
        pass



def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', required=True)
    parser.add_argument('--n_examples', required=True, type=int)
    parser.add_argument('--n_samples', required=True, type=int)
    parser.add_argument('--in_prefix', required=True)
    parser.add_argument('--n_job_scripts', default=1, type=int)
    parser.add_argument('--job_script_idx', default=0, type=int)
    parser.add_argument('--lig_map', default='my_lig_map')
    parser.add_argument('--dkoes_make_mol', default=False, action='store_true')
    #parser.add_argument('--out_prefix')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    lig_channels = g.atom_types.get_channels_from_file(
        map_file=args.lig_map,
        name_prefix='Ligand',
    )

    mol_maker = ValidMolMaker(args.dkoes_make_mol)

    examples = g.read_examples_from_data_file(args.data_file)

    for example_idx in range(args.n_examples):

        array_idx = example_idx*args.n_job_scripts + args.job_script_idx + 1

        for sample_idx in range(args.n_samples):

            lig_src_file = examples[example_idx][1]
            lig_src_no_ext = os.path.splitext(lig_src_file)[0]
            lig_name = os.path.basename(lig_src_no_ext)
            
            for grid_name in ['lig', 'lig_fit', 'lig_gen_fit']:

                struct_file = os.path.join(
                    args.in_prefix,
                    '_'.join([
                        args.in_prefix,
                        str(array_idx),
                        lig_name,
                        grid_name,
                        str(sample_idx),
                    ])
                ) + '.sdf'
                try:
                    struct = g.MolStruct.from_sdf(struct_file, lig_channels)
                    mol_maker.write(struct)

                except OSError as e:
                    print(e)
                    continue


if __name__ == '__main__':
    main(sys.argv[1:])
