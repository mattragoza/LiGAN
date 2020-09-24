#!/bin/bash
#SBATCH --job-name=find_pubchem_gninatypes
#SBATCH -p dept_cpu
#SBATCH -t 672:00:00

DATA_ROOT=/net/pulsar/home/koes/mtr22/pubchem
for sim in $(seq 0 0.1 1)
do
	sim_dir=$DATA_ROOT/molport_diff/${sim}
	types_file=pubchem_diff${sim}.types
	find $sim_dir -type f -name '*.gninatypes' -printf "1 0 10gs_rec.gninatypes molport_diff/${sim}/%f\n" > $types_file
	wc -l $types_file
done
cat pubchem_diff{0.{0,1,2,3,4,5,6,7,8,9},1.0}.types > pubchem_all_diffs.types
wc -l pubchem_all_diffs.types

