#!/bin/bash
#SBATCH --job-name=make_pubchem_gninatypes
#SBATCH -p dept_cpu
#SBATCH -t 672:00:00

DATA_ROOT=/net/pulsar/home/koes/mtr22/pubchem
for sim in $(seq 0 0.1 1)
do
	cd $DATA_ROOT/molport_diff/${sim}
	for sdf_file in $(ls *.sdf)
	do
		echo gninatyper $sdf_file
		gninatyper $sdf_file
	done
done
echo done

