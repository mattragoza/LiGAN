#!/bin/bash
#SBATCH --job-name=split_pubchem_sdf
#SBATCH -p dept_cpu
#SBATCH -t 672:00:00

DATA_ROOT=/net/pulsar/home/koes/mtr22/pubchem
for sim in $(seq 0 0.1 1)
do
	mkdir -p $DATA_ROOT/molport_diff/${sim}
	python3 $LIGAN_ROOT/split_sdf.py $DATA_ROOT/molport_diff/pubchem_${sim}_1k.sdf $DATA_ROOT/molport_diff/${sim}
done
echo done

