#!/bin/bash
#SBATCH --job-name=create_pubchem_caches
#SBATCH -p dept_cpu
#SBATCH -t 672:00:00

DATA_ROOT=/net/pulsar/home/koes/mtr22/pubchem
IN_FILE=pubchem_all_diffs.types
REC_CACHE=pubchem_all_diffs_rec.molcache2
LIG_CACHE=pubchem_all_diffs_lig.molcache2
python3 ~/scripts/create_caches2.py $IN_FILE -c 2 -d $DATA_ROOT --recmolcache $REC_CACHE --ligmolcache $LIG_CACHE

