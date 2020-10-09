#!/bin/bash
#SBATCH --job-name=split_each_molport_sdf
#SBATCH -p dept_cpu
#SBATCH -t 28-00:00:00

SCR_DIR=/scr/mtr22/molport/$SLURM_ARRAY_TASK_ID
mkdir -p $SCR_DIR

SDF_BASE=molportFULL_$SLURM_ARRAY_TASK_ID.sdf
python3 $LIGAN_ROOT/split_sdf.py $MOLPORT_ROOT/$SDF_BASE $SCR_DIR

mkdir -p $MOLPORT_ROOT/$SLURM_ARRAY_TASK_ID
rsync -r $SCR_DIR $MOLPORT_ROOT
echo done

