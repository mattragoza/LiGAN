#!/bin/bash
#SBATCH --job-name=openbabel_validify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=dept_cpu
#SBATCH --mem=32gb
#SBATCH --time=672:00:00
#SBATCH --qos=normal
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --dependency=afterany:2763865

source ~/.bashrc
cd $SLURM_SUBMIT_DIR

JOB_SCRIPTS_FILE=../openbabel_job_scripts
if [ ! -f "${JOB_SCRIPTS_FILE}" ];
then
	echo error: file ${JOB_SCRIPTS_FILE} does not exist >&2
	exit
fi

JOB_SCRIPT_INDEX=${SLURM_ARRAY_TASK_ID}
JOB_SCRIPT=../$(head -n"${JOB_SCRIPT_INDEX}" ${JOB_SCRIPTS_FILE} | tail -n1)
WORK_DIR=$(dirname $JOB_SCRIPT)
WORK_DIR=$(realpath $WORK_DIR)

cd $WORK_DIR
eval $JOB_SCRIPT $WORK_DIR $SLURM_ARRAY_JOB_ID
exit
