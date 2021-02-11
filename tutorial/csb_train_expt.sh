#!/bin/bash
#SBATCH --job-name=<EXPT_NAME>_trial<TRIAL_NUM>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --time=672:00:00
#SBATCH --qos=normal
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --dependency=singleton

source ~/.bashrc
cd $SLURM_SUBMIT_DIR

JOB_SCRIPTS_FILE=../trial<TRIAL_NUM>_job_scripts
if [ ! -f "${JOB_SCRIPTS_FILE}" ];
then
	echo error: file ${JOB_SCRIPTS_FILE} does not exist >&2
	exit
fi

N_JOB_SCRIPTS=$(cat ${JOB_SCRIPTS_FILE} | wc -l)
N_SEEDS=<N_SEEDS>

JOB_SCRIPT_INDEX=$(( (${SLURM_ARRAY_TASK_ID} - 1) / N_SEEDS + 1 ))
SEED_INDEX=$(( (${SLURM_ARRAY_TASK_ID} - 1) % N_SEEDS ))

JOB_SCRIPT=../$(head -n"${JOB_SCRIPT_INDEX}" ${JOB_SCRIPTS_FILE} | tail -n1)
WORK_DIR=$(dirname $JOB_SCRIPT)
WORK_DIR=$(realpath $WORK_DIR)

echo slurm_array_job_id $SLURM_ARRAY_JOB_ID
echo slurm_array_task_id $SLURM_ARRAY_TASK_ID
echo job_script_index $JOB_SCRIPT_INDEX
echo seed_index $SEED_INDEX

cd $WORK_DIR
eval $JOB_SCRIPT $WORK_DIR $SEED_INDEX $SLURM_ARRAY_JOB_ID

exit
