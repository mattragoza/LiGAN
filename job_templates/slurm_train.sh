#!/bin/bash
#SBATCH --job-name=<JOB_NAME>
#SBATCH --gres=gpu:1
#SBATCH --mem=<MEMORY>
#SBATCH --time=<WALLTIME>
#SBATCH --qos=normal
#SBATCH --cluster=gpu
#SBATCH --partition=<QUEUE>
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
<JOB_PARAMS>
source ~/.bashrc
cd ${SLURM_SUBMIT_DIR}

SEED=<SEED> #$((${SLURM_ARRAY_TASK_ID} / 4))
FOLD=<FOLD> #$((${SLURM_ARRAY_TASK_ID} % 4))

if (($FOLD == 3)); then
	FOLD=all;
fi

cp ${LIGAN_ROOT}/data/<DATA_PREFIX>*.types ${SLURM_SCRATCH}
cp ../models/<DATA_MODEL_PARAMS>.model ${SLURM_SCRATCH}
cp ../models/<GEN_MODEL_PARAMS>.model ${SLURM_SCRATCH}
cp ../models/<DISC_MODEL_PARAMS>.model ${SLURM_SCRATCH}
cp ../solvers/<SOLVER_PARAMS>.solver ${SLURM_SCRATCH}

if ((<CONT_ITER> > 0)); then
	cp <JOB_NAME>.*_iter_<CONT_ITER>.solverstate ${SLURM_SCRATCH}
	cp <JOB_NAME>.*_iter_<CONT_ITER>.caffemodel ${SLURM_SCRATCH}
fi

cd ${SLURM_SCRATCH}

trap "cp *.{model,solver,caffemodel,solverstate,training_output,png,pdf} ${SLURM_SUBMIT_DIR}" EXIT

python ${LIGAN_ROOT}/train.py -d <DATA_MODEL_PARAMS>.model -g <GEN_MODEL_PARAMS>.model -a <DISC_MODEL_PARAMS>.model -p <DATA_PREFIX> -r <DATA_ROOT> -n ${FOLD} --random_seed ${SEED} -s <SOLVER_PARAMS>.solver --max_iter <MAX_ITER> --cont_iter <CONT_ITER> --gen_train_iter <GEN_TRAIN_ITER> --disc_train_iter <DISC_TRAIN_ITER> --test_interval <TEST_INTERVAL> --test_iter <TEST_ITER> --instance_noise <INSTANCE_NOISE> --loss_weight <LOSS_WEIGHT> --loss_weight_decay <LOSS_WEIGHT_DECAY> <TRAIN_OPTIONS>  -o <JOB_NAME>.<DATA_PREFIX>.${SEED}

exit
