#!/bin/bash
#PBS -N GAN_NAME
#PBS -l nodes=1:ppn=1:gpus=1:exclusive_process
#PBS -l mem=MEMORY
#PBS -l walltime=WALLTIME
#PBS -q QUEUE
JOB_PARAMS
source ~/.bashrc
cd ${PBS_O_WORKDIR}

SEED=$((${PBS_ARRAYID} / 4))
FOLD=$((${PBS_ARRAYID} % 4))

if (($FOLD == 3)); then
	FOLD=all;
fi

SCR_DIR=/scr/$PBS_JOBID

if [[ ! -e $SCR_DIR ]]; then
	mkdir $SCR_DIR
fi

cp ${LIGAN_ROOT}/data/DATA_PREFIX*.types ${SCR_DIR}
cp ${LIGAN_ROOT}/models/DATA_MODEL_NAME.model ${SCR_DIR}
cp ${LIGAN_ROOT}/models/GEN_MODEL_NAME.model ${SCR_DIR}
cp ${LIGAN_ROOT}/models/DISC_MODEL_NAME.model ${SCR_DIR}
cp ${LIGAN_ROOT}/solvers/SOLVER_NAME.solver ${SCR_DIR}
cp ${LIGAN_ROOT}/weights/lig_gauss_conv.caffemodel ${SCR_DIR}

if ((CONT_ITER > 0)); then
	cp GAN_NAME.*_iter_CONT_ITER.solverstate ${SCR_DIR}
	cp GAN_NAME.*_iter_CONT_ITER.caffemodel ${SCR_DIR}
fi

cd ${SCR_DIR}

trap "cp *.{model,solver,caffemodel,solverstate,training_output,png,pdf} $PBS_O_WORKDIR" EXIT

python ${LIGAN_ROOT}/train.py -d DATA_MODEL_NAME.model -g GEN_MODEL_NAME.model -a DISC_MODEL_NAME.model -p DATA_PREFIX -r DATA_ROOT -n ${FOLD} --random_seed ${SEED} -s SOLVER_NAME.solver --max_iter MAX_ITER --cont_iter CONT_ITER --gen_train_iter GEN_TRAIN_ITER --disc_train_iter DISC_TRAIN_ITER --instance_noise INSTANCE_NOISE --loss_weight LOSS_WEIGHT --loss_weight_decay LOSS_WEIGHT_DECAY TRAIN_OPTIONS -o GAN_NAME.DATA_PREFIX.${SEED}

exit
