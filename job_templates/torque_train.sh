#!/bin/bash
#PBS -N <JOB_NAME>
#PBS -l nodes=1:ppn=1:gpus=1:exclusive_process
#PBS -l mem=<MEMORY>
#PBS -l walltime=<WALLTIME>
#PBS -q <QUEUE>
<JOB_PARAMS>
source ~/.bashrc
cd $PBS_O_WORKDIR

SCR_DIR=/scr/$PBS_JOBID
if [[ ! -e $SCR_DIR ]]; then
	mkdir $SCR_DIR
fi

cp <DATA_DIR>/<DATA_NAME>*.types $SCR_DIR
cp <MODEL_DIR>/<DATA_MODEL_NAME>.model $SCR_DIR
cp <MODEL_DIR>/<GEN_MODEL_NAME>.model $SCR_DIR
cp <MODEL_DIR>/<DISC_MODEL_NAME>.model $SCR_DIR
cp <SOLVER_DIR>/<SOLVER_NAME>.solver $SCR_DIR

CONT_ITER=<CONT_ITER>
if (($CONT_ITER > 0)); then
	cp <JOB_NAME>.*_iter_<CONT_ITER>.solverstate $SCR_DIR
	cp <JOB_NAME>.*_iter_<CONT_ITER>.caffemodel $SCR_DIR
	cp <JOB_NAME>.training_output $SCR_DIR
fi

cd $SCR_DIR

trap "cp *.{model,solver,caffemodel,solverstate,training_output,png,pdf} $PBS_O_WORKDIR" EXIT

python <LIGAN_DIR>/train.py -d <DATA_MODEL_NAME>.model -g <GEN_MODEL_NAME>.model -a <DISC_MODEL_NAME>.model -p <DATA_NAME> -r <DATA_ROOT> -n <FOLD> --random_seed <SEED> -s <SOLVER_NAME>.solver --max_iter <MAX_ITER> --cont_iter <CONT_ITER> --gen_train_iter <GEN_TRAIN_ITER> --disc_train_iter <DISC_TRAIN_ITER> --test_interval <TEST_INTERVAL> --test_iter <TEST_ITER> --instance_noise <INSTANCE_NOISE> --loss_weight <LOSS_WEIGHT> --loss_weight_decay <LOSS_WEIGHT_DECAY> <TRAIN_OPTIONS> -o <JOB_NAME>.<DATA_NAME>.<SEED>

exit
