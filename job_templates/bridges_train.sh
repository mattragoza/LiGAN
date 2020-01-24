#!/bin/bash
#SBATCH --job-name=<JOB_NAME>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=<QUEUE>
#SBATCH --gres=gpu:volta16:1
#SBATCH --mem=<MEMORY>
#SBATCH --time=<WALLTIME>
#SBATCH --qos=normal
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
<JOB_PARAMS>
source ~/.bashrc
cd $SLURM_SUBMIT_DIR

SCR_DIR=$SLURM_SUBMIT_DIR/$SLURM_JOB_ID
if [[ ! -e $SCR_DIR ]]; then
        mkdir -p $SCR_DIR
fi

echo ================================
echo Running on `whoami`@`hostname`
echo work_dir `pwd`
echo scr_dir $SCR_DIR
echo ld_library_path $LD_LIBRARY_PATH
echo ================================

cp <LIGAN_DIR>/my_{rec,lig}_map $SCR_DIR
cp <DATA_DIR>/<DATA_PREFIX>*.types $SCR_DIR
cp <MODEL_DIR>/<DATA_MODEL_NAME>.model $SCR_DIR
cp <MODEL_DIR>/<GEN_MODEL_NAME>.model $SCR_DIR
cp <MODEL_DIR>/<DISC_MODEL_NAME>.model $SCR_DIR
cp <SOLVER_DIR>/<SOLVER_NAME>.solver $SCR_DIR

CONT_ITER=<CONT_ITER>
if (($CONT_ITER > 0)); then
	cp <JOB_NAME>.*_iter_$CONT_ITER.solverstate $SCR_DIR
	cp <JOB_NAME>.*_iter_$CONT_ITER.caffemodel $SCR_DIR
	cp <JOB_NAME>.training_output $SCR_DIR
fi

cd $SCR_DIR

trap "cp *.{model,solver,caffemodel,solverstate,training_output,png,pdf} ${SLURM_SUBMIT_DIR}" EXIT

python3 <LIGAN_DIR>/train.py -d <DATA_MODEL_NAME>.model -g <GEN_MODEL_NAME>.model -a <DISC_MODEL_NAME>.model -p <DATA_PREFIX> -r <DATA_ROOT> -n <FOLD> --random_seed <SEED> -s <SOLVER_NAME>.solver --max_iter <MAX_ITER> --cont_iter $CONT_ITER --gen_train_iter <GEN_TRAIN_ITER> --disc_train_iter <DISC_TRAIN_ITER> --test_interval <TEST_INTERVAL> --test_iter <TEST_ITER> --instance_noise <INSTANCE_NOISE> --loss_weight <LOSS_WEIGHT> --loss_weight_decay <LOSS_WEIGHT_DECAY> <TRAIN_OPTIONS> -o <JOB_NAME>.<DATA_PREFIX>.<SEED>
exit
