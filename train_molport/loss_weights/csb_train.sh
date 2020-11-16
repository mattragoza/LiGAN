#!/bin/bash
#SBATCH --job-name=train_<JOB_NAME>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=672:00:00
#SBATCH --qos=normal
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --dependency=singleton
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

cp $LIGAN_ROOT/my_{rec,lig}_map $SCR_DIR

CONT_ITER=$(ls -U -1 | sed -n 's/.*_iter_\([0-9][0-9]*\).*/\1/p' | sort -n -r | head -n1)
if [ -z $CONT_ITER ]
then
	CONT_ITER=0
fi

if (($CONT_ITER > 0))
then
	cp <JOB_NAME>.<DATA_PREFIX>.<SEED>.<FOLD>_{gen,disc}_iter_$CONT_ITER.{caffemodel,solverstate} $SCR_DIR
	cp <JOB_NAME>.<DATA_PREFIX>.<SEED>.<FOLD>.training_output $SCR_DIR
fi

cd $SCR_DIR

trap "cp *.{model,solver,caffemodel,solverstate,training_output,png,pdf} ${SLURM_SUBMIT_DIR}" EXIT

python3 $LIGAN_ROOT/train.py \
	-d ../../<MODEL_DIR>/<DATA_MODEL_NAME>.model \
	-g ../../<MODEL_DIR>/<GEN_MODEL_NAME>.model \
	-a ../../<MODEL_DIR>/<DISC_MODEL_NAME>.model \
	-s ../../<SOLVER_DIR>/<SOLVER_NAME>.solver \
	-p ../../<DATA_DIR>/<DATA_PREFIX> \
	-r <DATA_ROOT> \
	-n <FOLD> \
	--random_seed <SEED> \
	--max_iter <MAX_ITER> \
	--cont_iter $CONT_ITER \
	--gen_train_iter <GEN_TRAIN_ITER> \
	--disc_train_iter <DISC_TRAIN_ITER> \
	--test_interval <TEST_INTERVAL> \
	--test_iter <TEST_ITER> \
	--snapshot <SNAPSHOT_ITER> \
	--instance_noise <INSTANCE_NOISE> \
	--loss_weight <LOSS_WEIGHT> \
	--loss_weight_decay <LOSS_WEIGHT_DECAY> \
	<TRAIN_OPTIONS> \
	-o <JOB_NAME>.<DATA_PREFIX>.<SEED>

exit
