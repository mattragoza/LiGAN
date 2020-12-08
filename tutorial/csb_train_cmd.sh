#!/bin/bash
<JOB_PARAMS>
source ~/.bashrc

WORK_DIR=$1
SEED=$2
JOB_ID=$3

cd $WORK_DIR

SCR_DIR=$WORK_DIR/$JOB_ID
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
	cp <JOB_NAME>_$SEED_{gen,disc}_iter_$CONT_ITER.{caffemodel,solverstate} $SCR_DIR
	cp <JOB_NAME>_$SEED.training_output $SCR_DIR
fi

cd $SCR_DIR

trap "cp *.{model,solver,caffemodel,solverstate,training_output,png,pdf} $WORK_DIR" EXIT

python3 ../../<LIGAN_DIR>/train.py \
	-d ../../<MODEL_DIR>/<DATA_MODEL_NAME>.model \
	-g ../../<MODEL_DIR>/<GEN_MODEL_NAME>.model \
	-a ../../<MODEL_DIR>/<DISC_MODEL_NAME>.model \
	-s ../../<SOLVER_DIR>/<SOLVER_NAME>.solver \
	-p ../../<DATA_DIR>/<DATA_PREFIX> \
	-r ../../<DATA_ROOT> \
	-n <FOLD> \
	--random_seed $SEED \
	--max_iter <MAX_ITER> \
	--cont_iter $CONT_ITER \
	--test_interval <TEST_INTERVAL> \
	--test_iter <TEST_ITER> \
	--snapshot <SNAPSHOT_ITER> \
	<TRAIN_OPTIONS> \
	-o <JOB_NAME>_$SEED

exit
