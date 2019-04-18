#!/bin/bash
#SBATCH --job-name=<JOB_NAME>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --cluster=gpu
#SBATCH --partition=<QUEUE>
#SBATCH --gres=gpu:1
#SBATCH --mem=<MEMORY>
#SBATCH --time=<WALLTIME>
#SBATCH --qos=normal
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
<JOB_PARAMS>
source ~/.bashrc
cd $SLURM_SUBMIT_DIR

cp <DATA_DIR>/<DATA_NAME>*.types $SLURM_SCRATCH
cp <MODEL_DIR>/<DATA_MODEL_NAME>.model $SLURM_SCRATCH
cp <MODEL_DIR>/<GEN_MODEL_NAME>.model $SLURM_SCRATCH
cp <MODEL_DIR>/<DISC_MODEL_NAME>.model $SLURM_SCRATCH
cp <SOLVER_DIR>/<SOLVER_NAME>.solver $SLURM_SCRATCH

CONT_ITER=<CONT_ITER>
if (($CONT_ITER > 0)); then
	cp <JOB_NAME>.*_iter_<CONT_ITER>.solverstate $SLURM_SCRATCH
	cp <JOB_NAME>.*_iter_<CONT_ITER>.caffemodel $SLURM_SCRATCH
	cp <JOB_NAME>.training_output $SLURM_SCRATCH
fi

cd $SLURM_SCRATCH

trap "cp *.{model,solver,caffemodel,solverstate,training_output,png,pdf} ${SLURM_SUBMIT_DIR}" EXIT

python <LIGAN_DIR>/train.py -d <DATA_MODEL_NAME>.model -g <GEN_MODEL_NAME>.model -a <DISC_MODEL_NAME>.model -p <DATA_NAME> -r <DATA_ROOT> -n <FOLD> --random_seed <SEED> -s <SOLVER_NAME>.solver --max_iter <MAX_ITER> --cont_iter <CONT_ITER> --gen_train_iter <GEN_TRAIN_ITER> --disc_train_iter <DISC_TRAIN_ITER> --test_interval <TEST_INTERVAL> --test_iter <TEST_ITER> --instance_noise <INSTANCE_NOISE> --loss_weight <LOSS_WEIGHT> --loss_weight_decay <LOSS_WEIGHT_DECAY> <TRAIN_OPTIONS> -o <JOB_NAME>.<DATA_NAME>.<SEED>
exit
