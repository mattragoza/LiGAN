#!/bin/bash
#SBATCH --job-name=fit_molport_<JOB_NAME>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --cluster=gpu
#SBATCH --partition=gtx1080
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=1:00:00
#SBATCH --qos=normal
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
<JOB_PARAMS>
source ~/.bashrc
cd $SLURM_SUBMIT_DIR

SCR_DIR=$SLURM_SUBMIT_DIR/$SLURM_ARRAY_JOB_ID
if [[ ! -e $SCR_DIR ]]; then
        mkdir -p $SCR_DIR
fi

echo ================================
echo Running on `whoami`@`hostname`
echo work_dir `pwd`
echo scr_dir $SCR_DIR
echo ld_library_path $LD_LIBRARY_PATH
echo ================================

TRAIN_JOB_NAME=<GEN_MODEL_NAME>_<DISC_MODEL_NAME>_<TRAIN_SEED>
GEN_WEIGHTS_NAME=$TRAIN_JOB_NAME.<TRAIN_DATA_NAME>.<TRAIN_SEED>.<TRAIN_FOLD>_gen_iter_<TRAIN_ITER>

DATA_FILE=<DATA_DIR>/<DATA_NAME>.types
if [${SLURM_ARRAY_TASK_ID} -gt $(wc -l ${DATA_FILE}) ];
then
	exit
fi
LINE=$(head ${DATA_FILE} -n${SLURM_ARRAY_TASK_ID} | tail -n1)
REC=$(echo $LINE | cut -d' ' -f 3)
LIG=$(echo $LINE | cut -d' ' -f 4)

cp $LIGAN_ROOT/my_{rec,lig}_map $SCR_DIR

cd $SCR_DIR

trap "cp *.{types,model,caffemodel,dx,sdf,pymol,gen_metrics} ${SLURM_SUBMIT_DIR}" EXIT

python3 $LIGAN_ROOT/generate.py --data_model ../<MODEL_DIR>/<DATA_MODEL_NAME>.model --gen_model ../<MODEL_DIR>/<GEN_MODEL_NAME>.model --gen_weights ../<WEIGHTS_DIR>/${GEN_WEIGHTS_NAME}.caffemodel -r ${REC} -l ${LIG} --data_root ../../../data -b lig -b lig_gen --fit_atoms --output_sdf --output_channels --verbose 3 --gpu <GEN_OPTIONS> --n_samples <N_SAMPLES> --atom_init <ATOM_INIT> --r_factor <R_FACTOR> --beam_size <BEAM_SIZE> --beam_stride <BEAM_STRIDE> --learning_rate <LEARNING_RATE> --beta1 <BETA1> --beta2 <BETA2> --weight_decay <WEIGHT_DECAY> --interm_iters <INTERM_ITERS> --final_iters <FINAL_ITERS> -o <JOB_NAME>_${SLURM_ARRAY_TASK_ID}
exit
