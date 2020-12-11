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
cp <DATA_DIR>/<DATA_NAME>.types $SCR_DIR
cp <MODEL_DIR>/<DATA_MODEL_NAME>.model $SCR_DIR
cp <MODEL_DIR>/<GEN_MODEL_NAME>.model $SCR_DIR
cp <WEIGHTS_DIR>/<GEN_WEIGHTS_NAME>.caffemodel $SCR_DIR

cd $SCR_DIR

trap "cp *.{types,model,caffemodel,dx,sdf,pymol,gen_metrics} ${SLURM_SUBMIT_DIR}" EXIT

python3 <LIGAN_DIR>/generate.py --data_model <DATA_MODEL_NAME>.model --gen_model <GEN_MODEL_NAME>.model --gen_weights <GEN_WEIGHTS_NAME>.caffemodel --data_file <DATA_NAME>.types --data_root <DATA_ROOT> -b lig -b lig_gen --fit_atoms --output_sdf --verbose 3 --gpu <GEN_OPTIONS> --n_samples <N_SAMPLES> --atom_init <ATOM_INIT> --r_factor <R_FACTOR> --beam_size <BEAM_SIZE> --beam_stride <BEAM_STRIDE> --learning_rate <LEARNING_RATE> --beta1 <BETA1> --beta2 <BETA2> --weight_decay <WEIGHT_DECAY> --interm_iters <INTERM_ITERS> --final_iters <FINAL_ITERS> -o <JOB_NAME>
exit
