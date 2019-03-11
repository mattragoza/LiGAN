NAME=ARC2019/
DATA=lowrmsd
ROOT=~/dvorak/net/pulsar/home/koes/dkoes/PDBbind/refined-set/
FOLD=all
ITER=60000
TARGET=2vfk

GAN=adam0_2_2_b_0.0_vl-le13_24_0.5_3_2l_32_2_1024_e_d11_24_3_1l_16_2_x
BASE_CMD="python generate.py -d ${GAN}/data*.model -g ${GAN}/_*.model -w ${GAN}/${GAN}.${DATA}.0.${FOLD}_gen_iter_${ITER}.caffemodel --data_root ${ROOT} --gpu --output_dx " #--fit_atom_types --max_iter 100 --output_sdf --n_fit_workers 1"

${BASE_CMD} -o ${NAME}/AE -b lig -b lig_gen --n_samples 10 \
	-r ${TARGET}/${TARGET}_rec.gninatypes -l ${TARGET}/${TARGET}_min.gninatypes 

GAN=adam0_2_2_b_0.0_vr-le13_24_0.5_3_2l_32_2_1024_e_d11_24_3_1l_16_2_x
BASE_CMD="python generate.py -d ${GAN}/data*.model -g ${GAN}/_*.model -w ${GAN}/${GAN}.${DATA}.0.${FOLD}_gen_iter_${ITER}.caffemodel --data_root ${ROOT} --gpu --output_dx " #--fit_atom_types --max_iter 100 --output_sdf --n_fit_workers 1"

${BASE_CMD} -o ${NAME}/CE -b rec -b lig_gen --n_samples 10 \
	-r ${TARGET}/${TARGET}_rec.gninatypes -l ${TARGET}/${TARGET}_min.gninatypes 

cat ${NAME}/AE.pymol ${NAME}/CE.pymol > ${NAME}/DENSITY.pymol
echo load ${ROOT}/${TARGET}/${TARGET}_rec.pdb >> ${NAME}/DENSITY.pymol
echo load ${ROOT}/${TARGET}/${TARGET}_min.sdf >> ${NAME}/DENSITY.pymol

exit
