NAME=ASDF
DATA=lowrmsd
ROOT=~dkoes/PDBbind/refined-set/ #~/dvorak/net/pulsar/home/koes/mtr22/gan/data/
FOLD=0
ITER=100000

GAN=adam0_2_2__0.0_1.0_1e-05_vl-le13_24_0.5_3_2l_32_2_1024_e_d11_24_3_1l_16_2_x
GEN=$(echo ${GAN} | sed 's/.*\(_v.*-le13_24_0.5.*\)_d11.*/\1/')

echo ${GAN}
echo ${GEN}

TARGET=2vfk
BASE_FLAGS="-d ${GAN}/data*.model -g ${GAN}/${GEN}.model -w ${GAN}/${GAN}.${DATA}.0.${FOLD}_gen_iter_${ITER}.caffemodel -r ${TARGET}/${TARGET}_rec.gninatypes -l ${TARGET}/${TARGET}_min.gninatypes --data_root ${ROOT} --gpu --output_dx"

python generate.py ${BASE_FLAGS} -o ${NAME}_post_mean    --mean
python generate.py ${BASE_FLAGS} -o ${NAME}_post_sample  --n_samples 10
python generate.py ${BASE_FLAGS} -o ${NAME}_prior_mean   --prior --mean
python generate.py ${BASE_FLAGS} -o ${NAME}_prior_sample --prior --n_samples 10

cat ${NAME}_{post,prior}_{mean,sample}.pymol | grep load >  ${NAME}_all.pymol
echo load ../../dkoes/PDBbind/refined-set/${TARGET}/${TARGET}_rec.pdb >> ${NAME}_all.pymol
echo load ../../dkoes/PDBbind/refined-set/${TARGET}/${TARGET}_min.sdf >> ${NAME}_all.pymol

exit
