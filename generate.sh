NAME=test_gen_refactor
DATA=lowrmsd
ROOT=~dkoes/PDBbind/refined-set/ #~/dvorak/net/pulsar/home/koes/mtr22/gan/data/
FOLD=all
ITER=60000

GAN=adam0_2_2_b_0.0_vl-le13_24_0.5_3_2l_32_2_1024_e_d11_24_3_1l_16_2_x
GEN=$(echo ${GAN} | sed 's/.*\(_v.*-le13_24_0.5.*\)_d11.*/\1/')

echo ${GAN}
echo ${GEN}

TARGET=1ajv
BASE_FLAGS="-d ${GAN}/data*.model -g ${GAN}/${GEN}.model -w ${GAN}/${GAN}.${DATA}.0.${FOLD}_gen_iter_${ITER}.caffemodel -r ${TARGET}/${TARGET}_rec.gninatypes -l ${TARGET}/${TARGET}_ligand_0.gninatypes --data_file data/lowrmsd.types --data_root ${ROOT} --gpu --fit_atom_types --max_iter 100"

python generate.py ${BASE_FLAGS} -o ${NAME}/post_sample  --n_samples 5
#python generate.py ${BASE_FLAGS} -o ${NAME}/post_mean    --n_samples 3 --mean
#python generate.py ${BASE_FLAGS} -o ${NAME}/prior_sample --n_samples 3 --prior
#python generate.py ${BASE_FLAGS} -o ${NAME}/prior_mean   --n_samples 3 --prior --mean

cat ${NAME}/{post,prior}_{mean,sample}.pymol | grep load >  ${NAME}/all.pymol
echo load ../../dkoes/PDBbind/refined-set/${TARGET}/${TARGET}_rec.pdb >> ${NAME}/all.pymol
echo load ../../dkoes/PDBbind/refined-set/${TARGET}/${TARGET}_min.sdf >> ${NAME}/all.pymol

exit
