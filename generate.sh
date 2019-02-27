NAME=TEST
DATA=lowrmsd
ROOT=/net/pulsar/home/koes/tmasuda/liGAN/PDBbind/refined-set/
FOLD=all
ITER=50000

GAN=adam0_2_2_b_0.0_vl-le13_24_0.5_3_2lid_32_2_1024_e_d11_24_3_1l_16_2_x
GEN=$(echo ${GAN} | sed 's/.*_\(v.*-le13_24_0.5.*\)_d11.*/\1/')
ENC=lig


echo ${GAN}
echo ${GEN}

TARGET=1ai5
BASE_FLAGS="-m models/${GEN}.model -w ${GAN}/${GAN}.${DATA}.0.all_gen_iter_${ITER}.caffemodel -r ${TARGET}/${TARGET}_rec.pdb -l ${TARGET}/${TARGET}_min.sdf --data_root ${ROOT} --gpu --verbose 3 --output_dx"

python generate.py ${BASE_FLAGS} -b lig_gen -b lig -o ${NAME}_POST_SAMPLE  --n_samples 10

python generate.py ${BASE_FLAGS} -b lig     -o ${NAME}_POST_TRUE    --n_samples 10

python generate.py ${BASE_FLAGS} -b lig_gen -o ${NAME}_POST_MEAN    --n_samples 10 --forward_from ${ENC}_latent_noise -f ${ENC}_latent_std:0

python generate.py ${BASE_FLAGS} -b lig_gen -o ${NAME}_POST_ROTATE  --n_samples 10 --forward_from ${ENC}_latent_noise -f ${ENC}_latent_std:0 --random_rotation

python generate.py ${BASE_FLAGS} -b lig_gen -o ${NAME}_PRIOR_SAMPLE --n_samples 10 --forward_from ${ENC}_latent_noise -f ${ENC}_latent_mean:0 -f ${ENC}_latent_std:1

python generate.py ${BASE_FLAGS} -b lig_gen -o ${NAME}_PRIOR_MEAN   --n_samples 10 --forward_from ${ENC}_latent_noise -f ${ENC}_latent_mean:0 -f ${ENC}_latent_std:0

python generate.py ${BASE_FLAGS} -b lig_gen -o ${NAME}_PRIOR_ROTATE --n_samples 10 --forward_from ${ENC}_latent_noise -f ${ENC}_latent_mean:0 -f ${ENC}_latent_std:0 --random_rotation

cat ${NAME}_{POST,PRIOR}_{TRUE,SAMPLE,MEAN,ROTATE}.pymol | grep load_group >  ${NAME}.pymol
echo load /net/pulsar/home/koes/tmasuda/liGAN/PDBbind/refined-set/${TARGET}/${TARGET}_rec.pdb >> ${NAME}.pymol
echo load /net/pulsar/home/koes/tmasuda/liGAN/PDBbind/refined-set/${TARGET}/${TARGET}_min.sdf >> ${NAME}.pymol

exit
