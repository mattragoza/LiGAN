NAME=ARC2019/EXAM
DATA=lowrmsd
ROOT=/net/pulsar/home/koes/dkoes/PDBbind/refined-set/
FOLD=all
ITER=60000

GAN=adam0_2_2_b_0.0_vl-le13_24_0.5_3_2l_32_2_1024_e_d11_24_3_1l_16_2_x

BASE_CMD="python generate.py -d ${GAN}/data*.model -g ${GAN}/_*.model -w ${GAN}/${GAN}.${DATA}.0.${FOLD}_gen_iter_${ITER}.caffemodel --data_root ${ROOT} --gpu --output_dx --fit_atom_types --max_iter 100 --output_sdf"

${BASE_CMD} -o ${NAME} \
	-r 3h78/3h78_rec.gninatypes -l 3h78/3h78_min.gninatypes \
	-r 4jx9/4jx9_rec.gninatypes -l 4jx9/4jx9_min.gninatypes \
	-r 3igp/3igp_rec.gninatypes -l 3igp/3igp_min.gninatypes \
	-r 4cwf/4cwf_rec.gninatypes -l 4cwf/4cwf_min.gninatypes \
	-r 1odi/1odi_rec.gninatypes -l 1odi/1odi_min.gninatypes \
	-r 1nli/1nli_rec.gninatypes -l 1nli/1nli_min.gninatypes \
	-r 4a95/4a95_rec.gninatypes -l 4a95/4a95_min.gninatypes \
	-r 4i74/4i74_rec.gninatypes -l 4i74/4i74_min.gninatypes \
	-r 2a5c/2a5c_rec.gninatypes -l 2a5c/2a5c_min.gninatypes \
	-r 1f4e/1f4e_rec.gninatypes -l 1f4e/1f4e_min.gninatypes \
