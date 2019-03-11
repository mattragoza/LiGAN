NAME=ARC2019/
DATA=lowrmsd
ROOT=~/dvorak/net/pulsar/home/koes/dkoes/PDBbind/refined-set/
FOLD=0
ITER=20000
TARGET=2vfk

GAN=ce12_24_0.5_3_2_64_3_ec
BASE_CMD="python generate.py -d models/data_24_0.5.model -g models/_${GAN}.model -w v12/${GAN}/${GAN}.${DATA}.0.${FOLD}_iter_${ITER}.caffemodel --data_root ${ROOT} --gpu --output_dx " #--fit_atom_types --max_iter 100 --output_sdf --n_fit_workers 1"

${BASE_CMD} -o ${NAME}/CE -b rec -b lig -b lig_gen --n_samples 10 \
	-r ${TARGET}/${TARGET}_rec.gninatypes -l ${TARGET}/${TARGET}_min.gninatypes 

exit
