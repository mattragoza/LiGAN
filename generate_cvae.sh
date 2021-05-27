REC_FILE=$1 # e.g. data/crossdock2020/PARP1_HUMAN_775_1012_0/2rd6_A_rec.pdb
LIG_FILE=$2 # e.g. data/crossdock2020/PARP1_HUMAN_775_1012_0/2rd6_A_rec_2rd6_78p_lig_tt_min.sdf

python3 generate.py \
	--data_model_file models/data_48_0.5_crossdock.model \
	--gen_model_file models/cvae.model \
	--gen_weights_file weights/lessskip_crossdocked_increased_1.lowrmsd.0_gen_iter_1500000.caffemodel \
	--rec_file $REC_FILE \
	--lig_file $LIG_FILE \
	--out_prefix CVAE \
	--n_samples 10 \
	--fit_atoms \
	--dkoes_make_mol \
	--output_sdf \
	--output_dx \
	--gpu
