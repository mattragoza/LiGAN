LIG_FILE=$1 # e.g. data/molport/0/102906000_8.sdf

python3 generate.py \
	--data_model_file models/data_48_0.5_molport.model \
	--gen_model_file models/vae.model \
	--gen_weights_file weights/gen_e_0.1_1_disc_x_10_0.molportFULL_rand_.0.0_gen_iter_100000.caffemodel \
	--rec_file data/molport/10gs_rec.pdb \
	--lig_file $LIG_FILE \
	--out_prefix VAE \
	--n_samples 10 \
	--fit_atoms \
	--dkoes_make_mol \
	--output_sdf \
	--output_dx \
	--gpu
