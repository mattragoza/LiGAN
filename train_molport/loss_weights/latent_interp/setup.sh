python3 ../../../job_scripts.py vae.params --template csb_fit_cmd.sh -o . -n 'latent_interp_{gen_model_name}_{disc_model_name}_{gen_options}_{data_name}'
python3 ../../../job_scripts.py ae.params --template csb_fit_cmd.sh -o . -n 'latent_interp_{gen_model_name}_{disc_model_name}_{gen_options}_{data_name}'
ls latent_interp_gen*/csb_fit_cmd.sh > vae_job_scripts
ls latent_interp_ae*/csb_fit_cmd.sh > ae_job_scripts
cat vae_job_scripts ae_job_scripts > all_job_scripts
chmod +x `cat all_job_scripts`

