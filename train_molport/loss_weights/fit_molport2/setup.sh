#python3 ../../../job_scripts.py vae.params -b crc_fit.sh -o . -n '{gen_model_name}_{disc_model_name}_{train_seed}_{train_iter}_{gen_options}'
python3 ../../../job_scripts.py vae.params -b csb_fit_cmd.sh -o . -n '{gen_model_name}_{disc_model_name}_{train_seed}_{train_iter}_{gen_options}'
python3 ../../../job_scripts.py ae.params -b csb_fit_cmd.sh -o . -n '{gen_model_name}_{disc_model_name}_{train_seed}_{train_iter}_{gen_options}'

