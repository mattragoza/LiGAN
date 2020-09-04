python3 ../../job_scripts.py post.params  -b csb_fit.sh -o . -n '{gen_model_name}_{disc_model_name}_{train_seed}_{train_iter}_{gen_options}_{atom_init}'
python3 ../../job_scripts.py prior.params -b csb_fit.sh -o . -n '{gen_model_name}_{disc_model_name}_{train_seed}_{train_iter}_{gen_options}_{atom_init}'

