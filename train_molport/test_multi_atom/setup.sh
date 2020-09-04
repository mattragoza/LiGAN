#python3 ../../job_scripts.py job.params -b csb_fit.sh     -o . -n '{gen_model_name}_{disc_model_name}_{train_seed}_{train_iter}_{gen_options}_{threshold}_{peak_value}_{min_dist}_{interm_gd_iters}_{final_gd_iters}'
#python3 ../../job_scripts.py job.params -b csb_fit_CPU.sh -o . -n '{gen_model_name}_{disc_model_name}_{train_seed}_{train_iter}_{gen_options}_{threshold}_{peak_value}_{min_dist}_{interm_gd_iters}_{final_gd_iters}'
python3 ../../job_scripts.py job.params -b csb_fit_cmd.sh -o . -n '{gen_model_name}_{disc_model_name}_{train_seed}_{train_iter}_{gen_options}_{threshold}_{peak_value}_{min_dist}_{interm_gd_iters}_{final_gd_iters}'

