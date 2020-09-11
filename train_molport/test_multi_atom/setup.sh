#CSB
#python3 ../../job_scripts.py job.params -t csb_fit.sh -n '{gen_model_name}_{disc_model_name}_{train_seed}_{train_iter}_{gen_options}_{threshold}_{peak_value}_{min_dist}_{interm_gd_iters}_{final_gd_iters}'
#python3 ../../job_scripts.py job.params -t csb_fit_CPU.sh -n '{gen_model_name}_{disc_model_name}_{train_seed}_{train_iter}_{gen_options}_{threshold}_{peak_value}_{min_dist}_{interm_gd_iters}_{final_gd_iters}'
python3 ../../job_scripts.py job.params -t csb_fit_cmd.sh -n '{gen_model_name}_{disc_model_name}_{train_seed}_{train_iter}_{gen_options}_{threshold}_{peak_value}_{min_dist}_{interm_gd_iters}_{final_gd_iters}'
#CRC
#python3 ../../job_scripts.py single.params -t crc_fit.sh -n '{gen_model_name}_{disc_model_name}_{train_seed}_{train_iter}_{gen_options}_{threshold}_{peak_value}_{min_dist}_{interm_gd_iters}_{final_gd_iters}'
#python3 ../../job_scripts.py multi.params -t crc_fit.sh -n '{gen_model_name}_{disc_model_name}_{train_seed}_{train_iter}_{gen_options}_{threshold}_{peak_value}_{min_dist}_{interm_gd_iters}_{final_gd_iters}'
python3 ../../job_scripts.py job.params -t crc_fit.sh -n '{gen_model_name}_{disc_model_name}_{train_seed}_{train_iter}_{gen_options}_{threshold}_{peak_value}_{min_dist}_{interm_gd_iters}_{final_gd_iters}'

