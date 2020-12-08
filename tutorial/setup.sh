python3 ../models.py data_model.params -o models -n data
python3 ../models.py gen_model.params -o models -n gen_{encode_type}
python3 ../models.py disc_model.params -o models -n disc
python3 ../solvers.py solver.params -o solvers -n adam0
python3 ../job_scripts.py job.params -t csb_train_cmd.sh -n train_{gen_model_name} | tee trial0_job_scripts
chmod +x train_*/csb_train_cmd.sh
python3 ../job_scripts.py expt.params -t csb_train_expt.sh -n trial{trial_num}

