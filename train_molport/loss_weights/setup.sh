python3 ../../models.py gen_model.params -n 'gen_{loss_types}_{loss_weight_KL}_{loss_weight_L2}' -o models
#python3 ../../models.py disc_model.params -n 'disc_{loss_types}_{loss_weight_log}' -o models
python3 ../../models.py wgan_disc_model.params -n 'disc_{loss_types}_{loss_weight_wass}' -o models
#python3 ../../job_scripts.py job.params -t csb_train.sh -n '{gen_model_name}_{disc_model_name}_{seed}'
python3 ../../job_scripts.py wgan_job.params -t csb_train_cmd.sh -n '{gen_model_name}_{disc_model_name}_{disc_train_iter}'

