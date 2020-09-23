python ../models.py disc_model.params -o models -n 'disc_{loss_types}'
python ../models.py gen_model.params  -o models -n 'gen_{loss_types}'
python ../job_scripts.py job.params -b bridges_train.sh -o loss_types -n '{gen_model_name}_{disc_model_name}_{seed}'

