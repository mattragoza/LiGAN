python3 ../../models.py gen_model.params -o models -n "gen_{loss_weight_KL}_{loss_weight_L2}"
python3 ../../models.py disc_model.params -o models -n "disc_{loss_weight_log}"
#python3 ../../job_scripts.py job.params -t csb_train.sh -o train -n "{gen_model_name}"

