#python3 ../../models.py gen_model.params -o models -n "{encode_type}"
python3 ../../job_scripts.py job.params -t csb_train.sh -o train -n "{gen_model_name}"

