#python3 ../../models.py gen_model.params -o models -n "{encode_type}" --benchmark 10 --gpu
python3 ../../job_scripts.py job.params -t csb_train.sh -n "train_{gen_model_name}"

