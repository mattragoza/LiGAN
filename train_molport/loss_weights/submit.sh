#python3 ../../submit_job.py */crc_train.sh
#python3 ../../submit_job.py $(cat unfinished_100k_job_scripts)
#python3 ../../submit_job.py $(cat unfinished_100k_job_scripts2)
python3 ../../submit_job.py wgan_trial0/csb_train_expt.sh --array 1-90%10
