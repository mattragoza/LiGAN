#python3 ../../submit_job.py -a 1-100%20 $(cat job_scripts_GPU)
#python3 ../../submit_job.py -a 1-100 $(cat job_scripts_CPU)
#python3 ../../submit_job.py -a 1-100%20 $(cat job_scripts_GPU_2)
#python3 ../../submit_job.py -a 1-10%10 $(cat new_job_scripts_after_debug)
python3 ../../submit_job.py -a 1-2160%20 array_refactor/csb_fit_expt.sh
