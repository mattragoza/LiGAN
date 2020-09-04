#CSB
#python3 ../../submit_job.py -a 1-100%20 $(cat job_scripts_GPU)
#python3 ../../submit_job.py -a 1-100 $(cat job_scripts_CPU)
#python3 ../../submit_job.py -a 1-100%20 $(cat job_scripts_GPU_2)
#python3 ../../submit_job.py -a 1-10%10 $(cat new_job_scripts_after_debug)
python3 ../../submit_job.py -a 1-2160%20 array_refactor/csb_fit_expt.sh
#CRC
#python3 ../../submit_job.py -a 1-10 ae*/crc_fit.sh
#python3 ../../submit_job.py -a 11-20 ae*/crc_fit.sh
#python3 ../../submit_job.py -a 1-10 $(cat very_low_threshold_job_scripts)
python3 ../../submit_job.py -a 1-10 $(cat threshold_0.2_job_scripts)
python3 ../../submit_job.py -a 1-10 $(cat multi_atom_job_scripts)
