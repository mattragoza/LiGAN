#python3 ../../submit_job.py -a 1-10 ae*/crc_fit.sh
#python3 ../../submit_job.py -a 11-20 ae*/crc_fit.sh
#python3 ../../submit_job.py -a 1-10 $(cat very_low_threshold_job_scripts)
python3 ../../submit_job.py -a 1-10 $(cat threshold_0.2_job_scripts)
python3 ../../submit_job.py -a 1-10 $(cat multi_atom_job_scripts)
