# CRC loss weight sweep
#python3 ../../../submit_job.py --array=1-10 $(cat all_job_scripts)
#python3 ../../../submit_job.py --array=1-10 $(cat tail_job_scripts)
#python3 ../../../submit_job.py --array=1-10 $(cat tail_job_scripts2)
#python3 ../../../submit_job.py --array=1-10 $(cat tail_job_scripts3)
#python3 ../../../submit_job.py --array=1-10 $(cat tail_job_scripts4)
#python3 ../../../submit_job.py --array=1-10 $(cat tail_job_scripts5)

# begin full train evaluations (this was too slow, moved to CSB)
#python3 ../../../submit_job.py --array=11-100%20 $(cat best_job_scripts)
#python3 ../../../submit_job.py --array=11-100%20 $(cat tail_best_job_scripts)

# cancelled the following jobs to try moving to gtx1080, but then moved back to titanx
#python3 ../../../submit_job.py --array=71-100%20 gen_e_0.1_1_disc_x_10_0_70000_pr/crc_fit.sh
#python3 ../../../submit_job.py --array=11-100%20 gen_e_0.1_1_disc_x_10_0_80000_pr/crc_fit.sh
#python3 ../../../submit_job.py --array=11-100%20 gen_e_0.1_1_disc_x_10_0_90000_r/crc_fit.sh
#python3 ../../../submit_job.py --array=11-100%20 gen_e_0.1_1_disc_x_10_0_90000_pr/crc_fit.sh
#python3 ../../../submit_job.py --array=11-100%20 gen_e_0.1_1_disc_x_10_0_100000_r/crc_fit.sh
#python3 ../../../submit_job.py --array=11-100%20 gen_e_0.1_1_disc_x_10_0_100000_pr/crc_fit.sh

# CSB finish train set fit evaluations
python3 ../../../submit_job.py --array=1-30000%20 array_refactor/csb_fit_expt.sh

