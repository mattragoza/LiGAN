# previously submitted
#python3 ../../submit_job.py --array=1-1000 ae_disc_x_0_{1,2,3,4,5,6}0000_r_none/csb_fit.sh
#python3 ../../submit_job.py --array=1-1000 gen_e_disc_x_0_{1,2,3,4,5,6}0000_r_none/csb_fit.sh
#python3 ../../submit_job.py --array=1-1000 gen_e_disc_x_0_{1,2,3,4,5,6}0000_pr_none/csb_fit.sh # these were only 1 sample

# now running
#python3 ../../submit_job.py --array=1-100 ae_disc_x_0_{7,8,9,10}0000_r_none/csb_fit.sh
#python3 ../../submit_job.py --array=1-100 gen_e_disc_x_0_{7,8}0000_r_none/csb_fit.sh
#python3 ../../submit_job.py --array=1-100 gen_e_disc_x_0_{1,2,3,4,5,6,7,8}0000_pr_none/csb_fit.sh # now with 10 samples

# running after those
#python3 ../../submit_job.py --array=101-1000 ae_disc_x_0_{7,8,9,10}0000_r_none/csb_fit.sh
#python3 ../../submit_job.py --array=101-1000 gen_e_disc_x_0_{7,8}0000_r_none/csb_fit.sh
#python3 ../../submit_job.py --array=101-1000 gen_e_disc_x_0_{1,2,3,4,5,6,7,8}0000_pr_none/csb_fit.sh # now with 10 samples

# no longer waiting for weights
#python3 ../../submit_job.py --array=1-100 gen_e_disc_x_0_{9,10}0000_r_none/csb_fit.sh
#python3 ../../submit_job.py --array=1-100 gen_e_disc_x_0_{9,10}0000_pr_none/csb_fit.sh
#python3 ../../submit_job.py --array=101-1000 gen_e_disc_x_0_{9,10}0000_r_none/csb_fit.sh
#python3 ../../submit_job.py --array=101-1000 gen_e_disc_x_0_{9,10}0000_pr_none/csb_fit.sh

#TODO wgan models
#python3 ../../submit_job.py --array=1-1000 ae_disc_w_0_{1,2,3,4,5,6,7,8,9,10}0000_r_none/csb_fit.sh
#python3 ../../submit_job.py --array=1-1000 gen_e_disc_w_0_{1,2,3,4,5,6,7,8,9,10}0000_r_none/csb_fit.sh
#python3 ../../submit_job.py --array=1-1000 gen_e_disc_w_0_{1,2,3,4,5,6,7,8,9,10}0000_pr_none/csb_fit.sh

# continuing to train VAE on Bridges, fitting on CRC
#python3 ../../submit_job.py --array=1-100 gen_e_disc_x_0_1{1,2,3,4,5,6,7}0000_r_none/crc_fit.sh
#python3 ../../submit_job.py --array=1-100 gen_e_disc_x_0_1{1,2,3,4,5,6,7}0000_pr_none/crc_fit.sh
#python3 ../../submit_job.py --array=101-1000 gen_e_disc_x_0_1{1,2,3,4,5,6,7}0000_r_none/crc_fit.sh
#python3 ../../submit_job.py --array=101-1000 gen_e_disc_x_0_1{1,2,3,4,5,6,7}0000_pr_none/crc_fit.sh
#python3 ../../submit_job.py --array=1-100 gen_e_disc_x_0_1{4,5,6,7,8,9}0000_r_none/crc_fit.sh
#python3 ../../submit_job.py --array=1-100 gen_e_disc_x_0_1{4,5,6,7,8,9}0000_pr_none/crc_fit.sh
#python3 ../../submit_job.py --array=101-1000 gen_e_disc_x_0_1{4,5,6,7,8,9}0000_r_none/crc_fit.sh
#python3 ../../submit_job.py --array=101-1000 gen_e_disc_x_0_1{4,5,6,7,8,9}0000_pr_none/crc_fit.sh
python3 ../../submit_job.py --array=1-1000 gen_e_disc_x_0_200000_{r,pr}_none/crc_fit.sh

