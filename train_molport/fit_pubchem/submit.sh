
# AE, VAE posterior (now running on CRC)
#python ../../submit_job.py --array=1-14  {ae,gen_e}_disc_x_0_100000_r_pubchem_diff0.0/crc_fit.sh
#python ../../submit_job.py --array=1-100 {ae,gen_e}_disc_x_0_100000_r_pubchem_diff0.{1,2,3,4,5,6,7,8,9}/crc_fit.sh
#python ../../submit_job.py --array=101-1000 ae_disc_x_0_100000_r_pubchem_diff0.{1,2,3,4,5,6,7,8,9}/crc_fit.sh

# rest of VAE posterior (now running on Bridges)
#python3 ../../submit_job.py --array=101-1000 gen_e_disc_x_0_100000_r_pubchem_diff0.{1,2,3,4,5,6,7,8,9}/bridges_fit.sh

# VAE prior (running on CRC and CSB, in diff orders)
python3 ../../submit_job.py --array=1-1000 gen_e_disc_x_0_100000_pr_pubchem_diff0.{9,8,7,6,5,4,3,2,1}/crc_fit.sh
python3 ../../submit_job.py --array=1-14   gen_e_disc_x_0_100000_pr_pubchem_diff0.0/crc_fit.sh

