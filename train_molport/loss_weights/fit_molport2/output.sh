#python3 ../../../job_errors.py --job_type fit --array_job --submitted 1-100 $(cat all_job_dirs) --output_file crc_fit_molport2.gen_metrics
python3 ../../../job_errors.py --job_type fit --array_job --submitted 1-30000 $(cat all_job_scripts | cut -d"/" -f1) --output_file csb_fit_molport2.gen_metrics
