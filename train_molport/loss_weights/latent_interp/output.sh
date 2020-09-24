python3 ../../../job_errors.py $(cat all_job_scripts | cut -d/ -f1) --job_type fit --array_job --submitted 4 --output_file latent_interp.gen_metrics
