JOB_SCRIPTS_FILE=trial0_job_scripts
n_total=0
for job_script in $(cat $JOB_SCRIPTS_FILE)
do
	job_dir=$(dirname $job_script)
	n_scr_dirs=$(ls $job_dir/*/ -d -1 -tr | wc -l)
	n_total=$(($n_total + $n_scr_dirs))
done
echo $n_total dirs to copy back
n_done=0
for job_script in $(cat $JOB_SCRIPTS_FILE)
do
	job_dir=$(dirname $job_script)
	echo $job_dir
	for scr_dir in $(ls $job_dir/*/ -d -1 -tr)
	do
		find $scr_dir -name "*.channels"    -exec cp {} $job_dir \;
		find $scr_dir -name "*.sdf"         -exec cp {} $job_dir \;
		find $scr_dir -name "*.pymol"       -exec cp {} $job_dir \;
		find $scr_dir -name "*.gen_metrics" -exec cp {} $job_dir \;
		n_done=$(($n_done + 1))
		echo "["$n_done"/"$n_total"] "$scr_dir
	done
done
