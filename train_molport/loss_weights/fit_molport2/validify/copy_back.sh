n_total=0
for job_dir in $(cat all_job_scripts | cut -d/ -f1)
do
	n_scr_dirs=$(ls $job_dir/*/ -d -1 -tr | wc -l)
	n_total=$(($n_total + $n_scr_dirs))
done
echo $n_total dirs to copy back
n_done=0
for job_dir in $(cat all_job_scripts | cut -d/ -f1)
do
	echo $job_dir
	for scr_dir in $(ls $job_dir/*/ -d -1 -tr)
	do
		#find $scr_dir -name "*_lig_src*.sdf" -exec cp {} $job_dir \;
		#find $scr_dir -name "*_lig_add*.sdf" -exec cp {} $job_dir \;
		#find $scr_dir -name "*_lig_fit_add*.sdf" -exec cp {} $job_dir \;
		#find $scr_dir -name "*_lig_gen_fit_add*.sdf" -exec cp {} $job_dir \;
		find $scr_dir -name "*.gen_metrics" -exec cp {} $job_dir \;
		n_done=$(($n_done + 1))
		echo "["$n_done"/"$n_total"] "$scr_dir
	done
done
