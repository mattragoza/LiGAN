for lig_base in $(cat molportFULL_rand_test0.types | cut -d" " -f 4 | cut -d. -f 1)
do
	ls -U -1 $MOLPORT_ROOT/$lig_base.sdf
done
