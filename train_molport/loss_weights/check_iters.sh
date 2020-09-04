for i in $(ls -d gen_*);
do
	wi=$(ls -U -1 $i/*.caffemodel  | sed -n 's/.*_iter_\([0-9][0-9]*\).*/\1/p' | sort -n -r | head -n1);
	si=$(ls -U -1 $i/*.solverstate | sed -n 's/.*_iter_\([0-9][0-9]*\).*/\1/p' | sort -n -r | head -n1);
	oi=$(tail -n1 $i/*.training_output | awk '{print $1}');
	echo $i $wi $si $oi;
done
