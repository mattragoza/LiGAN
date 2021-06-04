mol_src=$1
data_file=data/it2_tt_0_lowrmsd_mols.types
data_root=$CROSSDOCK_ROOT
mol_file=$data_root/$(grep $mol_src $data_file | awk '{print $5}')
ls $mol_file
