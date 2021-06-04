cmd="python3 scripts/valid_mols.py"
old_prefix=data/it2_tt_0_lowrmsd_mols
new_prefix=data/it2_tt_0_lowrmsd_valid_mols
$cmd ${old_prefix}.types $CROSSDOCK_ROOT > ${new_prefix}.types
$cmd ${old_prefix}_train0.types $CROSSDOCK_ROOT > ${new_prefix}_train0.types
$cmd ${old_prefix}_train1.types $CROSSDOCK_ROOT > ${new_prefix}_train1.types
$cmd ${old_prefix}_train2.types $CROSSDOCK_ROOT > ${new_prefix}_train2.types
$cmd ${old_prefix}_test0.types $CROSSDOCK_ROOT > ${new_prefix}_test0.types
$cmd ${old_prefix}_test1.types $CROSSDOCK_ROOT > ${new_prefix}_test1.types
$cmd ${old_prefix}_test2.types $CROSSDOCK_ROOT > ${new_prefix}_test2.types
