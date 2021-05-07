wget http://bits.csb.pitt.edu/files/ligan_data/ \
	--recursive \
	--no-parent \
	--no-host-directories \
	--cut-dirs=2 \
	--reject="index.html*" \
	--directory-prefix=data 

#wget https://bits.csb.pitt.edu/files/molcaches/molportFULL_lig.molcache2 -P data
#wget https://bits.csb.pitt.edu/files/molcaches/molportFULL_rec.molcache2 -P data
#wget https://bits.csb.pitt.edu/files/molcaches/pubchem_all_diffs_lig.molcache2 -P data
#wget https://bits.csb.pitt.edu/files/molcaches/pubchem_all_diffs_rec.molcache2 -P data
