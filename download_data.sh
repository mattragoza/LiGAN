wget https://bits.csb.pitt.edu/files/gan_data.tgz
tar zxvf gan_data.tgz
mv gan/data data
rmdir gan
wget https://bits.csb.pitt.edu/files/molcaches/molportFULL_lig.molcache2 -P data
wget https://bits.csb.pitt.edu/files/molcaches/molportFULL_rec.molcache2 -P data
wget https://bits.csb.pitt.edu/files/molcaches/pubchem_all_diffs_lig.molcache2 -P data
wget https://bits.csb.pitt.edu/files/molcaches/pubchem_all_diffs_rec.molcache2 -P data

