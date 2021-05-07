wget http://bits.csb.pitt.edu/files/ligan_weights/ \
	--recursive \
	--no-parent \
	--no-host-directories \
	--cut-dirs=2 \
	--reject="index.html*" \
	--directory-prefix=weights 
