import sys, os, gzip
from openbabel import pybel


def split_sdf(sdf_file):
	'''
	Split an sdf file into several with one
	pose per file, and write them gzipped.
	Don't overwrite existing file.
	'''
	out_name = os.path.splitext(sdf_file)[0]
	for i, mol in pybel.readfile('sdf', sdf_file):
		out_file = f'{out_name}_{i}.sdf.gz'
		if os.path.isfile(out_file):
			continue
		with gzip.open(out_file, 'wt') as f:
			f.write(mol.write('sdf'))


if __name__ == '__main__':
	_, data_file, data_root, n_labels, n_src_fields = argv

	for line in open(data_file):
		fields = line.split(' ')

		# iterate over fields and convert from .gninatypes
		#   to .sdf.gz, splitting sdf files as necessary
		for i in range(n_labels, n_labels + n_src_fields):

			src_no_ext, src_ext = os.path.splitext(fields[i])
			assert src_ext == '.gninatypes', fields[i]

			src_prefix, pose_idx = src_no_ext.rsplit('_', 1)
			pose_idx = int(pose_idx)

			is_lig_src = ((i - n_labels) % 2 == 1)
			if is_lig_src:
				new_src = f'{src_prefix}_{pose_idx}.sdf.gz'
			else:
				new_src = f'{src_prefix}.pdb'

			new_src_file = f'{data_root}/{new_src}'
			if is_lig_src and not os.path.isfile(new_src_file):
				split_sdf(f'{data_root}/{src_prefix}.sdf')

			assert os.path.isfile(new_src_file), new_src
			fields[i] = new_src

		line = ' '.join(fields)
		print(line, end='')
