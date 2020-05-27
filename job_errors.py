import sys, os, re
from collections import defaultdict

_, in_dir = sys.argv
in_files = os.listdir(in_dir)

met_pat = re.compile(r'.*_(\d+)\.gen_metrics')
err_pat = re.compile(r'slurm-(\d+)_(\d+)\.err')

# determine max array idx submitted yet
max_idx = -1
for in_file in in_files:
    m = err_pat.match(in_file)
    if m:
        idx = int(m.group(2))
        if idx > max_idx:
            max_idx = idx

# try to find metric files for all idxs up to max idx
found_idxs = set()
for in_file in in_files:
    m = met_pat.match(in_file)
    if m:
        idx = int(m.group(1))
        found_idxs.add(idx)

# determine idxs that are missing metrics
query_idxs = set(range(1, max_idx+1))
error_idxs = query_idxs - found_idxs
print(','.join(map(str, sorted(error_idxs))))

# find stderr files for idxs with missing metrics
err_files = defaultdict(list)
for in_file in in_files:
    m = err_pat.match(in_file)
    if m:
        job_id = int(m.group(1))
        idx = int(m.group(2))
        if idx in error_idxs:
            err_files[idx].append((job_id, in_file))


def read_err_file(err_file):
    wrn_pat = re.compile(r'Warning.*')
    err_pat = re.compile(r'.*(Error|Exception|error|fault|failed).*')
    error = None
    with open(err_file) as f:
        for line in f:
            if not wrn_pat.match(line) and err_pat.match(line):
                error = line.rstrip()
    return error


for i in sorted(error_idxs):
    _, last_err_file = err_files[i][-1]
    last_err_file = os.path.join(in_dir, last_err_file)
    error = read_err_file(last_err_file)
    print(last_err_file + '\t' + error)

