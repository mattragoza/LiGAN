import sys, os, re, shutil
from collections import defaultdict

_, in_dir = sys.argv

# cache calls to list dir contents
files_cache = dict()
def get_files(dir):
    if dir not in files_cache:
        files_cache[dir] = os.listdir(dir)
    return files_cache[dir]

met_pat = re.compile(r'.*_(\d+)\.gen_metrics')
err_pat = re.compile(r'slurm-(\d+)_(\d+)\.err')

# determine max array idx submitted yet
inf = float('inf')
min_idx = inf
max_idx = -1
job_ids = []
for in_file in get_files(in_dir):
    m = err_pat.match(in_file)
    if m:
        job_id = int(m.group(1))
        idx = int(m.group(2))
        if idx > max_idx:
            max_idx = idx
        if idx < min_idx:
            min_idx = idx
        job_ids.append(job_id)

n_submitted = max(0, max_idx - min_idx + 1)
print('array indices submitted: {}'.format(n_submitted))
if n_submitted > 0:
    print('{}-{}'.format(min_idx, max_idx))
else:
    exit()

query_idxs = set(range(min_idx, max_idx+1))
found_idxs = set()

# copy back metric files from latest job submission
do_copy = True
if do_copy:
    last_job_id = sorted(job_ids)[-1]
    for scr_file in get_files(in_dir + '/' + str(last_job_id)):
        m = met_pat.match(scr_file)
        if m:
            idx = int(m.group(1))
            found_idxs.add(idx)
            from_file = in_dir + '/' + str(last_job_id) + '/' + scr_file
            to_file =   in_dir + '/' + scr_file
            shutil.copyfile(from_file, to_file)

# try to find metric files for all idxs up to max idx
for in_file in get_files(in_dir):
    m = met_pat.match(in_file)
    if m:
        idx = int(m.group(1))
        found_idxs.add(idx)

# determine idxs that are missing metrics
error_idxs = query_idxs - found_idxs
n_missing = len(error_idxs)
print('array indices missing output: {}'.format(n_missing))
print(','.join(map(str, sorted(error_idxs))))

# find stderr files for idxs with missing metrics
err_files = defaultdict(list)
for in_file in get_files(in_dir):
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
    try:
        job_id, last_err_file = err_files[i][-1]
        last_err_file = os.path.join(in_dir, last_err_file)
        error = read_err_file(last_err_file)
        print(last_err_file + '\t' + str(error))
    except IndexError:
        print('no error file for job array_idx ' + str(i))

