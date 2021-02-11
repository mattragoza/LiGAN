import sys, os
from collections import Counter

from atom_types import get_default_lig_channels
from generate import read_gninatypes_file

channels = get_default_lig_channels(False)
data_file, data_root = sys.argv[1:]

with open(data_file, 'r') as f:
    lines = f.readlines()

cnt = Counter()
for line in lines:
    fields = line.rstrip().split()
    lig_file = os.path.join(data_root, fields[3])
    xyz, c = read_gninatypes_file(lig_file, channels)
    cnt.update(c)

n = 0
for i, channel in enumerate(channels):
    print('{} {} {}'.format(i, channel.name, cnt[i]))
    n += cnt[i]

print('{} total'.format(n))

