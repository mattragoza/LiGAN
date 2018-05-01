from __future__ import print_function
import sys
import pandas as pd


def parse_qstat(buf):
    all_job_data = []
    for job_buf in filter(len, buf.split('\n\n')):
        job_data = dict()
        for field_buf in filter(len, job_buf.split('\n    ')):
            name, value = field_buf.split(': ' if not job_data else ' = ', 1)
            job_data[name] = value.replace('\n\t', '')
        all_job_data.append(job_data)
    return pd.DataFrame(all_job_data).set_index('Job Id')


if __name__ == '__main__':
    _, in_file = sys.argv
    with open(in_file, 'r') as f:
        buf = f.read()
    df = parse_qstat(buf)

