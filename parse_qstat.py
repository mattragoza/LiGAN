from __future__ import print_function
import sys
import pandas as pd


def parse_qstat(buf):
    assert buf, 'nothing to parse'
    pbsnodes = not buf.startswith('Job Id')
    all_job_data = []
    job_delim = '\n\n'
    for job_buf in filter(len, buf.split(job_delim)):
        job_data = dict()
        field_delim = '\n' + (4, 5)[pbsnodes]*' '
        for field_buf in filter(len, job_buf.split(field_delim)):
            if not job_data:
                if pbsnodes:
                    name, value = 'Node Id', field_buf
                else:
                    name, value = field_buf.split(': ', 1)
            else:
                name, value = field_buf.split(' = ', 1)
            job_data[name] = value.replace('\n\t', '')
        all_job_data.append(job_data)
    return pd.DataFrame(all_job_data).set_index('Node Id' if pbsnodes else 'Job Id')


if __name__ == '__main__':
    _, in_file = sys.argv
    with open(in_file, 'r') as f:
        buf = f.read()
    df = parse_qstat(buf)

