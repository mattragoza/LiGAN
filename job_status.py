import sys, os, argparse
import pandas as pd

import job_queue


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Check job status')
    parser.add_argument('job_script', nargs='+')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    
    all_status = []
    for job_script in args.job_script:
        job_name = os.path.basename(os.path.dirname(job_script).rstrip('\\/'))
        queue = job_queue.get_job_queue(job_script)
        status = queue.get_status([job_name])
        all_status.append(status)

    print(pd.concat(all_status))


if __name__ == '__main__':
    main(sys.argv[1:])
