import sys, os, argparse

import job_queue


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Submit job scripts')
    parser.add_argument('job_script', nargs='+')
    parser.add_argument('--array', '-a')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    for job_script in args.job_script:
        queue = job_queue.get_job_queue(job_script)
        work_dir = os.path.dirname(job_script)
        job_id = queue.submit_job(job_script, work_dir=work_dir, array_idx=args.array)
        print(job_id)


if __name__ == '__main__':
    main(sys.argv[1:])
