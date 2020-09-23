import sys, os, re, time
import pandas as pd
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

import caffe_util


def check_gpu_ids(pid):

    with os.popen('nvidia-smi') as p:
        stdout = p.read()

    proc_buf = re.split(r'\n\s+\n', stdout)[1]
    proc_lines = proc_buf.split('\n')[4:-2]
    proc_data = [x.strip('| ').split() for x in proc_lines]
    proc_gpus = [int(row[0]) for row in proc_data]
    proc_pids = [int(row[1]) for row in proc_data]
    return [g for g, p in zip(proc_gpus, proc_pids) if p == pid]


def check_gpu_memory(gpu_id):

    with os.popen('nvidia-smi') as p:
        stdout = p.read()

    m = re.findall(r'(\d+)MiB\s*/\s*(\d+)MiB', stdout)
    return int(m[gpu_id][0]), int(m[gpu_id][1])


def time_function_call(func, *args, **kwargs):
    t_start = time.time()
    func(*args, **kwargs)
    return time.time() - t_start


def benchmark_net(net, n):

    df = pd.DataFrame(index=range(max(n, 1)))

    df['n_params'] = net.get_n_params()
    df['n_activs'] = net.get_n_activs()
    df['approx_size'] = net.get_approx_size()

    gpu_id = check_gpu_ids(os.getpid())[0]
    for i in range(n):
        df.loc[i, 't_forward']  = time_function_call(net.forward)
        df.loc[i, 't_backward'] = time_function_call(net.backward)
        df.loc[i, 't_fwd_bwd'] = df.loc[i, 't_forward'] + df.loc[i, 't_backward']
        gpu_mem_usage, gpu_mem_limit = check_gpu_memory(gpu_id)
        df.loc[i, 'gpu_mem_usage'] = gpu_mem_usage * 2**20
        df.loc[i, 'gpu_mem_limit'] = gpu_mem_limit * 2**20
        df.loc[i, 'gpu_mem_util'] = df.loc[i, 'gpu_mem_usage'] / df.loc[i, 'gpu_mem_limit']

    return df


if __name__ == '__main__':

    _, model_file, n = sys.argv

    net = caffe_util.Net(model_file, caffe.TEST)
    df = benchmark_net(net, n=int(n))

    print(df)
    print('MEAN')
    print(df.mean())
    print('MAX')
    print(df.max())

