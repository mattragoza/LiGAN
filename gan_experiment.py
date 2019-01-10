import sys, os, re, glob
import numpy as np

import torque_util


def get_cont_iter(dir_):
    cont_iter = 0
    states = glob.glob(os.path.join(dir_, '*.solverstate'))
    for state in states:
        m = re.match(dir_ + r'.*_iter_(\d+)\.solverstate', state)
        if m:
            iter_ = int(m.group(1))
            cont_iter = max(cont_iter, iter_)
    return cont_iter


if __name__ == '__main__':

    #_, params_file = sys.argv
    #params = [line.rstrip().split() for line in open(params_file)]

    data_name = 'lowrmsd'
    data_root = '/net/pulsar/home/koes/dkoes/PDBbind/refined-set/'
    max_iter = 20000
    seed = 0
    continue_ = False

    pbs_temps = [
        'adam0_10_10_ab_0.0.pbs'
        'adam0_10_10_b_0.0.pbs'
    ]

    data_model_file = 'data_24_0.5.model'

    gen_model_files = [
        'models/_rvl-le13_24_0.5_3_2l_32_2_1024_.model',
        'models/_rvl-le13_24_0.5_3_2l_32_2_1024_e.model',
        'models/_vl-le13_24_0.5_3_2l_32_2_1024_.model',
        'models/_vl-le13_24_0.5_3_2l_32_2_1024_e.model',
    ]

    disc_model_files = [
        'models/d11_24_3_1l_16_2_x.model',
    ]

    gan_names = []
    job_args = []
    for pbs_template in pbs_temps:
        for gen_model_file in gen_model_files:
            for disc_model_file in disc_model_files:
                for fold in [3]:

                    gan_type = os.path.splitext(os.path.basename(pbs_template))[0]

                    data_model_name = os.path.splitext(os.path.split(data_model_file)[1])[0]
                    gen_model_name = os.path.splitext(os.path.split(gen_model_file)[1])[0]
                    disc_model_name = os.path.splitext(os.path.split(disc_model_file)[1])[0]

                    seed, fold = int(seed), int(fold)
                    gen_warmup_name = gen_model_name.lstrip('_')
                    gan_name = '{}{}_{}'.format(gan_type, gen_model_name, disc_model_name)

                    if not os.path.isdir(gan_name):
                        os.makedirs(gan_name)

                    cont_iter = get_cont_iter(gan_name) if continue_ else 0

                    pbs_file = os.path.join(gan_name, pbs_template)

                    torque_util.write_pbs_file(pbs_file, pbs_template, gan_name,
                                               gan_name=gan_name,
                                               data_model_name=data_model_name,
                                               gen_model_name=gen_model_name,
                                               disc_model_name=disc_model_name,
                                               data_name=data_name,
                                               data_root=data_root,
                                               max_iter=max_iter,
                                               cont_iter=cont_iter,
                                               gen_warmup_name=gen_warmup_name)

                    gan_names.append(gan_name)
                    job_args.append((pbs_file, 4*seed + fold))

    with open('REPL', 'w') as f:
        f.write('\n'.join(gan_names))

    for a in job_args:
        torque_util.wait_for_free_gpus_and_submit_job(a, n_gpus_free=3, queue='dept_gpu_12GB', poll_every=1)

