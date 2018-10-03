import sys, os, glob
import numpy as np

import torque_util


if __name__ == '__main__':

    #_, params_file = sys.argv
    #params = [line.rstrip().split() for line in open(params_file)]

    data_name = '1ai5'
    data_root = '/net/pulsar/home/koes/dkoes/PDBbind/refined-set/'
    max_iter = 50000
    cont_iter = 0
    seed = 0

    pbs_temps = [
        'adam2_2_2_g_0.01.pbs',
    ]

    gen_model_files = [
        'models/_l-le13_24_0.5_2_1l_8_1_4_e.model',
        'models/_l-le13_24_0.5_2_1l_8_1_8_e.model',
        'models/_l-le13_24_0.5_2_1l_8_1_16_e.model',
        'models/_r-le13_24_0.5_2_1l_8_1_4_e.model',
        'models/_r-le13_24_0.5_2_1l_8_1_8_e.model',
        'models/_r-le13_24_0.5_2_1l_8_1_16_e.model',
        'models/_l-le13_24_0.5_2_1l_8_1_4_.model',
        'models/_l-le13_24_0.5_2_1l_8_1_8_.model',
        'models/_l-le13_24_0.5_2_1l_8_1_16_.model',
        'models/_r-le13_24_0.5_2_1l_8_1_4_.model',
        'models/_r-le13_24_0.5_2_1l_8_1_8_.model',
        'models/_r-le13_24_0.5_2_1l_8_1_16_.model',
        'models/_vl-le13_24_0.5_2_1l_8_1_4_e.model',
        'models/_vl-le13_24_0.5_2_1l_8_1_8_e.model',
        'models/_vl-le13_24_0.5_2_1l_8_1_16_e.model',
        'models/_vr-le13_24_0.5_2_1l_8_1_4_e.model',
        'models/_vr-le13_24_0.5_2_1l_8_1_8_e.model',
        'models/_vr-le13_24_0.5_2_1l_8_1_16_e.model',
        'models/_vl-le13_24_0.5_2_1l_8_1_4_.model',
        'models/_vl-le13_24_0.5_2_1l_8_1_8_.model',
        'models/_vl-le13_24_0.5_2_1l_8_1_16_.model',
        'models/_vr-le13_24_0.5_2_1l_8_1_4_.model',
        'models/_vr-le13_24_0.5_2_1l_8_1_8_.model',
        'models/_vr-le13_24_0.5_2_1l_8_1_16_.model',
    ]

    disc_model_files = [
        'models/d11_24_2_1l_16_1_x.model',
    ]

    gan_names = []
    job_args = []
    for pbs_template in pbs_temps:
        for gen_model_file in gen_model_files:
            for disc_model_file in disc_model_files:
                for fold in [3]:
                    gan_type = os.path.splitext(os.path.basename(pbs_template))[0]
                    gen_model_name = os.path.splitext(os.path.split(gen_model_file)[1])[0]
                    resolution = 0.5 #gen_model_name.split('_')[3]
                    data_model_name = 'data_24_{}_cov'.format(resolution)
                    disc_model_name = os.path.splitext(os.path.split(disc_model_file)[1])[0]
                    seed, fold = int(seed), int(fold)
                    gen_warmup_name = gen_model_name.lstrip('_')
                    gan_name = '{}{}_{}'.format(gan_type, gen_model_name, disc_model_name)
                    if not os.path.isdir(gan_name):
                        os.makedirs(gan_name)
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

    with open('GEN_ADV_GRAD_NORM', 'w') as f:
        f.write('\n'.join(gan_names))

    map(torque_util.wait_for_free_gpus_and_submit_job, job_args)

