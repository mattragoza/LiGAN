import sys, os, glob

import torque_util


if __name__ == '__main__':

    #_, params_file = sys.argv
    #params = [line.rstrip().split() for line in open(params_file)]

    data_name = 'two_atoms' #'lowrmsd' #'genlowrmsd'
    data_root = '/net/pulsar/home/koes/mtr22/gan/data/' #'/net/pulsar/home/koes/dkoes/PDBbind/refined-set/' #general-set-with-refined/'
    max_iter = 25000
    cont_iter = 0
    seed = 0

    pbs_temps = [
        'adam0_2_2__0.0.pbs',
        'adam0_2_2_g_0.0.pbs',
        'adam0_2_2_s_0.0.pbs',
        'adam0_2_2__0.01.pbs',
        'adam0_2_2_g_0.01.pbs',
        'adam0_2_2_s_0.01.pbs',
        'adam1_2_2__0.0.pbs',
        'adam1_2_2_g_0.0.pbs',
        'adam1_2_2_s_0.0.pbs',
        'adam1_2_2__0.01.pbs',
        'adam1_2_2_g_0.01.pbs',
        'adam1_2_2_s_0.01.pbs',
        'adam2_2_2__0.0.pbs',
        'adam2_2_2_g_0.0.pbs',
        'adam2_2_2_s_0.0.pbs',
        'adam2_2_2__0.01.pbs',
        'adam2_2_2_g_0.01.pbs',
        'adam2_2_2_s_0.01.pbs',
        'adam3_2_2__0.0.pbs',
        'adam3_2_2_g_0.0.pbs',
        'adam3_2_2_s_0.0.pbs',
        'adam3_2_2__0.01.pbs',
        'adam3_2_2_g_0.01.pbs',
        'adam3_2_2_s_0.01.pbs'
    ]

    pbs_temps = np.random.choice(pbs_temps, 10, replace=False)

    gan_names = []
    job_args = []
    for pbs_template in pbs_temps:
        for gen_model_file in ['models/_vr-le13_12_0.5_1_2l_8_1_8_', 'models/_vr-le13_12_0.5_1_2l_16_1_8_']:
            for disc_model_file in ['models/disc_12_1_1l_8_1_in', 'models/disc_12_1_1l_16_1_in']:
                for fold in [3]:
                    gan_type = os.path.splitext(os.path.basename(pbs_template))[0]
                    gen_model_name = os.path.splitext(os.path.split(gen_model_file)[1])[0]
                    resolution = 0.5 #gen_model_name.split('_')[3]
                    data_model_name = 'data_12_{}_cov_origin'.format(resolution)
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

    with open('GAN_NAMES', 'w') as f:
        f.write('\n'.join(gan_names))

    map(torque_util.wait_for_free_gpus_and_submit_job, job_args)

