import sys, os

import torque_util


if __name__ == '__main__':

    _, params_file = sys.argv
    params = [line.rstrip().split() for line in open(params_file)]

    data_name = 'lowrmsd' #'genlowrmsd'
    data_root = '/net/pulsar/home/koes/dkoes/PDBbind/refined-set/' #general-set-with-refined/'
    max_iter = 20000
    cont_iter = 0

    job_args = []
    for pbs_template, gen_model_file, disc_model_file, seed, fold in params:

        gan_type = os.path.splitext(os.path.basename(pbs_template))[0]
        gen_model_name = os.path.splitext(os.path.split(gen_model_file)[1])[0]
        resolution = gen_model_name.split('_')[3]
        data_model_name = 'data_24_{}'.format(resolution)
        disc_model_name = os.path.splitext(os.path.split(disc_model_file)[1])[0]
        seed, fold = int(seed), int(fold)
        solver_name = 'adam0'
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
                                   solver_name=solver_name,
                                   max_iter=max_iter,
                                   cont_iter=cont_iter,
                                   gen_warmup_name=gen_warmup_name)

        job_args.append((pbs_file, 4*seed + fold))

    map(torque_util.wait_for_free_gpus_and_submit_job, job_args)
