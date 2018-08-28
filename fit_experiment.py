import sys, os

import torque_util


if __name__ == '__main__':

    _, params_file = sys.argv
    params = [line.rstrip('\n').split(' ', 1) for line in open(params_file)]

    data_name = 'lowrmsdtest0' #'genlowrmsd'
    data_root = '/net/pulsar/home/koes/dkoes/PDBbind/refined-set/' #general-set-with-refined/'

    job_args = []
    for out_prefix, fit_params in params:
        pbs_template = 'fit.pbs'
        model_name = 'rvl-le13_24_0.5_3_3_32_2_1024_e'
        weights_file = 'abgan_rvl-le13_24_0.5_3_3_32_2_1024_e_disc2/abgan_rvl-le13_24_0.5_3_3_32_2_1024_e_disc2.lowrmsd.0.0_gen_iter_20000.caffemodel'
        blob_name = 'lig_gen'
        fit_name = out_prefix
        if not os.path.isdir(fit_name):
            os.makedirs(fit_name)
        pbs_file = os.path.join(fit_name, pbs_template)
        torque_util.write_pbs_file(pbs_file, pbs_template, fit_name,
                                   model_name=model_name,
                                   weights_file=weights_file
                                   data_name=data_name,
                                   data_root=data_root,
                                   blob_name=blob_name,
                                   out_prefix=out_prefix,
                                   fit_params=fit_params)

        job_args.append((pbs_file, None))

    map(torque_util.submit_job, job_args)
