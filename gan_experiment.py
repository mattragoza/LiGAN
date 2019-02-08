import sys, os, re, glob
import numpy as np

from pbs_templates import fill_template
import torque_util


DATA_PREFIX = 'lowrmsd'
DATA_ROOT   = '/net/pulsar/home/koes/dkoes/PDBbind/refined-set/'

PBS_TEMPLATE_FILES = [
    'pbs_templates/adam0_2_2_b_0.0.pbs'
]

SEED = 0
FOLD = 3 # 3 for 'all'

CONTINUE = False
MAX_ITER = 50000

DATA_MODEL_FILES = [
    'models/data_24_0.5_batch25.model'
]

GEN_MODEL_FILES  = [
    'models/_vl-le13_24_0.5_3_2l_32_2_1024_e.model',
    'models/_vl-le13_24_0.5_3_2ld_32_2_1024_e.model',    
    'models/_vl-le13_24_0.5_3_2li_32_2_1024_e.model',
    'models/_vl-le13_24_0.5_3_2lid_32_2_1024_e.model',
    'models/_vl-le13_24_0.5_3_3l_32_2_1024_e.model',
    'models/_vl-le13_24_0.5_3_3ld_32_2_1024_e.model',
    'models/_vl-le13_24_0.5_3_3li_32_2_1024_e.model',
    'models/_vl-le13_24_0.5_3_3lid_32_2_1024_e.model',
    'models/_vr-le13_24_0.5_3_2l_32_2_1024_e.model',
    'models/_vr-le13_24_0.5_3_2ld_32_2_1024_e.model',
    'models/_vr-le13_24_0.5_3_2li_32_2_1024_e.model',
    'models/_vr-le13_24_0.5_3_2lid_32_2_1024_e.model',
    'models/_vr-le13_24_0.5_3_3l_32_2_1024_e.model',
    'models/_vr-le13_24_0.5_3_3ld_32_2_1024_e.model',
    'models/_vr-le13_24_0.5_3_3li_32_2_1024_e.model',
    'models/_vr-le13_24_0.5_3_3lid_32_2_1024_e.model',
]

DISC_MODEL_FILES = [
    'models/d11_24_3_1l_16_2_x.model',
]


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

    for pbs_template_file in PBS_TEMPLATE_FILES:

        with open(pbs_template_file, 'r') as f:
            pbs_template = f.read()

        for data_model_file in DATA_MODEL_FILES:
            for gen_model_file in GEN_MODEL_FILES:
                for disc_model_file in DISC_MODEL_FILES:

                    pbs_name = os.path.splitext(os.path.basename(pbs_template_file))[0]
                    data_model_name = os.path.splitext(os.path.basename(data_model_file))[0]
                    gen_model_name  = os.path.splitext(os.path.basename(gen_model_file))[0]
                    disc_model_name = os.path.splitext(os.path.basename(disc_model_file))[0]

                    gan_name = '{}{}_{}'.format(pbs_name, gen_model_name, disc_model_name)
                    if not os.path.isdir(gan_name):
                        os.makedirs(gan_name)

                    cont_iter = get_cont_iter(gan_name) if CONTINUE else 0

                    pbs_file = os.path.join(gan_name, pbs_name + '.pbs')
                    pbs_filled = fill_template(pbs_template,
                        gan_name=gan_name,
                        data_root=DATA_ROOT,
                        data_prefix=DATA_PREFIX,
                        max_iter=MAX_ITER,
                        cont_iter=cont_iter,
                        data_model_name=data_model_name,
                        gen_model_name=gen_model_name,
                        disc_model_name=disc_model_name,
                    )

                    with open(pbs_file, 'w') as f:
                        f.write(pbs_filled)
 
                    pbs_array_idx = 4*SEED + FOLD
                    torque_util.wait_for_free_gpus_and_submit_job((pbs_file, pbs_array_idx), 
                                                                  n_gpus_free=0, poll_every=2)
                    
