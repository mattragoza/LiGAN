from __future__ import print_function, division
import sys, os, re, itertools, ast
from collections import defaultdict
import numpy as np
import scipy as sp

import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', 80)

import caffe
caffe.set_mode_gpu()

import caffe_util
import generate

if __name__ == '__main__':

    out_prefix = '1AO'
    data_name = 'two_atoms'
    iter_ = 25000
    n_samples = 50

    #model_file = 'models/vr-le13_12_0.5_1_2l_8_1_8_.model'
    #model_name = 'adam2_2_2_b_0.01_vr-le13_12_0.5_1_2l_8_1_8__d11_12_1_1l_16_1_x'
    model_file = 'models/vr-le13_12_0.5_2_1lg_8_2_16_f.model'
    model_name = 'adam2_2_2_b_0.01_vr-le13_12_0.5_2_1lg_8_2_16_f_d11_12_2_1l_8_1_x'
    weights_file = '{}/{}.{}.0.all_gen_iter_{}.caffemodel'.format(model_name, model_name, data_name, iter_)

    data_root = '/home/mtr22/dvorak' + '/net/pulsar/home/koes/mtr22/gan/data/' #dkoes/PDBbind/refined-set/'
    data_file = 'data/two_atoms.types'

    net_param = caffe_util.NetParameter.from_prototxt(model_file)
    net_param.set_molgrid_data_source(data_file, '')
    data_param = net_param.get_molgrid_data_param(caffe.TEST)
    data_param.random_rotation = True
    data_param.fix_center_to_origin = True
    resolution = data_param.resolution
    
    params = ast.literal_eval(net_param.layer[-2].python_param.param_str)
    params['gninatypes_file'] = '/home/mtr22/dvorak' + params['gninatypes_file']
    net_param.layer[-2].python_param.param_str = str(params)

    model_file = out_prefix + '.model'
    net_param.to_prototxt(model_file)

    argv = '-m {} -w {} -B rec -b lig_gen --max_iter 3 --fit_atom_types --verbose 1 --data_file {} --data_root {} -o {} --n_samples {} --random_rotation --fix_center_to_origin'  \
            .format(model_file, weights_file, data_file, data_root, out_prefix, n_samples).split()
    generate.main(argv)
