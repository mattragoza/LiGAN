import copy
from ast import literal_eval
from functools import partial
import multiprocessing as mp
import numpy as np
import scipy as sp
import caffe
caffe.set_mode_gpu()

import caffe_util
import atom_types
import generate


class AtomFittingLayer(caffe.Layer):

    def setup(self, bottom, top):

        if len(bottom) != 1:
            raise Exception('AtomFittingLayer takes 1 bottom blob')

        params = literal_eval(self.param_str)
        self.resolution = params['resolution']
        self.channels = atom_types.get_default_lig_channels(params['use_covalent_radius'])
        types_file = params.get('gninatypes_file', None)
        if types_file:
            self.c = generate.read_gninatypes_file(types_file, self.channels)[1]
        else:
            self.c = None
        self.map = mp.Pool().map

    def reshape(self, bottom, top):

        top[0].reshape(*bottom[0].shape)

    def forward(self, bottom, top):

        f = partial(generate.fit_atoms_to_grid,
                    channels=self.channels,
                    center=np.zeros(3),
                    resolution=self.resolution,
                    max_iter=10, lr=0.01, mo=0.9,
                    fit_channels=self.c)

        top[0].data[...] = zip(*self.map(f, bottom[0].data))[3]

    def backward(self, top, propagate_down, bottom):
        pass


class ChannelEuclideanLossLayer(caffe.Layer):
    '''
    Compute a Euclidean loss for each channel and take their sum weighted in
    inverse proportion to the mean L2 norm of the channel in the current batch.
    '''
    EPS = 1e-3

    def setup(self, bottom, top):

        if len(bottom) != 2:
            raise Exception('Need two inputs')

    def reshape(self, bottom, top):

        input0_shape = tuple(bottom[0].shape)
        input1_shape = tuple(bottom[1].shape)
        if input0_shape != input1_shape:
            raise Exception('Inputs must have the same shape ({} vs. {})'
                            .format(input0_shape, input1_shape))

        self.diff = np.zeros(input0_shape, dtype=np.float32)

        n_channels = input0_shape[1]
        self.chan_sse = np.zeros(n_channels, dtype=np.float32)
        self.chan_norm = np.zeros(n_channels, dtype=np.float32)
        self.chan_weight = np.zeros(n_channels, dtype=np.float32)

        n_dims = len(input0_shape)
        self.non_chan_axes = tuple(i for i in range(n_dims) if i != 1)
        self.chan_shape = tuple(n_channels if i == 1 else 1 for i in range(n_dims))

        top[0].reshape(1)

    def forward(self, bottom, top):

        # get total squared error in each channel (batch mean)
        batch_size = bottom[0].shape[0]
        self.diff[...] = bottom[0].data - bottom[1].data
        self.chan_sse[...] = np.sum(self.diff**2, axis=self.non_chan_axes) / batch_size / 2.0

        # weights are inversely proportional to label channel squared L2 norms and have mean of 1.0
        self.chan_norm[...] = np.sum(bottom[1].data**2, axis=self.non_chan_axes) / batch_size + EPS
        self.chan_weight[...] = (1 / np.mean(1 / self.chan_norm)) / self.chan_norm

        # weighted sum across channels
        top[0].data[...] = np.sum(self.chan_sse * self.chan_weight)

    def backward(self, top, propagate_down, bottom):

        batch_size = bottom[0].shape[0]
        if propagate_down[0]:
            bottom[0].diff[...] = self.diff * np.reshape(self.chan_weight, self.chan_shape) / batch_size


class MaskedEuclideanLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception('Need two inputs')

    def reshape(self, bottom, top):
        shape0 = tuple(bottom[0].shape)
        shape1 = tuple(bottom[1].shape)
        if shape0 != shape1:
            raise Exception('Inputs must have the same shape ({} vs. {})'.format(shape0, shape1))
        self.diff = np.zeros(shape0, dtype=np.float64)
        i = shape0[2]//4
        self.mask = np.zeros(shape0, dtype=np.float64)
        self.mask[:,:,i:-i,i:-i,i:-i] = 1.0
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = (bottom[0].data - bottom[1].data) * self.mask
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.0

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if propagate_down[i]:
                bottom[i].diff[...] = (1, -1)[i] * self.diff / bottom[i].num
