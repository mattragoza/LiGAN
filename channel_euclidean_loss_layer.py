import copy
import numpy as np
import scipy as sp
import caffe
caffe.set_mode_gpu()

import caffe_util

EPS = 1e-3


class ChannelEuclideanLossLayer(caffe.Layer):
    '''
    Compute a Euclidean loss for each channel and take their sum weighted in
    inverse proportion to the mean L2 norm of the channel in the current batch.
    '''
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


if __name__ == '__main__':

    net = caffe_util.Net.from_param(
        force_backward=True,
        layer=[
            dict(
                name='input0',
                type='Input',
                top=['input0'],
                input_param=dict(
                    shape=[dict(dim=[1, 10, 3, 3, 3])]
                )
            ),
            dict(
                name='input1',
                type='Input',
                top=['input1'],
                input_param=dict(
                    shape=[dict(dim=[1, 10, 3, 3, 3])]
                )
            ),
            dict(
                name='loss',
                type='Python',
                bottom=['input0', 'input1'],
                top=['loss'],
                python_param=dict(
                    module='channel_euclidean_loss_layer',
                    layer='ChannelEuclideanLossLayer'
                )
            )
        ],
        phase=caffe.TEST)

    def eval(x0, x1):
        net.blobs['input0'].data[...] = x0
        net.blobs['input1'].data[...] = x1
        net.forward()
        net.backward()
        return np.array(net.blobs['loss'].data), \
               np.array(net.blobs['input0'].diff)

    for i in range(1000):

        # evaluate f(x) and df/dx at random point x0
        x0 = np.random.randn(*net.blobs['input0'].shape)
        x1 = np.random.randn(*net.blobs['input1'].shape)
        f, g = eval(x0, x1)

        # approximate df/dx = (f(x+h) - f(x-h)) / 2h
        g_approx = np.zeros_like(x0)
        h = 1e-3
        for j in np.ndindex(*x0.shape):

            x0p = copy.deepcopy(x0)
            x0p[j] += h
            fp, _ = eval(x0p, x1)

            x0n = copy.deepcopy(x0)
            x0n[j] -= h
            fn, _ = eval(x0n, x1)

            g_approx[j] = (fp - fn) / (2*h)

        norm = lambda x: np.sum(np.abs(x))
        rel_err = norm(g - g_approx) / np.maximum(norm(g), norm(g_approx))
        print('rel_err = {}, log_10 = {}'.format(rel_err, np.log10(rel_err)))
