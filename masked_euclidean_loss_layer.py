import caffe
import numpy as np


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
