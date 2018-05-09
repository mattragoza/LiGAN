import caffe
import numpy as np

eps = 1.0
testing = False


class ChannelEuclideanLossLayer(caffe.Layer):
    '''
    Compute a Euclidean loss for each channel and take
    their sum weighted by the channel amount in the label.
    '''
    def setup(self, bottom, top):
        # check number of inputs
        if len(bottom) != 2:
            raise Exception('Need two inputs')

    def reshape(self, bottom, top):

        # check input shapes
        shape0 = tuple(bottom[0].shape)
        shape1 = tuple(bottom[1].shape)
        if shape0 != shape1:
            raise Exception('Inputs must have the same shape ({} vs. {})'.format(shape0, shape1))

        # gradient is same shape as inputs
        self.diff = np.zeros(shape0, dtype=np.float32)

        # intermediate sum keeps channels separate
        self.non_chan_sse = np.zeros(shape0[1], dtype=np.float32)
        self.non_chan_sum = np.zeros(shape0[1], dtype=np.float32)
        self.non_chan_axes = (0,) + tuple(range(2, len(shape0)))
        self.chan_shape = tuple([1, shape0[1]] + [1 for i in range(2, len(shape0))])

        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):

        # get total squared error in each channel
        self.diff[...] = bottom[0].data - bottom[1].data
        self.non_chan_sse[...] = np.sum(self.diff**2, axis=self.non_chan_axes) / 2.
        if testing:
            print 'non_chan_sse =', self.non_chan_sse

        # get total channel density in label blob
        self.non_chan_sum[...] = np.sum(bottom[1].data, axis=self.non_chan_axes) + eps
        if testing:
            print 'non_chan_sum =', self.non_chan_sum

        # weighted sum across channels
        top[0].data[...] = np.sum(self.non_chan_sse / self.non_chan_sum)
        if testing:
            print 'loss =', top[0].data

    def backward(self, top, propagate_down, bottom):

        if propagate_down[0]:
            bottom[0].diff[...] = self.diff / np.reshape(self.non_chan_sum, self.chan_shape)

        if propagate_down[1]:
            pass
            # quotient rule, but should test this
            #bottom[1].diff[...] = (self.non_chan_sum * -self.diff - (self.non_chan_sse**2)/2) / self.non_chan_sum**2


if __name__ == '__main__':
    testing = True
    param = '''
    force_backward: true
    layer {
      name: "input0"
      type: "Input"
      top: "input0"
      input_param {
        shape {
          dim: 2
          dim: 2
          dim: 2
        }
      }
    }
    layer {
      name: "input1"
      type: "Input"
      top: "input1"
      input_param {
        shape {
          dim: 2
          dim: 2
          dim: 2
        }
      }
    }
    layer {
      name: "loss"
      type: "Python"
      top: "loss"
      bottom: "input0"
      bottom: "input1"
      python_param {
        module: "channel_euclidean_loss_layer"
        layer: "ChannelEuclideanLossLayer"
      }
    }
    '''
    open('asdf.model','w').write(param)
    net = caffe.Net('asdf.model', caffe.TEST)

    def clear_net():
        net.blobs['input0'].data[...] = 0
        net.blobs['input1'].data[...] = 0
        net.blobs['input0'].diff[...] = 0
        net.blobs['input1'].diff[...] = 0

    print 'test0'
    clear_net()
    net.forward()
    assert net.blobs['loss'].data == 0.
    net.backward()
    assert np.allclose(net.blobs['input0'].diff, [[[0.,0.], [0.,0.]], [[0.,0.], [0.,0.]]])

    print 'test1'
    clear_net()
    net.blobs['input0'].data[0,0,0] = 2.
    net.forward()
    assert net.blobs['loss'].data == (2.-0.)**2 / 2 / (0 + eps)
    net.backward()
    assert np.allclose(net.blobs['input0'].diff, [[[2.,0.], [0.,0.]], [[0.,0.], [0.,0.]]])

    print 'test2'
    clear_net()
    net.blobs['input1'].data[0,0,0] = 2.
    net.forward()
    assert net.blobs['loss'].data == (0.-2.)**2 / 2 / (2 + eps)
    net.backward()
    assert np.allclose(net.blobs['input0'].diff, [[[-2./3.,0.], [0.,0.]], [[0.,0.], [0.,0.]]])

    print 'test3'
    clear_net()
    net.blobs['input0'].data[0,0,0] = 4.
    net.forward()
    assert net.blobs['loss'].data == (4.-0.)**2 / 2 / (0 + eps)
    net.backward()
    assert np.allclose(net.blobs['input0'].diff, [[[4.,0.], [0.,0.]], [[0.,0.], [0.,0.]]])

    print 'test4'
    clear_net()
    net.blobs['input0'].data[0,0,0] = 2.
    net.blobs['input0'].data[0,0,1] = 4.
    net.forward()
    assert net.blobs['loss'].data == ((2.-0.)**2 + (4.-0.)**2) / 2 / (0 + eps)
    net.backward()
    assert np.allclose(net.blobs['input0'].diff, [[[2.,4.], [0.,0.]], [[0.,0.], [0.,0.]]])

    print 'test5'
    clear_net()
    net.blobs['input0'].data[0,0,0] = 2.
    net.blobs['input0'].data[0,1,0] = 4.
    net.forward()
    assert net.blobs['loss'].data == (2.-0.)**2 / 2 / (0+eps) + (4.-0.)**2 / 2 / (0+eps)
    net.backward()
    assert np.allclose(net.blobs['input0'].diff, [[[2.,0.], [4.,0.]], [[0.,0.], [0.,0.]]])

    print 'test6'
    clear_net()
    net.blobs['input1'].data[0,0,0] = 2.
    net.blobs['input1'].data[0,0,1] = 4.
    net.forward()
    assert net.blobs['loss'].data == ((0.-2.)**2 + (0.-4.)**2) / 2 / (2 + 4 + eps)
    net.backward()
    assert np.allclose(net.blobs['input0'].diff, [[[-2./7,-4./7], [0.,0.]], [[0.,0.], [0.,0.]]])

    print 'numeric gradient tests'
    h = 1e-6
    for n in range(100):
        for i in range(2):
            for j in range(2):
                for k in range(2):

                    clear_net()
                    net.blobs['input0'].data[...] = 0.01*np.abs(np.random.randn(2, 2, 2))
                    net.blobs['input1'].data[...] = 0.01*np.abs(np.random.randn(2, 2, 2))

                    net.forward()
                    y = net.blobs['loss'].data[0]

                    net.backward()
                    m = net.blobs['input0'].diff[i,j,k]

                    net.blobs['input0'].data[i,j,k] += h
                    net.forward()
                    yh = net.blobs['loss'].data[0]

                    mapprox = (yh - y) / h
                    assert np.abs(m - mapprox) < 0.01, (m, mapprox)

