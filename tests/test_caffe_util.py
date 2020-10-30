import sys, os, pytest
import numpy as np
from numpy import isclose
from numpy.linalg import norm

os.environ['GLOG_minloglevel'] = '1'
import caffe

sys.path.insert(0, '..')
import caffe_util as cu


curr_dir = os.path.dirname(__file__)
rec_map = os.path.join(curr_dir, '../my_rec_map')
lig_map = os.path.join(curr_dir, '../my_lig_map')


def print_side_by_side(param1, param2):
	'''
	Print two protobuf messages side-by-side.
	'''
	lines1 = str(param1).split('\n')
	lines2 = str(param2).split('\n')
	n_lines_diff = len(lines1) - len(lines2)
	if n_lines_diff > 0:
		lines2 += [''] * n_lines_diff
	elif n_lines_diff < 0:
		lines1 += [''] * -n_lines_diff
	max_line_len = 0
	for line in lines1:
		if len(line) > max_line_len:
			max_line_len = len(line)
	for line1, line2 in zip(lines1, lines2):
		print(line1.ljust(max_line_len) + '|' + line2)


def get_caffe_nodes(n):
	return (cu.CaffeNode() for i in range(n))


def get_caffe_blobs(n):
	return (cu.CaffeBlob() for i in range(n))


def get_caffe_layers(n):
	return (cu.Split() for i in range(n))


class TestCaffeNode(object):

	def test_init(self):
		x = cu.CaffeNode()
		assert x.name == hex(id(x))
		assert x.bottoms == []
		assert x.tops == []
		assert x.net is None

	def test_init_name(self):
		x = cu.CaffeNode(name='asdf')
		assert x.name == 'asdf'

	def test_add_bottom(self):
		x, y = get_caffe_nodes(2)
		x.add_bottom(y)
		assert x.bottoms == [y]
		assert y.tops == []

	def test_add_top(self):
		x, y = get_caffe_nodes(2)
		x.add_top(y)
		assert x.tops == [y]
		assert y.bottoms == []

	def test_replace_bottom(self):
		x, y, z = get_caffe_nodes(3)
		x.add_bottom(y)
		x.replace_bottom(y, z)
		assert x.bottoms == [z]
		assert y.tops == []
		assert z.tops == []

	def test_replace_bottom_dup(self):
		x, y, z = get_caffe_nodes(3)
		x.add_bottom(y)
		x.add_bottom(y)
		x.replace_bottom(y, z)
		assert x.bottoms == [z, y]

	def test_replace_bottom_err(self):
		x, y, z = get_caffe_nodes(3)
		with pytest.raises(ValueError):
			x.replace_bottom(y, z)

	def test_replace_top(self):
		x, y, z = get_caffe_nodes(3)
		x.add_top(y)
		x.replace_top(y, z)
		assert x.tops == [z]
		assert y.bottoms == []
		assert z.bottoms == []

	def test_replace_top_dup(self):
		x, y, z = get_caffe_nodes(3)
		x.add_top(y)
		x.add_top(y)
		x.replace_top(y, z)
		assert x.tops == [z, y]

	def test_replace_top_err(self):
		x, y, z = get_caffe_nodes(3)
		with pytest.raises(ValueError):
			x.replace_top(y, z)


class TestCaffeBlob(object):

	def test_init(self):
		x = cu.CaffeBlob()
		assert x.name == hex(id(x))
		assert x.bottoms == []
		assert x.tops == []
		assert x.net is None

	def test_init_name(self):
		x = cu.CaffeBlob(name='asdf')
		assert x.name == 'asdf'

	def add_bottom(self):
		x = cu.CaffeBlob()
		f = cu.Split()
		x.add_bottom(f)
		assert x.bottoms == [f]
		assert f.tops == []

	def add_bottom_err(self):
		x, y = get_caffe_blobs(2)
		with pytest.raises(AssertionError):
			x.add_bottom(y)

	def test_add_top(self):
		x = cu.CaffeBlob()
		f = cu.Split()
		x.add_top(f)
		assert x.tops == [f]
		assert f.bottoms == []

	def test_add_top_err(self):
		x, y = get_caffe_blobs(2)
		with pytest.raises(AssertionError):
			x.add_top(y)


class TestCaffeLayer(object):

	def test_base_init(self):
		with pytest.raises(TypeError):
			f = cu.CaffeLayer()

	def test_subclass_init(self):
		for layer_name in caffe.layer_type_list():
			f = getattr(cu, layer_name)()
			assert f.n_tops == 1
			assert f.in_place == False
			assert f.loss_weight is None
			assert f.lr_mult is None
			assert f.decay_mult is None
			if layer_name in cu.param_type_map:
				param_type = cu.param_type_map[layer_name]
				assert isinstance(f.param, param_type)
			else:
				assert f.param is None

	def add_bottom(self):
		x = cu.CaffeBlob()
		f = cu.Split()
		f.add_bottom(x)
		assert f.bottoms == [x]
		assert x.tops == []

	def add_bottom_err(self):
		f, g = get_caffe_layers(2)
		with pytest.raises(AssertionError):
			f.add_bottom(g)

	def test_add_top(self):
		x = cu.CaffeBlob()
		f = cu.Split()
		f.add_top(x)
		assert f.tops == [x]
		assert x.bottoms == []

	def test_add_top_err(self):
		f, g = get_caffe_layers(2)
		with pytest.raises(AssertionError):
			f.add_top(g)

	def test_call(self):
		x = cu.CaffeBlob()
		f = cu.Split()
		y = f(x)
		assert isinstance(y, cu.CaffeBlob)
		assert y.bottoms == [f]
		assert f.bottoms == [x]
		assert x.tops == [f]
		assert f.tops == [y]

	def test_call_n_tops(self):
		x = cu.CaffeBlob()
		f = cu.Split(n_tops=2)
		y = f(x)
		assert len(y) == 2
		assert isinstance(y[0], cu.CaffeBlob)
		assert isinstance(y[1], cu.CaffeBlob)
		assert y[0] != y[1]
		assert y[0].bottoms == [f]
		assert y[1].bottoms == [f]
		assert f.tops == y

	def test_call_two_args(self):
		x, y = get_caffe_blobs(2)
		f = cu.Split()
		z = f(x, y)
		assert f.bottoms == [x, y]
		assert x.tops == [f]
		assert y.tops == [f]

	def test_call_tuple_arg(self):
		x, y = get_caffe_blobs(2)
		f = cu.Split()
		z = f((x, y))
		assert f.bottoms == [x, y]
		assert x.tops == [f]
		assert y.tops == [f]

	def test_call_in_place(self):
		x = cu.CaffeBlob()
		f = cu.Split(in_place=True)
		y = f(x)
		assert y is x

	def test_call_in_place_err(self):
		with pytest.raises(AssertionError):
			f = cu.Split(in_place=True, n_tops=2)

	def test_call_in_place_err2(self):
		x, y = get_caffe_blobs(2)
		f = cu.Split(in_place=True)
		with pytest.raises(AssertionError):
			f(x, y)



if False:
	def get_gen(self, forward):
		return Generator(self.get_net(scaffold=True), forward=forward)

	########## CHECKS ##########

	def loss_data_are_zero(self, gen): 
		for l in self.losses:
			loss_data = gen.net.blobs[l].data
			yield isclose(0, norm(loss_data))

	def loss_diff_are_one(self, gen):
		for l in self.losses:
			loss_diff = gen.net.blobs[l].diff
			yield isclose(1, norm(loss_diff))

	def input_data_are_zero(self, gen):
		for i in self.inputs:
			input_data = gen.net.blobs[i].data
			yield isclose(0, norm(input_data))

	def output_data_is_zero(self, gen):
		return isclose(0, norm(gen.net.blobs['lig_gen'].data))

	def input_diff_are_zero(self, gen):
		for i in self.inputs:
			input_diff = gen.net.blobs[i].diff
			yield isclose(0, norm(input_diff))

	def output_diff_is_zero(self, gen):
		return isclose(0, norm(gen.net.blobs['lig_gen'].diff))

	########## BEGIN TESTS ##########

	def test_net_init(self):
		param = self.get_param()
		net = CaffeNet(param, scaffold=False)
		assert len(net.layers_) == len(param.layer)
		assert set(net.layers_) == set(l.name for l in param.layer)
		got_param = net.to_param()
		print_side_by_side(param, got_param)
		assert got_param == param
		assert not net.has_scaffold()

	def test_net_scaffold(self):
		net = self.get_net(scaffold=True)
		assert net.has_scaffold()
		assert len(net.layers_) == len(net.layers)
		assert len(net.blobs_) == len(net.blobs)

	def test_generator_init(self):
		gen = self.get_gen(forward=False)
		assert gen.variational == self.variational
		assert len(gen.encoders) == len(self.inputs)
		assert all(i in gen.encoders for i in self.inputs)
		assert gen.latent is not None
		assert gen.decoder is not None
		assert len(gen.losses) == len(self.losses)
		assert all(i in gen.losses for i in self.losses)

	def test_generator_forward_zero(self):
		gen = self.get_gen(forward=False)
		gen.forward(**{i:0 for i in self.inputs})
		assert all(self.input_data_are_zero(gen))
		assert self.output_data_is_zero(gen)
		assert all(self.loss_data_are_zero(gen))

	def test_generator_forward_ones(self):
		gen = self.get_gen(forward=False)
		gen.forward(**{i:1 for i in self.inputs})
		assert not any(self.input_data_are_zero(gen))
		assert not self.output_data_is_zero(gen)
		assert not any(self.loss_data_are_zero(gen))

	def test_generator_forward_zero_backward(self):
		gen = self.get_gen(forward=False)
		gen.forward(**{i:0 for i in self.inputs})
		gen.backward()
		assert all(self.loss_data_are_zero(gen))
		assert all(self.loss_diff_are_one(gen))
		assert self.output_diff_is_zero(gen)
		assert all(self.input_diff_are_zero(gen))

	def test_generator_forward_ones_backward(self):
		gen = self.get_gen(forward=False)
		gen.net.draw(type(self).__name__ + '.png')
		gen.net.print_norms()
		gen.forward(**{i:1 for i in self.inputs})
		gen.backward()
		gen.net.print_norms()
		assert not any(self.loss_data_are_zero(gen))
		assert all(self.loss_diff_are_one(gen))
		assert not self.output_diff_is_zero(gen)
		assert not any(self.input_diff_are_zero(gen))



#class TestRec2Lig(TestLig2Lig):
#	encode_type = '_r-l'
#	inputs = ['rec']
