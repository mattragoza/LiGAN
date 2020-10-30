import sys, os, pytest
import numpy as np
from numpy import isclose
from numpy.linalg import norm

os.environ['GLOG_minloglevel'] = '1'
import caffe

sys.path.insert(0, '..')
import caffe_util
import models


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


class TestModels(object):

	def test_encoder_decoder_init(self):
		m = models.EncoderDecoder()


class TestLig2LigAE(object):

	encode_type = '_l-l'
	variational = False
	inputs = ['lig']
	losses = ['L2_loss']

	########## CONSTRUCTORS ##########

	def get_params(self):

		return dict(
			model_type=self.model_type,
			data_dim=12,
			rec_map=rec_map,
			lig_map=lig_map,
			n_latent=128,
			loss_types='e',
		)

	def get_net_param(self):
		'''
		Returns a simple NetParameter.
		'''
		return make_model(**self.get_params())


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
