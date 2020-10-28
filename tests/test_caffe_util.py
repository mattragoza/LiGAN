import sys, os, pytest
import numpy as np
from numpy import isclose
from numpy.linalg import norm
os.environ['GLOG_minloglevel'] = '1'

sys.path.insert(0, '..')
from caffe_util import CaffeNet, CaffeSubNet
from models import make_model
from train import MolGridData
from generate import MolGridGenerator


curr_dir = os.path.dirname(__file__)
rec_map = os.path.join(curr_dir, '../my_rec_map')
lig_map = os.path.join(curr_dir, '../my_lig_map')


class TestLig2Lig(object):

	encode_type = '_l-l'
	variational = False
	inputs = ['lig']
	losses = ['L2_loss']

	def make_param(self):
		'''
		Returns a simple NetParameter.
		'''
		return make_model(
			encode_type=self.encode_type,
			data_dim=12,
			rec_map=rec_map,
			lig_map=lig_map,
			n_latent=128,
			loss_types='e',
		)

	def init_net(self):
		'''
		Returns a simple CaffeNet, no scaffold.
		'''
		param = self.make_param()
		param.force_backward = True
		return CaffeNet(param, scaffold=False)

	def scaffold_net(self):
		'''
		Returns a simple CaffeNet, with scaffold.
		'''
		net = self.init_net()
		net.scaffold()
		return net

	def init_generator(self):
		'''
		Returns a MolGridGenerator, no forward pass.
		'''
		net = self.scaffold_net()
		return MolGridGenerator(net, forward=False)

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
		net = self.init_net()
		assert not net.has_scaffold()

	def test_net_scaffold(self):
		net = self.scaffold_net()
		assert net.has_scaffold()

	def test_generator_init(self):
		gen = self.init_generator()
		assert gen.variational == self.variational
		assert len(gen.encoders) == len(self.inputs)
		assert all(i in gen.encoders for i in self.inputs)
		assert gen.latent is not None
		assert gen.decoder is not None
		assert len(gen.losses) == len(self.losses)
		assert all(i in gen.losses for i in self.losses)

	def test_generator_forward_zero(self):
		gen = self.init_generator()
		gen.forward(**{i:0 for i in self.inputs})
		assert all(self.input_data_are_zero(gen))
		assert self.output_data_is_zero(gen)
		assert all(self.loss_data_are_zero(gen))

	def test_generator_forward_ones(self):
		gen = self.init_generator()
		gen.forward(**{i:1 for i in self.inputs})
		assert not any(self.input_data_are_zero(gen))
		assert not self.output_data_is_zero(gen)
		assert not any(self.loss_data_are_zero(gen))

	def test_generator_forward_zero_backward(self):
		gen = self.init_generator()
		gen.forward(**{i:0 for i in self.inputs})
		gen.backward()
		assert all(self.loss_data_are_zero(gen))
		assert all(self.loss_diff_are_one(gen))
		assert self.output_diff_is_zero(gen)
		assert all(self.input_diff_are_zero(gen))

	def test_generator_forward_ones_backward(self):
		gen = self.init_generator()
		gen.print_blob_diff_norms()
		gen.forward(**{i:1 for i in self.inputs})
		gen.backward()
		gen.print_blob_diff_norms()
		assert not any(self.loss_data_are_zero(gen))
		assert all(self.loss_diff_are_one(gen))
		assert not self.output_diff_is_zero(gen)
		assert not any(self.input_diff_are_zero(gen))



class TestRec2Lig(TestLig2Lig):
	encode_type = '_r-l'
	inputs = ['rec']
