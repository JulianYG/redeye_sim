import os
import sys
from collections import OrderedDict
sys.path.append(os.path.join(os.path.dirname(os.getcwd()),'Tools'))
import addNoiseDeploy as adder
from runTest import new_run, goog_run
from math import log

net_type = "goog_net"

class protoPoint():

	def __init__(self, stddev_list, q_param, name, iters, depth=1,optional_trainedfile=None,optional_modelfile=None,optional_imagemean=None):
		self.depth = depth
		self.proto_dic = self._generate_proto(stddev_list, -q_param, q_param)
		if (net_type=="goog_net"):
			self.conv_param = OrderedDict([("conv1/7x7_s2", ([7, 7, 3], [112, 112, 64])), 
				("conv2/3x3_reduce", ([1, 1, 64], [56, 56, 64])), 
				("conv2/3x3", ([3, 3, 64], [56, 56, 192]))])
		else:
			self.conv_param = OrderedDict([("conv1", ([11, 11, 3], [55, 55, 96])), 
				("conv2", ([5, 5, 48], [27, 27, 256])), 
				("conv3", ([3, 3, 256], [13, 13, 382])),
				("conv4", ([3, 3, 192], [13, 13, 382])),
				("conv5", ([3, 3, 192], [13, 13, 256]))])
		self.name = name
		self.neighbors = []
		self.g_params = stddev_list
		self.q_params = [-q_param, q_param]
		self.score = -1
		self.iterNum = iters
		self.loss = float("inf")
		self.optional_trainedfile = optional_trainedfile
		self.optional_modelfile = optional_modelfile
		self.optional_imagemean = optional_imagemean

	def get_name(self):
		return self.name

	def get_score(self):
		if self.score == -1:
			 self.score = self.run_deploy()
		return self.score 

	def run_deploy(self):
		self.write_out()
		self.loss = self._energy_loss()
		if (net_type=="goog_net"):
			return goog_run(self.name, self.name, self.iterNum,self.optional_trainedfile,self.optional_modelfile,self.optional_imagemean)
		else:
			return new_run(self.name, self.name, self.iterNum,self.optional_trainedfile,self.optional_modelfile,self.optional_imagemean)
		#return new_run(self.name, self.name, self.iterNum)

	def write_out(self):
		if (net_type=="goog_net"):
			adder.build_new_proto('googletab', self.proto_dic, self.name,self.depth)
		else:
			adder.build_new_proto('alextab', self.proto_dic, self.name,self.depth)

	def get_neighbors(self):
		return self.neighbors
	
	def set_depth(self, depth):
		self.depth = depth;

	def set_neighbor(self, point):
		self.neighbors.append(point)

	def get_param(self):
		return self.g_params + self.q_params[-1:]

	def get_energy_loss(self):
		return self.loss

	def _energy_loss(self):
		loss = 1;
		return loss;
		# for p, conv in zip(self.get_param()[:-1], self.conv_param.values()):
		# 	unit_loss = 3.424e-15 / (p * p)
		# 	kernel_param, output_data_param = conv
		# 	kernel_h, kernel_w, kernel_d = kernel_param
		# 	output_h, output_w, output_d = output_data_param
		# 	loss += unit_loss * (kernel_h + 2) * kernel_w * kernel_w * kernel_d * output_h * output_w * output_d
		# bits = log(128 / self.get_param()[-1], 2)
		# loss += 2.56e-12 * 128 / self.get_param()[-1] + 4.97e-13 * bits
		# return loss

	def _generate_proto(self, stddev_list, q_min, q_max):
		"""
		Layers needed: data, pool1, conv2, pool2, incep_pool, incep_3, incep_5, incep_out.
		"""

		if (self.depth == 1):
			if (net_type=='goog_net'):
				return self._generate_proto1_goog(stddev_list,q_min,q_max);
			else:
				return self._generate_proto1_alex(stddev_list,q_min,q_max);

		if (self.depth == 2):
			if (net_type=='goog_net'):
				return self._generate_proto2_goog(stddev_list,q_min,q_max);
			else:
				return self._generate_proto2_alex(stddev_list,q_min,q_max);

		if (self.depth == -3):
			if (net_type=='goog_net'):
				return self._generate_proto_neg3_goog(stddev_list,q_min,q_max);
			else:
				print "Doesn't exist"
		
		if (self.depth == 3):
			if (net_type=='goog_net'):
				return self._generate_proto3_goog(stddev_list,q_min,q_max);
			else:
				return self._generate_proto3_alex(stddev_list,q_min,q_max);
		
		if (self.depth == 4):
			if (net_type=='goog_net'):
				return self._generate_proto4_goog(stddev_list,q_min,q_max);
			else:
				return self._generate_proto4_alex(stddev_list,q_min,q_max);

		if (self.depth == 5):
			if (net_type=='goog_net'):
				return self._generate_proto5_goog(stddev_list,q_min,q_max);
			else:
				return self._generate_proto5_alex(stddev_list,q_min,q_max);

	def _generate_proto1_alex(self, stddev_list, q_min, q_max):

			scale_list = [1]*3
			noise_param_list = 3*[0]
			for i in range(3):
				noise_param_list[i] = adder.build_g_param(std=stddev_list[i], scale=scale_list[i])
			noise_param_str_gn0 = noise_param_list[0]
			noise_param_str_gn1 = noise_param_list[1]
			noise_param_str_gn2 = noise_param_list[2]

			noise_param_q = adder.build_q_param(min_u=q_min, max_u=q_max)


			alex_data_n0_conv1 = OrderedDict([("lr_mult", str(0)),
								   ("decay_mult", str(0)),
								   ("ntype", "GAUSSIAN"),
								   ("pass", "false"),
								   ("noise_param", noise_param_str_gn0),
								   ("diff_scale", str(1.0))])
			alex_relu1_norm1 = OrderedDict([("lr_mult", str(0)),
								   ("decay_mult", str(0)),
								   ("ntype", "GAUSSIAN"),
								   ("pass", "false"),
								   ("noise_param", noise_param_str_gn1),
								   ("diff_scale", str(1.0))])
			alex_norm1_pool1 = OrderedDict([("lr_mult", str(0)),
								   ("decay_mult", str(0)),
								   ("ntype", "GAUSSIAN"),
								   ("pass", "false"),
								   ("noise_param", noise_param_str_gn2), 
								   ("diff_scale", str(1.0))])
			alex_pool1_conv2 = OrderedDict([("lr_mult", str(0)),
								   ("decay_mult", str(0)),
								   ("ntype", "UNIFORM"),
								   ("pass", "false"),
								   ("noise_param", noise_param_q), 
								   ("diff_scale", str(1.0))])

			layers = OrderedDict([("alex_data_n0_conv1", alex_data_n0_conv1),
				("alex_relu1_norm1", alex_relu1_norm1),
				("alex_norm1_pool1", alex_norm1_pool1),
				("alex_pool1_conv2", alex_pool1_conv2)])
			return layers


	def _generate_proto1_goog(self, stddev_list, q_min, q_max):

		scale_list = [1]*3
		noise_param_list = 3*[0]
		for i in range(3):
			noise_param_list[i] = adder.build_g_param(std=stddev_list[i], scale=scale_list[i])
		noise_param_str_gn0 = noise_param_list[0]
		noise_param_str_gn1 = noise_param_list[1]
		noise_param_str_gn2 = noise_param_list[2]

		noise_param_q = adder.build_q_param(min_u=q_min, max_u=q_max)


		goog_data_n0_conv1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn0),
							   ("diff_scale", str(1.0))])
		goog_conv1_relu7x7_pool1_3x3s2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn1),
							   ("diff_scale", str(1.0))])
		goog_pool1_norm1_qnoise = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn1),
							   ("diff_scale", str(1.0))])
		goog_gnoise_conv2_3x3_reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "UNIFORM"),
							   ("pass", "false"),
							   ("noise_param", noise_param_q), 
							   ("diff_scale", str(1.0))])
		
		layers = OrderedDict([("goog_data_n0_conv1", goog_data_n0_conv1),
			("goog_conv1_relu7x7_pool1_3x3s2", goog_conv1_relu7x7_pool1_3x3s2),
			("goog_pool1_norm1_qnoise", goog_pool1_norm1_qnoise),
			("goog_gnoise_conv2_3x3_reduce", goog_gnoise_conv2_3x3_reduce)])
		return layers
	
	def _generate_proto2_alex(self, stddev_list, q_min, q_max):
		"""
		Layers needed: data, pool1, conv2, pool2, incep_pool, incep_3, incep_5, incep_out.
		"""
		#self.depth
		scale_list = [1]*5
		noise_param_list = 5*[0]
		for i in range(5):
			noise_param_list[i] = adder.build_g_param(std=stddev_list[i], scale=scale_list[i])
		noise_param_str_gn0 = noise_param_list[0]
		noise_param_str_gn1 = noise_param_list[1]
		noise_param_str_gn2 = noise_param_list[2]
		noise_param_str_gn3 = noise_param_list[3]
		noise_param_str_gn4 = noise_param_list[4]

		noise_param_q = adder.build_q_param(min_u=q_min, max_u=q_max)


		alex_data_n0_conv1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn0),
							   ("diff_scale", str(1.0))])
		alex_relu1_norm1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn1),
							   ("diff_scale", str(1.0))])
		alex_norm1_pool1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn2),
							   ("diff_scale", str(1.0))])
		alex_conv2_norm2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn3),
							   ("diff_scale", str(1.0))])
		alex_norm2_pool2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn4),
							   ("diff_scale", str(1.0))])
		alex_pool2_conv3 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "UNIFORM"),
							   ("pass", "false"),
							   ("noise_param", noise_param_q), 
							   ("diff_scale", str(1.0))])
		
		layers = OrderedDict([("alex_data_n0_conv1", alex_data_n0_conv1),
			("alex_relu1_norm1", alex_relu1_norm1),
			("alex_norm1_pool1", alex_norm1_pool1),
			("alex_conv2_norm2", alex_conv2_norm2),
			("alex_norm2_pool2", alex_norm2_pool2),
			("alex_pool2_conv3", alex_pool2_conv3)])
		return layers

	def _generate_proto2_goog(self, stddev_list, q_min, q_max):
		"""
		Layers needed: data, pool1, conv2, pool2, incep_pool, incep_3, incep_5, incep_out.
		"""
		#self.depth
		scale_list = [1]*6
		noise_param_list = 6*[0]
		for i in range(6):
			noise_param_list[i] = adder.build_g_param(std=stddev_list[i], scale=scale_list[i])
		noise_param_str_gn0 = noise_param_list[0]
		noise_param_str_gn1 = noise_param_list[1]
		noise_param_str_gn2 = noise_param_list[2]
		noise_param_str_gn3 = noise_param_list[3]
		noise_param_str_gn4 = noise_param_list[4]
		noise_param_str_gn5 = noise_param_list[5]

		noise_param_q = adder.build_q_param(min_u=q_min, max_u=q_max)


		goog_data_n0_conv1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn0),
							   ("diff_scale", str(1.0))])
		goog_conv1_relu7x7_pool1_3x3s2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn1),
							   ("diff_scale", str(1.0))])
		goog_pool1_norm1_conv2_3x3reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn2),
							   ("diff_scale", str(1.0))])
		goog_conv2_relu3x3reduce_conv2_3x3 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn3),
							   ("diff_scale", str(1.0))])
		goog_conv2_relu3x3_conv2_norm2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn4),
							   ("diff_scale", str(1.0))])
		goog_conv2_norm2_pool2_3x3s2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn5),
							   ("diff_scale", str(1.0))])		
		goog_pool2_3x3s2_inception3a_1x1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "UNIFORM"),
							   ("pass", "false"),
							   ("noise_param", noise_param_q), 
							   ("diff_scale", str(1.0))])
		
		layers = OrderedDict([("goog_data_n0_conv1", goog_data_n0_conv1),
			("goog_conv1_relu7x7_pool1_3x3s2", goog_conv1_relu7x7_pool1_3x3s2),
			("goog_pool1_norm1_conv2_3x3reduce", goog_pool1_norm1_conv2_3x3reduce),
			("goog_conv2_relu3x3reduce_conv2_3x3", goog_conv2_relu3x3reduce_conv2_3x3),
			("goog_conv2_relu3x3_conv2_norm2", goog_conv2_relu3x3_conv2_norm2),
			("goog_conv2_norm2_pool2_3x3s2", goog_conv2_norm2_pool2_3x3s2),
			("goog_pool2_3x3s2_inception3a_1x1", goog_pool2_3x3s2_inception3a_1x1)])
		return layers

	def _generate_proto3_alex(self, stddev_list, q_min, q_max):

		#self.depth
		scale_list = [1]*6
		noise_param_list = 6*[0]
		for i in range(6):
			noise_param_list[i] = adder.build_g_param(std=stddev_list[i], scale=scale_list[i])
		noise_param_str_gn0 = noise_param_list[0]
		noise_param_str_gn1 = noise_param_list[1]
		noise_param_str_gn2 = noise_param_list[2]
		noise_param_str_gn3 = noise_param_list[3]
		noise_param_str_gn4 = noise_param_list[4]
		noise_param_str_gn5 = noise_param_list[5]

		noise_param_q = adder.build_q_param(min_u=q_min, max_u=q_max)

		alex_data_n0_conv1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn0),
							   ("diff_scale", str(1.0))])
		alex_relu1_norm1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn1),
							   ("diff_scale", str(1.0))])
		alex_norm1_pool1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn2),
							   ("diff_scale", str(1.0))])
		alex_conv2_norm2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn3),
							   ("diff_scale", str(1.0))])
		alex_norm2_pool2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn4),
							   ("diff_scale", str(1.0))])
		alex_conv3_qnoise = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn5),
							   ("diff_scale", str(1.0))])		
		alex_gnoise_conv4 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "UNIFORM"),
							   ("pass", "false"),
							   ("noise_param", noise_param_q), 
							   ("diff_scale", str(1.0))])
		
		layers = OrderedDict([("alex_data_n0_conv1", alex_data_n0_conv1),
			("alex_relu1_norm1", alex_relu1_norm1),
			("alex_norm1_pool1", alex_norm1_pool1),
			("alex_conv2_norm2", alex_conv2_norm2),
			("alex_norm2_pool2", alex_norm2_pool2),
			("alex_conv3_qnoise", alex_conv3_qnoise),
			("alex_gnoise_conv4", alex_gnoise_conv4)])
		return layers

	def _generate_proto_neg3_goog(self, stddev_list, q_min, q_max):
		"""
		Layers needed: data, pool1, conv2, pool2, incep_pool, incep_3, incep_5, incep_out.
		"""
		#self.depth
		scale_list = [1]*12
		noise_param_list = 12*[0]
		for i in range(12):
			noise_param_list[i] = adder.build_g_param(std=stddev_list[i], scale=scale_list[i])
		noise_param_str_gn0 = noise_param_list[0]
		noise_param_str_gn1 = noise_param_list[1]
		noise_param_str_gn2 = noise_param_list[2]
		noise_param_str_gn3 = noise_param_list[3]
		noise_param_str_gn4 = noise_param_list[4]
		noise_param_str_gn5 = noise_param_list[5]
		noise_param_str_gn6 = noise_param_list[6]
		noise_param_str_gn7 = noise_param_list[7]
		noise_param_str_gn8 = noise_param_list[8]
		noise_param_str_gn9 = noise_param_list[9]
		noise_param_str_gn10 = noise_param_list[10]
		noise_param_str_gn11 = noise_param_list[11]

		noise_param_q = adder.build_q_param(min_u=q_min, max_u=q_max)

		goog_data_n0_conv1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn0),
							   ("diff_scale", str(1.0))])
		goog_conv1_relu7x7_pool1_3x3s2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn1),
							   ("diff_scale", str(1.0))])
		goog_pool1_norm1_conv2_3x3reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn2),
							   ("diff_scale", str(1.0))])
		goog_conv2_relu3x3reduce_conv2_3x3 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn3),
							   ("diff_scale", str(1.0))])
		goog_conv2_relu3x3_conv2_norm2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn4),
							   ("diff_scale", str(1.0))])
		goog_conv2_norm2_pool2_3x3s2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn5),
							   ("diff_scale", str(1.0))])
		goog_inception3a_relu1x1_inception3a_3x3reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn6),
							   ("diff_scale", str(1.0))])
		goog_inception3a_3x3reduce_inception3a_3x3 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn7),
							   ("diff_scale", str(1.0))])
		goog_inception3a_3x3_inception3a_5x5reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn8),
							   ("diff_scale", str(1.0))])
		goog_inception3a_5x5reduce_inception3a_5x5 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn9),
							   ("diff_scale", str(1.0))])
		goog_inception3a_5x5_inception3a_pool = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn10),
							   ("diff_scale", str(1.0))])
		goog_inception3a_poolproj_inception3a_output = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn11),
							   ("diff_scale", str(1.0))])
		goog_inception3a_output3b_1x1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "UNIFORM"),
							   ("pass", "false"),
							   ("noise_param", noise_param_q), 
							   ("diff_scale", str(1.0))])
		
		layers = OrderedDict([("goog_data_n0_conv1", goog_data_n0_conv1),
			("goog_conv1_relu7x7_pool1_3x3s2", goog_conv1_relu7x7_pool1_3x3s2),
			("goog_pool1_norm1_conv2_3x3reduce", goog_pool1_norm1_conv2_3x3reduce),
			("goog_conv2_relu3x3reduce_conv2_3x3", goog_conv2_relu3x3reduce_conv2_3x3),
			("goog_conv2_relu3x3_conv2_norm2", goog_conv2_relu3x3_conv2_norm2),
			("goog_conv2_norm2_pool2_3x3s2", goog_conv2_norm2_pool2_3x3s2),
			("goog_inception3a_relu1x1_inception3a_3x3reduce", goog_inception3a_relu1x1_inception3a_3x3reduce),
			("goog_inception3a_3x3reduce_inception3a_3x3", goog_inception3a_3x3reduce_inception3a_3x3),
			("goog_inception3a_3x3_inception3a_5x5reduce", goog_inception3a_3x3_inception3a_5x5reduce),
			("goog_inception3a_5x5reduce_inception3a_5x5", goog_inception3a_5x5reduce_inception3a_5x5),
			("goog_inception3a_5x5_inception3a_pool", goog_inception3a_5x5_inception3a_pool),
			("goog_inception3a_poolproj_inception3a_output", goog_inception3a_poolproj_inception3a_output),
			("goog_inception3a_output3b_1x1", goog_inception3a_output3b_1x1)])
		return layers

	def _generate_proto3_goog(self, stddev_list, q_min, q_max):
		"""
		Layers needed: data, pool1, conv2, pool2, incep_pool, incep_3, incep_5, incep_out.
		"""
		#self.depth
		scale_list = [1]*18
		noise_param_list = 18*[0]
		for i in range(18):
			noise_param_list[i] = adder.build_g_param(std=stddev_list[i], scale=scale_list[i])
		noise_param_str_gn0 = noise_param_list[0]
		noise_param_str_gn1 = noise_param_list[1]
		noise_param_str_gn2 = noise_param_list[2]
		noise_param_str_gn3 = noise_param_list[3]
		noise_param_str_gn4 = noise_param_list[4]
		noise_param_str_gn5 = noise_param_list[5]
		noise_param_str_gn6 = noise_param_list[6]
		noise_param_str_gn7 = noise_param_list[7]
		noise_param_str_gn8 = noise_param_list[8]
		noise_param_str_gn9 = noise_param_list[9]
		noise_param_str_gn10 = noise_param_list[10]
		noise_param_str_gn11 = noise_param_list[11]
		noise_param_str_gn12 = noise_param_list[12]
		noise_param_str_gn13 = noise_param_list[13]
		noise_param_str_gn14 = noise_param_list[14]
		noise_param_str_gn15 = noise_param_list[15]
		noise_param_str_gn16 = noise_param_list[16]
		noise_param_str_gn17 = noise_param_list[17]

		noise_param_q = adder.build_q_param(min_u=q_min, max_u=q_max)

		goog_data_n0_conv1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn0),
							   ("diff_scale", str(1.0))])
		goog_conv1_relu7x7_pool1_3x3s2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn1),
							   ("diff_scale", str(1.0))])
		goog_pool1_norm1_conv2_3x3reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn2),
							   ("diff_scale", str(1.0))])
		goog_conv2_3x3reduce_conv2_3x3 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn3),
							   ("diff_scale", str(1.0))])
		goog_conv2_relu3x3_conv2_norm2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn4),
							   ("diff_scale", str(1.0))])
		goog_conv2_norm2_pool2_3x3s2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn5),
							   ("diff_scale", str(1.0))])
		goog_inception3a_relu1x1_inception3a_3x3reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn6),
							   ("diff_scale", str(1.0))])
		goog_inception3a_3x3reduce_inception3a_3x3 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn7),
							   ("diff_scale", str(1.0))])
		goog_inception3a_3x3_inception3a_5x5reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn8),
							   ("diff_scale", str(1.0))])
		goog_inception3a_5x5reduce_inception3a_5x5 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn9),
							   ("diff_scale", str(1.0))])
		goog_inception3a_5x5_inception3a_pool = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn10),
							   ("diff_scale", str(1.0))])
		goog_inception3a_poolproj_inception3a_output = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn11),
							   ("diff_scale", str(1.0))])
		goog_inception3b_1x1_inception3b_3x3reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn12),
							   ("diff_scale", str(1.0))])
		goog_inception3b_3x3reduce_inception3b_3x3 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn13),
							   ("diff_scale", str(1.0))])
		goog_inception3b_3x3_inception3b_5x5reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn14),
							   ("diff_scale", str(1.0))])
		goog_inception3b_5x5reduce_inception3b_5x5 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn15),
							   ("diff_scale", str(1.0))])
		goog_inception3b_5x5_inception3b_pool = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn16),
							   ("diff_scale", str(1.0))])
		goog_inception3b_poolproj_inception3b_output = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn17),
							   ("diff_scale", str(1.0))])
		goog_inception3b_output_inception4a_1x1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "UNIFORM"),
							   ("pass", "false"),
							   ("noise_param", noise_param_q), 
							   ("diff_scale", str(1.0))])
		
		layers = OrderedDict([("goog_data_n0_conv1", goog_data_n0_conv1),
			("goog_conv1_relu7x7_pool1_3x3s2", goog_conv1_relu7x7_pool1_3x3s2),
			("goog_pool1_norm1_conv2_3x3reduce", goog_pool1_norm1_conv2_3x3reduce),
			("goog_conv2_3x3reduce_conv2_3x3", goog_conv2_3x3reduce_conv2_3x3),
			("goog_conv2_relu3x3_conv2_norm2", goog_conv2_relu3x3_conv2_norm2),
			("goog_conv2_norm2_pool2_3x3s2", goog_conv2_norm2_pool2_3x3s2),
			("goog_inception3a_relu1x1_inception3a_3x3reduce", goog_inception3a_relu1x1_inception3a_3x3reduce),
			("goog_inception3a_3x3reduce_inception3a_3x3", goog_inception3a_3x3reduce_inception3a_3x3),
			("goog_inception3a_3x3_inception3a_5x5reduce", goog_inception3a_3x3_inception3a_5x5reduce),
			("goog_inception3a_5x5reduce_inception3a_5x5", goog_inception3a_5x5reduce_inception3a_5x5),
			("goog_inception3a_5x5_inception3a_pool", goog_inception3a_5x5_inception3a_pool),
			("goog_inception3a_poolproj_inception3a_output", goog_inception3a_poolproj_inception3a_output),
			("goog_inception3b_1x1_inception3b_3x3reduce", goog_inception3b_1x1_inception3b_3x3reduce),
			("goog_inception3b_3x3reduce_inception3b_3x3", goog_inception3b_3x3reduce_inception3b_3x3),
			("goog_inception3b_3x3_inception3b_5x5reduce", goog_inception3b_3x3_inception3b_5x5reduce),
			("goog_inception3b_5x5reduce_inception3b_5x5", goog_inception3b_5x5reduce_inception3b_5x5),
			("goog_inception3b_5x5_inception3b_pool", goog_inception3b_5x5_inception3b_pool),
			("goog_inception3b_poolproj_inception3b_output", goog_inception3b_poolproj_inception3b_output),
			("goog_inception3b_output_inception4a_1x1", goog_inception3b_output_inception4a_1x1)])
		return layers

	def _generate_proto4_alex(self, stddev_list, q_min, q_max):

		#self.depth
		scale_list = [1]*7
		noise_param_list = 7*[0]
		for i in range(7):
			noise_param_list[i] = adder.build_g_param(std=stddev_list[i], scale=scale_list[i])
		noise_param_str_gn0 = noise_param_list[0]
		noise_param_str_gn1 = noise_param_list[1]
		noise_param_str_gn2 = noise_param_list[2]
		noise_param_str_gn3 = noise_param_list[3]
		noise_param_str_gn4 = noise_param_list[4]
		noise_param_str_gn5 = noise_param_list[5]
		noise_param_str_gn6 = noise_param_list[6]

		noise_param_q = adder.build_q_param(min_u=q_min, max_u=q_max)


		alex_data_n0_conv1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn0),
							   ("diff_scale", str(1.0))])
		alex_relu1_norm1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn1),
							   ("diff_scale", str(1.0))])
		alex_norm1_pool1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn2),
							   ("diff_scale", str(1.0))])
		alex_conv2_norm2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn3),
							   ("diff_scale", str(1.0))])
		alex_norm2_pool2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn4),
							   ("diff_scale", str(1.0))])
		alex_conv3_conv4 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn5),
							   ("diff_scale", str(1.0))])
		alex_conv4_qnoise = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn6),
							   ("diff_scale", str(1.0))])
		alex_gnoise_conv5 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "UNIFORM"),
							   ("pass", "false"),
							   ("noise_param", noise_param_q), 
							   ("diff_scale", str(1.0))])

		layers = OrderedDict([("alex_data_n0_conv1", alex_data_n0_conv1),
			("alex_relu1_norm1", alex_relu1_norm1),
			("alex_norm1_pool1", alex_norm1_pool1),
			("alex_conv2_norm2", alex_conv2_norm2),
			("alex_norm2_pool2", alex_norm2_pool2),
			("alex_conv3_conv4", alex_conv3_conv4),
			("alex_conv4_qnoise", alex_conv4_qnoise),
			("alex_gnoise_conv5", alex_gnoise_conv5)])
		
		return layers

	def _generate_proto5_alex(self, stddev_list, q_min, q_max):

		#self.depth
		scale_list = [1]*8
		noise_param_list = 8*[0]
		for i in range(8):
			noise_param_list[i] = adder.build_g_param(std=stddev_list[i], scale=scale_list[i])
		noise_param_str_gn0 = noise_param_list[0]
		noise_param_str_gn1 = noise_param_list[1]
		noise_param_str_gn2 = noise_param_list[2]
		noise_param_str_gn3 = noise_param_list[3]
		noise_param_str_gn4 = noise_param_list[4]
		noise_param_str_gn5 = noise_param_list[5]
		noise_param_str_gn6 = noise_param_list[6]
		noise_param_str_gn7 = noise_param_list[7]

		noise_param_q = adder.build_q_param(min_u=q_min, max_u=q_max)

		alex_data_n0_conv1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn0),
							   ("diff_scale", str(1.0))])
		alex_relu1_norm1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn1),
							   ("diff_scale", str(1.0))])
		alex_norm1_pool1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn2),
							   ("diff_scale", str(1.0))])
		alex_conv2_norm2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn3),
							   ("diff_scale", str(1.0))])
		alex_norm2_pool2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn4),
							   ("diff_scale", str(1.0))])
		alex_conv3_conv4 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn5),
							   ("diff_scale", str(1.0))])
		alex_conv4_conv5 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn6),
							   ("diff_scale", str(1.0))])
		alex_conv5_pool5 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn7),
							   ("diff_scale", str(1.0))])
		alex_pool5_fc6 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "UNIFORM"),
							   ("pass", "false"),
							   ("noise_param", noise_param_q), 
							   ("diff_scale", str(1.0))])
		
		layers = OrderedDict([("alex_data_n0_conv1", alex_data_n0_conv1),
			("alex_relu1_norm1", alex_relu1_norm1),
			("alex_norm1_pool1", alex_norm1_pool1),
			("alex_conv2_norm2", alex_conv2_norm2),
			("alex_norm2_pool2", alex_norm2_pool2),
			("alex_conv3_conv4", alex_conv3_conv4),
			("alex_conv4_conv5", alex_conv4_conv5),
			("alex_conv5_pool5", alex_conv5_pool5),
			("alex_pool5_fc6", alex_pool5_fc6)])
		return layers


	def _generate_proto4_goog(self, stddev_list, q_min, q_max):

		#self.depth
		scale_list = [1]*24
		noise_param_list = 24*[0]
		for i in range(24):
			noise_param_list[i] = adder.build_g_param(std=stddev_list[i], scale=scale_list[i])
		noise_param_str_gn0 = noise_param_list[0]
		noise_param_str_gn1 = noise_param_list[1]
		noise_param_str_gn2 = noise_param_list[2]
		noise_param_str_gn3 = noise_param_list[3]
		noise_param_str_gn4 = noise_param_list[4]
		noise_param_str_gn5 = noise_param_list[5]
		noise_param_str_gn6 = noise_param_list[6]
		noise_param_str_gn7 = noise_param_list[7]
		noise_param_str_gn8 = noise_param_list[8]
		noise_param_str_gn9 = noise_param_list[9]
		noise_param_str_gn10 = noise_param_list[10]
		noise_param_str_gn11 = noise_param_list[11]
		noise_param_str_gn12 = noise_param_list[12]
		noise_param_str_gn13 = noise_param_list[13]
		noise_param_str_gn14 = noise_param_list[14]
		noise_param_str_gn15 = noise_param_list[15]
		noise_param_str_gn16 = noise_param_list[16]
		noise_param_str_gn17 = noise_param_list[17]
		noise_param_str_gn18 = noise_param_list[18]
		noise_param_str_gn19 = noise_param_list[19]
		noise_param_str_gn20 = noise_param_list[20]
		noise_param_str_gn21 = noise_param_list[21]
		noise_param_str_gn22 = noise_param_list[22]
		noise_param_str_gn23 = noise_param_list[23]		

		noise_param_q = adder.build_q_param(min_u=q_min, max_u=q_max)

		goog_data_n0_conv1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn0),
							   ("diff_scale", str(1.0))])
		goog_conv1_relu7x7_pool1_3x3s2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn1),
							   ("diff_scale", str(1.0))])
		goog_pool1_norm1_conv2_3x3reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn2),
							   ("diff_scale", str(1.0))])
		goog_conv2_3x3reduce_conv2_3x3 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn3),
							   ("diff_scale", str(1.0))])
		goog_conv2_relu3x3_conv2_norm2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn4),
							   ("diff_scale", str(1.0))])
		goog_conv2_norm2_pool2_3x3s2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn5),
							   ("diff_scale", str(1.0))])
		goog_inception3a_relu1x1_inception3a_3x3reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn6),
							   ("diff_scale", str(1.0))])
		goog_inception3a_3x3reduce_inception3a_3x3 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn7),
							   ("diff_scale", str(1.0))])
		goog_inception3a_3x3_inception3a_5x5reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn8),
							   ("diff_scale", str(1.0))])
		goog_inception3a_5x5reduce_inception3a_5x5 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn9),
							   ("diff_scale", str(1.0))])
		goog_inception3a_5x5_inception3a_pool = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn10),
							   ("diff_scale", str(1.0))])
		goog_inception3a_poolproj_inception3a_output = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn11),
							   ("diff_scale", str(1.0))])
		goog_inception3b_1x1_inception3b_3x3reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn12),
							   ("diff_scale", str(1.0))])
		goog_inception3b_3x3reduce_inception3b_3x3 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn13),
							   ("diff_scale", str(1.0))])
		goog_inception3b_3x3_inception3b_5x5reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn14),
							   ("diff_scale", str(1.0))])
		goog_inception3b_5x5reduce_inception3b_5x5 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn15),
							   ("diff_scale", str(1.0))])
		goog_inception3b_5x5_inception3b_pool = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn16),
							   ("diff_scale", str(1.0))])
		goog_inception3b_poolproj_inception3b_output = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn17),
							   ("diff_scale", str(1.0))])





		goog_inception4a_1x1_inception4a_3x3reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn18),
							   ("diff_scale", str(1.0))])
		goog_inception4a_3x3reduce_inception4a_3x3 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn19),
							   ("diff_scale", str(1.0))])
		goog_inception4a_3x3reduce_inception4a_5x5reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn20),
							   ("diff_scale", str(1.0))])
		goog_inception4a_5x5reduce_inception41_5x5 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn21),
							   ("diff_scale", str(1.0))])
		goog_inception4a_5x5_inception4a_pool = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn22),
							   ("diff_scale", str(1.0))])
		goog_inception4a_poolproj_inception4a_output = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn23),
							   ("diff_scale", str(1.0))])


		goog_inception4a_output_inception4b_1x1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "UNIFORM"),
							   ("pass", "false"),
							   ("noise_param", noise_param_q), 
							   ("diff_scale", str(1.0))])
		
		layers = OrderedDict([("goog_data_n0_conv1", goog_data_n0_conv1),
			("goog_conv1_relu7x7_pool1_3x3s2", goog_conv1_relu7x7_pool1_3x3s2),
			("goog_pool1_norm1_conv2_3x3reduce", goog_pool1_norm1_conv2_3x3reduce),
			("goog_conv2_3x3reduce_conv2_3x3", goog_conv2_3x3reduce_conv2_3x3),
			("goog_conv2_relu3x3_conv2_norm2", goog_conv2_relu3x3_conv2_norm2),
			("goog_conv2_norm2_pool2_3x3s2", goog_conv2_norm2_pool2_3x3s2),
			("goog_inception3a_relu1x1_inception3a_3x3reduce", goog_inception3a_relu1x1_inception3a_3x3reduce),
			("goog_inception3a_3x3reduce_inception3a_3x3", goog_inception3a_3x3reduce_inception3a_3x3),
			("goog_inception3a_3x3_inception3a_5x5reduce", goog_inception3a_3x3_inception3a_5x5reduce),
			("goog_inception3a_5x5reduce_inception3a_5x5", goog_inception3a_5x5reduce_inception3a_5x5),
			("goog_inception3a_5x5_inception3a_pool", goog_inception3a_5x5_inception3a_pool),
			("goog_inception3a_poolproj_inception3a_output", goog_inception3a_poolproj_inception3a_output),
			("goog_inception3b_1x1_inception3b_3x3reduce", goog_inception3b_1x1_inception3b_3x3reduce),
			("goog_inception3b_3x3reduce_inception3b_3x3", goog_inception3b_3x3reduce_inception3b_3x3),
			("goog_inception3b_3x3_inception3b_5x5reduce", goog_inception3b_3x3_inception3b_5x5reduce),
			("goog_inception3b_5x5reduce_inception3b_5x5", goog_inception3b_5x5reduce_inception3b_5x5),
			("goog_inception3b_5x5_inception3b_pool", goog_inception3b_5x5_inception3b_pool),
			("goog_inception3b_poolproj_inception3b_output", goog_inception3b_poolproj_inception3b_output),

			("goog_inception4a_1x1_inception4a_3x3reduce", goog_inception4a_1x1_inception4a_3x3reduce),
			("goog_inception4a_3x3reduce_inception4a_3x3", goog_inception4a_3x3reduce_inception4a_3x3),
			("goog_inception4a_3x3reduce_inception4a_5x5reduce", goog_inception4a_3x3reduce_inception4a_5x5reduce),
			("goog_inception4a_5x5reduce_inception41_5x5", goog_inception4a_5x5reduce_inception41_5x5),
			("goog_inception4a_5x5_inception4a_pool", goog_inception4a_5x5_inception4a_pool),
			("goog_inception4a_poolproj_inception4a_output", goog_inception4a_poolproj_inception4a_output),

			("goog_inception4a_output_inception4b_1x1", goog_inception4a_output_inception4b_1x1)])
		return layers

	def _generate_proto5_goog(self, stddev_list, q_min, q_max):

		#self.depth
		scale_list = [1]*30
		noise_param_list = 30*[0]
		for i in range(30):
			noise_param_list[i] = adder.build_g_param(std=stddev_list[i], scale=scale_list[i])
		noise_param_str_gn0 = noise_param_list[0]
		noise_param_str_gn1 = noise_param_list[1]
		noise_param_str_gn2 = noise_param_list[2]
		noise_param_str_gn3 = noise_param_list[3]
		noise_param_str_gn4 = noise_param_list[4]
		noise_param_str_gn5 = noise_param_list[5]
		noise_param_str_gn6 = noise_param_list[6]
		noise_param_str_gn7 = noise_param_list[7]
		noise_param_str_gn8 = noise_param_list[8]
		noise_param_str_gn9 = noise_param_list[9]
		noise_param_str_gn10 = noise_param_list[10]
		noise_param_str_gn11 = noise_param_list[11]
		noise_param_str_gn12 = noise_param_list[12]
		noise_param_str_gn13 = noise_param_list[13]
		noise_param_str_gn14 = noise_param_list[14]
		noise_param_str_gn15 = noise_param_list[15]
		noise_param_str_gn16 = noise_param_list[16]
		noise_param_str_gn17 = noise_param_list[17]
		noise_param_str_gn18 = noise_param_list[18]
		noise_param_str_gn19 = noise_param_list[19]
		noise_param_str_gn20 = noise_param_list[20]
		noise_param_str_gn21 = noise_param_list[21]
		noise_param_str_gn22 = noise_param_list[22]
		noise_param_str_gn23 = noise_param_list[23]
		noise_param_str_gn24 = noise_param_list[24]
		noise_param_str_gn25 = noise_param_list[25]
		noise_param_str_gn26 = noise_param_list[26]
		noise_param_str_gn27 = noise_param_list[27]
		noise_param_str_gn28 = noise_param_list[28]
		noise_param_str_gn29 = noise_param_list[29]

		noise_param_q = adder.build_q_param(min_u=q_min, max_u=q_max)

		goog_data_n0_conv1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn0),
							   ("diff_scale", str(1.0))])
		goog_conv1_relu7x7_pool1_3x3s2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn1),
							   ("diff_scale", str(1.0))])
		goog_pool1_norm1_conv2_3x3reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn2),
							   ("diff_scale", str(1.0))])
		goog_conv2_3x3reduce_conv2_3x3 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn3),
							   ("diff_scale", str(1.0))])
		goog_conv2_relu3x3_conv2_norm2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn4),
							   ("diff_scale", str(1.0))])
		goog_conv2_norm2_pool2_3x3s2 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn5),
							   ("diff_scale", str(1.0))])
		goog_inception3a_relu1x1_inception3a_3x3reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn6),
							   ("diff_scale", str(1.0))])
		goog_inception3a_3x3reduce_inception3a_3x3 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn7),
							   ("diff_scale", str(1.0))])
		goog_inception3a_3x3_inception3a_5x5reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn8),
							   ("diff_scale", str(1.0))])
		goog_inception3a_5x5reduce_inception3a_5x5 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn9),
							   ("diff_scale", str(1.0))])
		goog_inception3a_5x5_inception3a_pool = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn10),
							   ("diff_scale", str(1.0))])
		goog_inception3a_poolproj_inception3a_output = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn11),
							   ("diff_scale", str(1.0))])
		goog_inception3b_1x1_inception3b_3x3reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn12),
							   ("diff_scale", str(1.0))])
		goog_inception3b_3x3reduce_inception3b_3x3 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn13),
							   ("diff_scale", str(1.0))])
		goog_inception3b_3x3_inception3b_5x5reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn14),
							   ("diff_scale", str(1.0))])
		goog_inception3b_5x5reduce_inception3b_5x5 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn15),
							   ("diff_scale", str(1.0))])
		goog_inception3b_5x5_inception3b_pool = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn16),
							   ("diff_scale", str(1.0))])
		goog_inception3b_poolproj_inception3b_output = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn17),
							   ("diff_scale", str(1.0))])





		goog_inception4a_1x1_inception4a_3x3reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn18),
							   ("diff_scale", str(1.0))])
		goog_inception4a_3x3reduce_inception4a_3x3 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn19),
							   ("diff_scale", str(1.0))])
		goog_inception4a_3x3reduce_inception4a_5x5reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn20),
							   ("diff_scale", str(1.0))])
		goog_inception4a_5x5reduce_inception41_5x5 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn21),
							   ("diff_scale", str(1.0))])
		goog_inception4a_5x5_inception4a_pool = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn22),
							   ("diff_scale", str(1.0))])
		goog_inception4a_poolproj_inception4a_output = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn23),
							   ("diff_scale", str(1.0))])

		
		goog_inception4b_1x1_inception4b_3x3reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn24),
							   ("diff_scale", str(1.0))])
		goog_inception4b_3x3reduce_inception4b_3x3 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn25),
							   ("diff_scale", str(1.0))])
		goog_inception4b_3x3_inception4b_5x5reduce = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn26),
							   ("diff_scale", str(1.0))])
		goog_inception4b_5x5reduce_inception4b_5x5 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn27),
							   ("diff_scale", str(1.0))])
		goog_inception4b_5x5_inception4b_pool = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn28),
							   ("diff_scale", str(1.0))])
		goog_inception4b_poolproj_inception4b_output = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "GAUSSIAN"),
							   ("pass", "false"),
							   ("noise_param", noise_param_str_gn29),
							   ("diff_scale", str(1.0))])



		goog_inception4b_output_inception4c_1x1 = OrderedDict([("lr_mult", str(0)),
							   ("decay_mult", str(0)),
							   ("ntype", "UNIFORM"),
							   ("pass", "false"),
							   ("noise_param", noise_param_q), 
							   ("diff_scale", str(1.0))])
		
		layers = OrderedDict([("goog_data_n0_conv1", goog_data_n0_conv1),
			("goog_conv1_relu7x7_pool1_3x3s2", goog_conv1_relu7x7_pool1_3x3s2),
			("goog_pool1_norm1_conv2_3x3reduce", goog_pool1_norm1_conv2_3x3reduce),
			("goog_conv2_3x3reduce_conv2_3x3", goog_conv2_3x3reduce_conv2_3x3),
			("goog_conv2_relu3x3_conv2_norm2", goog_conv2_relu3x3_conv2_norm2),
			("goog_conv2_norm2_pool2_3x3s2", goog_conv2_norm2_pool2_3x3s2),
			("goog_inception3a_relu1x1_inception3a_3x3reduce", goog_inception3a_relu1x1_inception3a_3x3reduce),
			("goog_inception3a_3x3reduce_inception3a_3x3", goog_inception3a_3x3reduce_inception3a_3x3),
			("goog_inception3a_3x3_inception3a_5x5reduce", goog_inception3a_3x3_inception3a_5x5reduce),
			("goog_inception3a_5x5reduce_inception3a_5x5", goog_inception3a_5x5reduce_inception3a_5x5),
			("goog_inception3a_5x5_inception3a_pool", goog_inception3a_5x5_inception3a_pool),
			("goog_inception3a_poolproj_inception3a_output", goog_inception3a_poolproj_inception3a_output),
			("goog_inception3b_1x1_inception3b_3x3reduce", goog_inception3b_1x1_inception3b_3x3reduce),
			("goog_inception3b_3x3reduce_inception3b_3x3", goog_inception3b_3x3reduce_inception3b_3x3),
			("goog_inception3b_3x3_inception3b_5x5reduce", goog_inception3b_3x3_inception3b_5x5reduce),
			("goog_inception3b_5x5reduce_inception3b_5x5", goog_inception3b_5x5reduce_inception3b_5x5),
			("goog_inception3b_5x5_inception3b_pool", goog_inception3b_5x5_inception3b_pool),
			("goog_inception3b_poolproj_inception3b_output", goog_inception3b_poolproj_inception3b_output),

			("goog_inception4a_1x1_inception4a_3x3reduce", goog_inception4a_1x1_inception4a_3x3reduce),
			("goog_inception4a_3x3reduce_inception4a_3x3", goog_inception4a_3x3reduce_inception4a_3x3),
			("goog_inception4a_3x3reduce_inception4a_5x5reduce", goog_inception4a_3x3reduce_inception4a_5x5reduce),
			("goog_inception4a_5x5reduce_inception41_5x5", goog_inception4a_5x5reduce_inception41_5x5),
			("goog_inception4a_5x5_inception4a_pool", goog_inception4a_5x5_inception4a_pool),
			("goog_inception4a_poolproj_inception4a_output", goog_inception4a_poolproj_inception4a_output),

			("goog_inception4b_1x1_inception4b_3x3reduce", goog_inception4b_1x1_inception4b_3x3reduce),
			("goog_inception4b_3x3reduce_inception4b_3x3", goog_inception4b_3x3reduce_inception4b_3x3),
			("goog_inception4b_3x3_inception4b_5x5reduce", goog_inception4b_3x3_inception4b_5x5reduce),
			("goog_inception4b_5x5reduce_inception4b_5x5", goog_inception4b_5x5reduce_inception4b_5x5),
			("goog_inception4b_5x5_inception4b_pool", goog_inception4b_5x5_inception4b_pool),
			("goog_inception4b_poolproj_inception4b_output", goog_inception4b_poolproj_inception4b_output),

			("goog_inception4b_output_inception4c_1x1", goog_inception4b_output_inception4c_1x1)])
		return layers