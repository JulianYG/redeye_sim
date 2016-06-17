from collections import OrderedDict
import sys
import os
sys.path.insert(0,"../caffe_test")
import setup

path = setup.config()
alex = path.getLoc('alexTabPath')
goog = path.getLoc('googTabPath')

# alex = '/home/roblkw/caffe_exp/caffe/alextab'
# goog = '/home/roblkw/caffe_exp/caffe/googletab'

# below for gaussian noise; change ntype to 0
# noise_param_str_an0 = _build_g_param()
# noise_param_str_an1 = _build_g_param()
# noise_param_str_an2 = _build_g_param()
# noise_param_str_an3 = _build_g_param()
# noise_param_str_an4 = _build_g_param()

# alex_data_n0_conv1 = OrderedDict([("lr_mult", str(0)),	
# 					   ("decay_mult", str(0)),	# determines learning or not
# 					   ("ntype", "GAUSSIAN"),	# determines noise type, 0 for gaussian
# 					   						# 1 for poisson, 2 for uniform
# 					   ("pass", "false"),	# whether data pass through
# 					   ("noise_param", noise_param_str_an0),	# noise param
# 					   ("diff_scale", str(1.0))])		# scaling factor of diff values
# alex_pool1_n1_conv2 = OrderedDict([("lr_mult", str(0)),
# 					   ("decay_mult", str(0)),
# 					   ("ntype", "GAUSSIAN"),
# 					   ("pass", "false"),
# 					   ("noise_param", noise_param_str_an1),
# 					   ("diff_scale", str(1.0))])
# alex_pool2_n2_conv3 = OrderedDict([("lr_mult", str(0)),
# 					   ("decay_mult", str(0)),
# 					   ("ntype", "GAUSSIAN"),
# 					   ("pass", "false"),
# 					   ("noise_param", noise_param_str_an2),
# 					   ("diff_scale", str(1.0))])
# alex_conv3_n3_conv4 = OrderedDict([("lr_mult", str(0)),
# 					   ("decay_mult", str(0)),
# 					   ("ntype", "GAUSSIAN"),
# 					   ("pass", "false"),
# 					   ("noise_param", noise_param_str_an3),
# 					   ("diff_scale", str(1.0))])
# alex_conv4_n4_conv5 = OrderedDict([("lr_mult", str(0)),
# 					   ("decay_mult", str(0)),
# 					   ("ntype", "GAUSSIAN"),
# 					   ("pass", "false"),
# 					   ("noise_param", noise_param_str_an4),
# 					   ("diff_scale", str(1.0))])

# layers_alex = {"alex_data_n0_conv1": alex_data_n0_conv1, 
# 	"alex_pool1_n1_conv2": alex_pool1_n1_conv2,
# 	"alex_pool2_n2_conv3": alex_pool2_n2_conv3,
# 	"alex_conv3_n3_conv4": alex_conv3_n3_conv4, 
# 	"alex_conv4_n4_conv5": alex_conv4_n4_conv5}

def build_g_param(mean=0, std=10e-7, min_sd=10e-10, max_sd=None, scale=1):

	param_to_build = "gaussian_param {\n      mean: " + str(mean) + '\n      '
	param_to_build += "stddev: " + str(std) + '\n      '
	if min_sd:
 		param_to_build += "min_sd: " + str(min_sd) + '\n      '
 	if max_sd:
 		param_to_build += "max_sd: " + str(max_sd) + '\n      '
 	param_to_build += "scale: " + str(scale) + "\n    }\n"
	return param_to_build

def build_q_param(min_u=-1, max_u=1, scale=1):
	param_to_build = "uniform_param {\n      min_u: " + str(min_u) + '\n      max_u: '
	param_to_build += str(max_u) + '\n      scale: ' + str(scale) + '\n    }\n'
	return param_to_build

def build_new_proto(tab, layers, name,depth):
	"""
	tab indicates whether to use googlenet or alexnet as model 
	name is the new name suffix of deploy.prototxt
	"""
	fp = ''
	params = ["lr_mult", "decay_mult", "ntype", "pass", "noise_param", "diff_scale"]
	if tab.lower() == 'alextab':
		fp = os.path.join(alex, 'deploy_stencil_'+str(depth)+'g.prototxt')
		export = os.path.join(alex, 'deploy_' + str(name) + '.prototxt')
	# if tab.lower() == 'alextab':
	# 	fp = os.path.join(alex, 'deploy_stencil.prototxt')
	# 	export = os.path.join(alex, 'deploy_' + str(name) + '.prototxt')
	elif tab.lower() == 'googletab':
		print goog
		fp = os.path.join(goog, 'deploy_stencil_'+str(depth)+'g.prototxt')
		export = os.path.join(goog, 'deploy_' + str(name) + '.prototxt')
	else:
		raise NameError("Net is not recognized!")	
	with open(fp, 'r') as fd:
		stencil = fd.read()
	for layer_name in layers:
		for param_type in params:
			recognition_str = '$' + layer_name + '_' + param_type + '&'
			stencil = stencil.replace(recognition_str, layers[layer_name][param_type])
	with open(export, 'w') as x:
		x.write(stencil)

# build_new_proto('googletab', 'L'+str(l)+'g'+str(i)+'q'+str(j+1))
