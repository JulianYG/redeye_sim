from collections import OrderedDict
import os
import setup

path = setup.config()
alex = path.getLoc('alexTabPath')
goog = path.getLoc('googTabPath')

def _build_g_param(mean=0, std=10e-7, min_sd=None, max_sd=None, scale=1):

	param_to_build = "gaussian_param {\n      mean: " + str(mean) + '\n      '
	param_to_build += "stddev: " + str(std) + '\n      '
	if min_sd:
 		param_to_build += "min_sd: " + str(min_sd) + '\n      '
 	if max_sd:
 		param_to_build += "max_sd: " + str(max_sd) + '\n      '
 	param_to_build += "scale: " + str(scale) + "\n    }\n"
	return param_to_build

def _build_q_param(min_u=-1, max_u=1, scale=1):
	param_to_build = "uniform_param {\n      min_u: " + str(min_u) + '\n      max_u: '
	param_to_build += str(max_u) + '\n      scale: ' + str(scale) + '\n    }\n'
	return param_to_build

params = ["lr_mult", "decay_mult", "ntype", "pass", "noise_param", "diff_scale"]

# below for gaussian noise; change ntype to 0
noise_param_str_an0 = _build_g_param()
noise_param_str_an1 = _build_g_param()
noise_param_str_an2 = _build_g_param()
noise_param_str_an3 = _build_g_param()
noise_param_str_an4 = _build_g_param()

noise_param_str_gn0 = _build_g_param()
noise_param_str_gn1 = _build_g_param()
noise_param_str_gn2 = _build_g_param()
noise_param_str_gn3_0 = _build_g_param()
noise_param_str_gn3_1 = _build_g_param()
noise_param_str_gn3_2 = _build_g_param()
noise_param_str_gn3_3 = _build_g_param()
noise_param_str_gn4_0 = _build_g_param()
noise_param_str_gn4_1 = _build_g_param()
noise_param_str_gn5 = _build_g_param()

# below for quantization noise; change ntype to 2
# noise_param_str_an0 = _build_q_param()
# noise_param_str_an1 = _build_q_param()
# noise_param_str_an2 = _build_q_param()
# noise_param_str_an3 = _build_q_param()
# noise_param_str_an4 = _build_q_param()

# noise_param_str_gn0 = _build_q_param()
# noise_param_str_gn1 = _build_q_param()
# noise_param_str_gn2 = _build_q_param()
# noise_param_str_gn3_0 = _build_q_param()
# noise_param_str_gn3_1 = _build_q_param()
# noise_param_str_gn3_2 = _build_q_param()
# noise_param_str_gn3_3 = _build_q_param()
# noise_param_str_gn4_0 = _build_q_param()
# noise_param_str_gn4_1 = _build_q_param()
# noise_param_str_gn5 = _build_q_param()

alex_data_n0_conv1 = OrderedDict([("lr_mult", str(0)),	
					   ("decay_mult", str(0)),	# determines learning or not
					   ("ntype", "GAUSSIAN"),	# indicates noise type
					   ("pass", "false"),	# whether data pass through
					   ("noise_param", noise_param_str_an0),	# noise param
					   ("diff_scale", str(1.0))])		# scaling factor of diff values
alex_pool1_n1_conv2 = OrderedDict([("lr_mult", str(0)),
					   ("decay_mult", str(0)),
					   ("ntype", "GAUSSIAN"),
					   ("pass", "false"),
					   ("noise_param", noise_param_str_an1),
					   ("diff_scale", str(1.0))])
alex_pool2_n2_conv3 = OrderedDict([("lr_mult", str(0)),
					   ("decay_mult", str(0)),
					   ("ntype", "GAUSSIAN"),
					   ("pass", "false"),
					   ("noise_param", noise_param_str_an2),
					   ("diff_scale", str(1.0))])
alex_conv3_n3_conv4 = OrderedDict([("lr_mult", str(0)),
					   ("decay_mult", str(0)),
					   ("ntype", "GAUSSIAN"),
					   ("pass", "false"),
					   ("noise_param", noise_param_str_an3),
					   ("diff_scale", str(1.0))])
alex_conv4_n4_conv5 = OrderedDict([("lr_mult", str(0)),
					   ("decay_mult", str(0)),
					   ("ntype", "GAUSSIAN"),
					   ("pass", "false"),
					   ("noise_param", noise_param_str_an4),
					   ("diff_scale", str(1.0))])
goog_data_n0_conv1 = OrderedDict([("lr_mult", str(0)),
					   ("decay_mult", str(0)),
					   ("ntype", "GAUSSIAN"),
					   ("pass", "false"),
					   ("noise_param", noise_param_str_gn0),
					   ("diff_scale", str(1.0))])
goog_pool1_n1_conv2_3x3red = OrderedDict([("lr_mult", str(0)),
					   ("decay_mult", str(0)),
					   ("ntype", "GAUSSIAN"),
					   ("pass", "false"),
					   ("noise_param", noise_param_str_gn1),
					   ("diff_scale", str(1.0))])
goog_conv2_3x3red_n2_conv2_3x3 = OrderedDict([("lr_mult", str(0)),
					   ("decay_mult", str(0)),
					   ("ntype", "GAUSSIAN"),
					   ("pass", "false"),
					   ("noise_param", noise_param_str_gn2),
					   ("diff_scale", str(1.0))])
goog_pool2_n3_incep3a_1x1 = OrderedDict([("lr_mult", str(0)),
					   ("decay_mult", str(0)),
					   ("ntype", "GAUSSIAN"),
					   ("pass", "false"),
					   ("noise_param", noise_param_str_gn3_0),
					   ("diff_scale", str(1.0))])
goog_pool2_n3_incep3a_3x3red = OrderedDict([("lr_mult", str(0)),
					   ("decay_mult", str(0)),
					   ("ntype", "GAUSSIAN"),
					   ("pass", "false"),
					   ("noise_param", noise_param_str_gn3_1),
					   ("diff_scale", str(1.0))])
goog_pool2_n3_incep3a_5x5red = OrderedDict([("lr_mult", str(0)),
					   ("decay_mult", str(0)),
					   ("ntype", "GAUSSIAN"),
					   ("pass", "false"),
					   ("noise_param", noise_param_str_gn3_2),
					   ("diff_scale", str(1.0))])
goog_incep3a_pool_n3_incep3a_pool_proj = OrderedDict([("lr_mult", str(0)),
					   ("decay_mult", str(0)),
					   ("ntype", "GAUSSIAN"),
					   ("pass", "false"),
					   ("noise_param", noise_param_str_gn3_3),
					   ("diff_scale", str(1.0))])
goog_incep3a_3x3relred_n4_incep3a_3x3 = OrderedDict([("lr_mult", str(0)),
					   ("decay_mult", str(0)),
					   ("ntype", "GAUSSIAN"),
					   ("pass", "false"),
					   ("noise_param", noise_param_str_gn4_0),
					   ("diff_scale", str(1.0))])
goog_incep3a_5x5red_n4_incep3a_5x5 = OrderedDict([("lr_mult", str(0)),
					   ("decay_mult", str(0)),
					   ("ntype", "GAUSSIAN"),
					   ("pass", "false"),
					   ("noise_param", noise_param_str_gn4_1),
					   ("diff_scale", str(1.0))])
goog_incep3a_out_n5_incep3b = OrderedDict([("lr_mult", str(0)),
					   ("decay_mult", str(0)),
					   ("ntype", "GAUSSIAN"),
					   ("pass", "false"),
					   ("noise_param", noise_param_str_gn5),
					   ("diff_scale", str(1.0))])

layers_alex = {"alex_data_n0_conv1": alex_data_n0_conv1, 
	"alex_pool1_n1_conv2": alex_pool1_n1_conv2,
	"alex_pool2_n2_conv3": alex_pool2_n2_conv3,
	"alex_conv3_n3_conv4": alex_conv3_n3_conv4, 
	"alex_conv4_n4_conv5": alex_conv4_n4_conv5}

layers_goog = {"goog_data_n0_conv1": goog_data_n0_conv1,
	"goog_pool1_n1_conv2_3x3red": goog_pool1_n1_conv2_3x3red,
	"goog_conv2_3x3red_n2_conv2_3x3": goog_conv2_3x3red_n2_conv2_3x3,
	"goog_pool2_n3_incep3a_1x1": goog_pool2_n3_incep3a_1x1,
	"goog_pool2_n3_incep3a_3x3red": goog_pool2_n3_incep3a_3x3red,
	"goog_pool2_n3_incep3a_5x5red": goog_pool2_n3_incep3a_5x5red,
	"goog_incep3a_pool_n3_incep3a_pool_proj": goog_incep3a_pool_n3_incep3a_pool_proj,
	"goog_incep3a_3x3relred_n4_incep3a_3x3": goog_incep3a_3x3relred_n4_incep3a_3x3,
	"goog_incep3a_5x5red_n4_incep3a_5x5": goog_incep3a_5x5red_n4_incep3a_5x5,
	"goog_incep3a_out_n5_incep3b": goog_incep3a_out_n5_incep3b}

def build_new_proto(tab, name):
	"""
	tab indicates whether to use googlenet or alexnet as model 
	name is the new name suffix of deploy.prototxt
	"""
	fp = ''
	if tab.lower() == 'alextab':
		fp = os.path.join(alex, 'train_val_stencil.prototxt')
		export = os.path.join(alex, 'train_val_' + str(name) + '.prototxt')
		layers = layers_alex
	elif tab.lower() == 'googletab':
		fp = os.path.join(goog, 'train_val_stencil.prototxt')
		export = os.path.join(goog, 'train_val_' + str(name) + '.prototxt')
		layers = layers_goog
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





