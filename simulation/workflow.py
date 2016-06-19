# This is a complete workflow for redeye. Starting from prototxt 
# generation, it finetunes the parameters of noise-inserted models,
# run for testing and plots the results under /plots folder.

# Usage: python workflow.py --test_iter t --tune_iter i --dest d --g x,x --q y,y

import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(os.getcwd())))
from proto_writer import snr_sweep as sweep
from lib import utils, setup, grammar
import caffe
caffe.set_mode_gpu()
"""
Set up the template directories and tuning parameters
"""
dir_config = setup.config()
train_prototxt_template = dir_config['train_prototxt_template']
solver_prototxt_template = dir_config['solver_prototxt_template']
pretrained_model = dir_config['pretrained_model']
gaussian_interval = dir_config['gaussian_intvl']
uniform_interval = dir_config['uniform_intvl']
depth = dir_config['depth']
digitization_bits = dir_config['digitization_bits']

argv = argparse.ArgumentParser(description='Sweep SNR space, train and generate plots')
argv.add_argument('--test_iter', type=int, help='number of iterations for testing', required=True)
argv.add_argument('--tune_iter', type=int, help='number of iterations for fine tuning', required=True)
argv.add_argument('--dest', type=str, help='complete store path of stat csv file', required=True)
argv.add_argument('--g', type=grammar._range_format, help='gaussian noise layer SNR range', required=True)
argv.add_argument('--q', type=grammar._range_format, help='uniform noise layer SNR range', required=True)

arg = argv.parse_args()
result_filename = arg.dest
test_iter = arg.test_iter
tuning_iter = arg.tune_iter
gaussian_range = grammar.parse_rf(arg.g)
uniform_range = grammar.parse_rf(arg.q)

#########################################################################
"""
Start workflow
"""

train_list = sweep(gaussian_range, gaussian_interval, uniform_range, 
	uniform_interval, depth, template=train_prototxt_template)

solvers, models = utils.write_solver_prototxt(solver_prototxt_template,
	 train_list, tuning_iter)

# save the original testing results on model with noise inserted only in data layer
control_net = caffe.Net(train_prototxt_template, pretrained_model, caffe.TEST)
utils.digitize_params(control_net, digitization_bits, depth=depth)
utils.append_to_csv("0_0_0_0", utils.test_accuracy(control_net,
	test_iter), result_filename)
del(control_net)

for i in range(len(solvers)):
	solver = solvers[i]
	if not os.path.isfile(models[i]):
		os.system('../caffe/build/tools/caffe train --solver=' + solver + \
			' -weights ' + pretrained_model)
	finet = caffe.Net(train_list[i], models[i], caffe.TEST)
	utils.digitize_params(finet, digitization_bits, depth=depth)
	# write intermediate results to file
	proto_filename = os.path.basename(train_list[i]).split('.')[0].split('_')\
		[:-(depth + 2):-1]
	utils.append_to_csv("_".join(proto_filename), utils.test_accuracy(finet, test_iter),
		result_filename)
	del(finet)

utils.plot_results(result_filename, os.path.join('../plots/', 
	os.path.basename(result_filename).split('.')[0] + '.png'))
