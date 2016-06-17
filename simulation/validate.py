# Test the performance of model on validation dataset
# Usage: python validate.py --iter i --proto s --model m --f f.csv --q x --d y

import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(os.getcwd())))
from lib import utils, setup
import caffe
caffe.set_mode_gpu()

argv = argparse.ArgumentParser(description='Sweep SNR space, train and generate plots')
argv.add_argument('--iter', type=int, help='number of iterations for testing', required=True)
argv.add_argument('--proto', type=str, help='the full path of prototxt', required=True)
argv.add_argument('--model', type=str, help='the full path of trained model', required=True)
argv.add_argument('--f', type=str, help='the full path to save result file', required=True)
argv.add_argument('--q', type=int, help='number of quantization bits', required=True)
argv.add_argument('--d', type=int, help='depth of digitization', required=True)

arg = argv.parse_args()
iter_num = arg.iter
prototxt = arg.proto
model = arg.model
stat_file = argv.f
quantization_bits = argv.q
quantization_depth = argv.d

finet = caffe.Net(prototxt, model, caffe.TEST)
utils.quantize_params(finet, quantization_bits, depth=quantization_depth)
utils.append_to_csv('validate', utils.test_accuracy(finet, iter_num), stat_file)

