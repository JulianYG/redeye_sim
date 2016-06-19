# Something outside the workflow but helpful for intuition.
# Directly feed uniform noise to input data layer, as well as 
# tunable gamma uncorrection, and digitize given depth layers of weights.
# It runs on given GoogLeNet model to generate prediction accuracy, and
# use original GoogLeNet train_val prototxt as template.
# Usage:
# python dataSNR_quant.py --s 10,60 --g 2.2 --i 5 --n gaussian --test_iter 1000 --qb 4,10 --d 3 --fn model_test
import sys, os, argparse, csv
sys.path.append(os.path.join(os.path.dirname(os.getcwd())))
from lib import utils, setup, grammar
from simulation import proto_writer as pw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import caffe
caffe.set_mode_gpu()

dir_config = setup.config()
train_prototxt_template = dir_config['train_prototxt_template']
pretrained_model = dir_config['pretrained_model']#'../models/googlenet/goog_gamma.caffemodel'


argv = argparse.ArgumentParser(description='Sweep SNR on image input, digitization bits on depth for testing')
argv.add_argument('--i', type=int, help='SNR interval', required=True)
argv.add_argument('--s', type=grammar._range_format, help='SNR range, e.g. 10,60', required=True)
argv.add_argument('--test_iter', type=int, help='number of iterations for testing', required=True)
argv.add_argument('--qb', type=grammar._range_format, help='digitization bits range, e.g. 6,8', required=True)
argv.add_argument('--d', type=int, help='depth of digitizing learned layer weights', required=True)
argv.add_argument('--g', type=float, help='gamma constant for uncorrection', required=False)
argv.add_argument('--n', type=str, help='noise type: gaussian or uniform', required=True)
argv.add_argument('--fn', type=str, help='single file name to save results without extension', required=True)
arg = argv.parse_args()
qb_low, qb_high = grammar.parse_rf(arg.qb)
snr_low, snr_high = grammar.parse_rf(arg.s)
depth = arg.d
noise_type = arg.n
digitization_bits = range(qb_low, qb_high + 1)
SNR = range(snr_low, snr_high + 1, arg.i)
i = arg.i
iter_num = arg.test_iter
gamma = arg.g
if gamma == None:
	gamma = 2.2
file_name = arg.fn
save_csv = os.path.join('../stats/', file_name + '.csv')
save_plot = os.path.join('../stats/', file_name + '.png')

protos = pw.snr_data_sweep(SNR, i, gamma, noise_type,
	template='../prototxt/train/origin/goog_val_data.prototxt')
for proto in protos:
	for q in digitization_bits:
		net = caffe.Net(proto, pretrained_model, caffe.TEST)
		utils.digitize_params(net, q, depth=depth)
		proto_snr = os.path.basename(proto).split('.')[0].split('_')[3]
		utils.append_to_csv(str(proto_snr) + "_" + str(q), 
			utils.test_accuracy(net, iter_num), save_name +'.csv')
        del(net)

utils.plot_results(save_csv, save_plot)







