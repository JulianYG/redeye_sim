import numpy as np
from google.protobuf import text_format as proto
from caffe.proto import caffe_pb2
import os
import csv
from ast import literal_eval as le
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict

def quantize_params(net, bitnum, depth=5):
	"""
	Given path to train prototxt and trained models for 
	fine tuning parameters, and number of bits for 
	quantization, scale the parameters to fit the n bits
	of accuracy
	"""
	for param in net.params.items()[:depth - 1]:
		if "noise" not in param[0]:
			for blob in param[1]:
				_digitize(blob.data, bitnum)

def _digitize(data, bnum):
	i = data.std()*6/(2.**bnum - 1)
	minVal = data.min()
	if minVal < data.mean() - 3 * data.std():
		minVal = data.mean() - 3 * data.std()
	maxVal = data.max()
	if maxVal > data.mean() + 3 * data.std():
		maxVal = data.mean() + 3 * data.std()
	# clamp the data
	data[data > maxVal] = maxVal
	data[data < minVal] = minVal
	# quantize the data
	data = i * np.floor((data - minVal) / i) + minVal

def write_solver_prototxt(template, train_prototxt, maxiter, 
	path='../prototxt/solver/redeye/'):
	model_name, proto_name = [], []
	for prototxt in train_prototxt:
		solver = caffe_pb2.SolverParameter()
		proto.Merge((open(template).read()), solver)
		solver.snapshot_prefix = os.path.join('../models/googlenet/snapshots/',
			os.path.basename(prototxt).split('.')[0])
		solver.max_iter = maxiter
		solver.snapshot = maxiter
		model_name.append(os.path.join('../models/googlenet/snapshots/',
			os.path.basename(prototxt).split('.')[0] + \
			"_iter_" + str(maxiter) + ".caffemodel"))
		solver.net = prototxt
		solver.test_interval = 50000
		new_solver = proto.MessageToString(solver)
		file_name = os.path.join(path, 'goog_solver_' + \
			os.path.basename(prototxt).split('.')[0] + '.prototxt')
		proto_name.append(file_name)
		with open(file_name, 'w') as new_proto:
			new_proto.write(new_solver)
	return proto_name, model_name

def test_accuracy(net, iters):

	res = {"loss3/top-5": 0.0, "loss3/top-1": 0.0,
		"loss2/top-5": 0.0, "loss2/top-1": 0.0,
		"loss1/top-5": 0.0, "loss1/top-1": 0.0}

	for i in range(iters):
		out = net.forward()
		for typ in res.keys():
			if typ in out.keys():
				res[typ] += out[typ] / iters
	return res

def write_to_csv_bin(res_dic, filename):
	writer = csv.writer(open(os.path.join('../stats/',
		filename), 'wb'))
	for key, value in res_dic.items():
		writer.writerow([key, value])

def append_to_csv(name, acc, filename):
	with open(filename,'ab') as file:
		writer = csv.writer(file)
		writer.writerow([name, acc])

def row2col(raw_stat):
	processed_stat = OrderedDict()
	for key, value in raw_stat.items():
		quant_snr = key[-1]
		if quant_snr not in processed_stat:
			d = {}
			for loss_type, acc in value.items():
				if loss_type not in d:
					d[loss_type] = {}
				d[loss_type][key[0]] = acc
			processed_stat[quant_snr] = d
		else:
			for loss_type, acc in value.items():
				processed_stat[quant_snr][loss_type][key[0]] = acc
	return processed_stat

def plot_results(filename, plot_dir):
	raw_stat = OrderedDict()
	with open(filename, 'r') as f:
		dic = OrderedDict(csv.reader(f))
		for key, value in dic.items():
			raw_stat[tuple(map(int, key.split('_')))] = le(value)
	processed_stat = row2col(raw_stat)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	c = ['r','g','b','y','m','c']
	for qSNR, acc in processed_stat.items():
		for i in range(len(acc)):
			xs = np.array(acc[acc.keys()[i]].keys())+(i-len(acc)/2)*0.4
			ys = acc[acc.keys()[i]].values()
			ax.bar(xs, ys, width=0.4, zs=qSNR, zdir='y', 
				color=c[i], align='center', label=acc.keys())
			for x, y in zip(xs, ys):
				ax.text(x, qSNR, y, '%2.2f'%y, ha='left', va='bottom')

	ax.set_xlabel('Gaussian SNR (dB)')
	ax.set_ylabel('Quantization SNR (dB)')
	ax.set_zlabel('Testing Accuracy')
	#plt.legend()
	plt.show()
	fig.savefig(plot_dir)
