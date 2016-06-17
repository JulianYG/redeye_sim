import caffe
import os
import numpy
import setup

def write_param_data(weights_root, modelNet):

	if not os.path.exists(weights_root): 
		os.makedirs(weights_root)

	for param_name, param_blob_vec in modelNet.params.items():
		# those are all length two blobVecs, 0 as weights and 1 as bias
		param_name_str = param_name.split('/')
		valid_param_fn = param_name_str[0] + '\\' + param_name_str[1]
		with open(os.path.join(weights_root, valid_param_fn + '_weights.dat'), 'w') as w:
			weights = param_blob_vec[0].data
			w.write("max: " + str(weights.max()) + '\n')
			w.write("min: " + str(weights.min()) + '\n')
			w.write("max(abs): " + str(numpy.absolute(weights).max()) + '\n')
			w.write("min(abs): " + str(numpy.absolute(weights).min()) + '\n')
			w.write("mean: " + str(numpy.mean(weights, axis=0)) + '\n')
			w.write("median: " + str(numpy.median(weights)) + '\n')
			w.write("stddev: " + str(weights.std()) + '\n')
		if len(param_blob_vec) > 1:
			with open(os.path.join(weights_root, valid_param_fn + '_bias.dat'), 'w') as b:
				bias = param_blob_vec[1].data
				b.write("max: " + str(bias.max()) + '\n')
				b.write("min: " + str(bias.min()) + '\n')
				b.write("max(abs): " + str(numpy.absolute(bias).max()) + '\n')
				b.write("min(abs): " + str(numpy.absolute(bias).min()) + '\n')
				b.write("mean: " + str(numpy.mean(bias, axis=0)) + '\n')
				b.write("median: " + str(numpy.median(bias)) + '\n')
				b.write("stddev: " + str(bias.std()) + '\n')

def write_blob_data(data_root, modelNet):

	if not os.path.exists(data_root):
		os.makedirs(data_root)
	for blob_name, blob_obj in modelNet.blobs.items():
		data = blob_obj.data
		diff = blob_obj.diff
		if '/' in blob_name:
			blob_name_str = blob_name.split('/')
			valid_blob_fn = blob_name_str[0] + '\\' + blob_name_str[1]
		else:
			valid_blob_fn = blob_name
		with open(os.path.join(data_root, valid_blob_fn + '_data.dat'), 'w') as da:
			da.write("max: " + str(data.max()) + '\n')
			da.write("min: " + str(data.min()) + '\n')
			da.write("max(abs): " + str(numpy.absolute(data).max()) + '\n')
			da.write("min(abs): " + str(numpy.absolute(data).min()) + '\n')
			da.write("mean: " + str(numpy.mean(data)) + '\n')
			da.write("median: " + str(numpy.median(data)) + '\n')
			da.write("stddev: " + str(data.std()) + '\n')
		with open(os.path.join(data_root, valid_blob_fn + '_diff.dat'), 'w') as di:
			di.write("max: " + str(diff.max()) + '\n')
			di.write("min: " + str(diff.min()) + '\n')
			di.write("max(abs): " + str(numpy.absolute(diff).max()) + '\n')
			di.write("min(abs): " + str(numpy.absolute(diff).min()) + '\n')
			di.write("mean: " + str(numpy.mean(diff)) + '\n')
			di.write("median: " + str(numpy.median(diff)) + '\n')
			di.write("stddev: " + str(diff.std()) + '\n')

def get_bandwidth(logFileSavePath, modelNet):
	with open(logFileSavePath, 'w') as fd:
		for layer_name, blob_data in modelNet.blobs.items():
			fd.write(layer_name + "\n" + str(blob_data.count * 32 / 8) + " bytes for single pass, carries 2x\n")

def get_blobshape(logFileSavePath, modelNet):
	with open(logFileSavePath, 'w') as fd:
		for layer_name, blob_data in modelNet.blobs.items():
			fd.write(layer_name + "\t" + str(blob_data.num) + '\t' + str(blob_data.channels) + '\t' + str(blob_data.height) + '\t' + str(blob_data.width) + '\n')

