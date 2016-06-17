# setup the data path for train/test

import os
from google.protobuf import text_format as proto
from caffe.proto import caffe_pb2

run_settings = '../configs/run.config'
data_settings = '../configs/data.config'

def config():
	return _load_configs(run_settings)

def _load_configs(path):
	config = {}
	with open(path, 'r') as f:
		for line in f:
			(key, val) = line.split()
			if val.isdigit():
				config[key] = int(val)
			else:
				config[key] = val
	return config

data_dirs = {}
with open(data_settings, 'r') as d:
	for line in d:
		(name, path) = line.split('=')
		data_dirs[name] = path

train_dir = os.path.join(data_dirs['DB'].strip('\n'), data_dirs['TRAIN_LMDB'].strip('\n'))
val_dir = os.path.join(data_dirs['DB'].strip('\n'), data_dirs['VAL_LMDB'].strip('\n'))

train_prototxt = _load_configs(run_settings)['train_prototxt_template']
val_prototxt = _load_configs(run_settings)['val_prototxt_template']


train_net = caffe_pb2.NetParameter()
proto.Merge((open(train_prototxt).read()), train_net)

for net in train_net.layer:
	if str(net.type) == "Data":
		net.data_param.batch_size = int(data_dirs['batch_size'])
		# in case of train
		if net.include[0].phase == 0:
			net.data_param.source = train_dir
		# in case of val
		if net.include[0].phase == 1:
			net.data_param.source = val_dir

	if str(net.type) == "Noise" and str(net.bottom[0]) == "data":
		net.noise_param.gaussian_param.stddev = float(data_dirs['noise_std'])
	net = net

modified_proto = proto.MessageToString(train_net)
with open(train_prototxt, 'w') as template:
	template.write(modified_proto)
del(train_net)

val_net = caffe_pb2.NetParameter()
proto.Merge((open(val_prototxt).read()), val_net)

for net in val_net.layer:
	if str(net.type) == "Data":
		net.data_param.batch_size = int(data_dirs['batch_size'])
		# in case of train
		if net.include[0].phase == 0:
			net.data_param.source = train_dir
		# in case of val
		if net.include[0].phase == 1:
			net.data_param.source = val_dir

	if str(net.type) == "Noise" and str(net.bottom[0]) == "data":
		net.noise_param.gaussian_param.stddev = float(data_dirs['noise_std'])
	net = net

modified_proto = proto.MessageToString(val_net)
with open(val_prototxt, 'w') as template:
	template.write(modified_proto)
del(val_net)

