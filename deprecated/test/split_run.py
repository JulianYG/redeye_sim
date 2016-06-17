import caffe
import setup
import os
import numpy as np
import h5py
import time

caffe.set_mode_gpu()
caffe_dir = setup.config().getLoc('caffeRoot')

layers = ['data', 'label', 'conv1/7x7_s2', 'pool1/3x3_s2', 'pool1/norm1', 'conv2/3x3_reduce',
'conv2/3x3', 'conv2/norm2', 'pool2/3x3_s2', 'inception_3a/1x1', 'inception_3a/3x3_reduce',
'inception_3a/3x3', 'inception_3a/5x5_reduce', 'inception_3a/5x5', 'inception_3a/pool',
'inception_3a/pool_proj', 'inception_3a/output', 'inception_3b/1x1', 'inception_3b/3x3_reduce',
'inception_3b/3x3', 'inception_3b/5x5_reduce', 'inception_3b/5x5', 'inception_3b/pool', 'inception_3b/pool_proj',
'inception_3b/output', 'pool3/3x3_s2', 'inception_4a/1x1', 'inception_4a/3x3_reduce',
'inception_4a/3x3', 'inception_4a/5x5_reduce', 'inception_4a/5x5', 'inception_4a/pool', 'inception_4a/pool_proj',
'inception_4a/output', 'inception_4b/1x1', 'inception_4b/3x3_reduce',
'inception_4b/3x3', 'inception_4b/5x5_reduce', 'inception_4b/5x5', 'inception_4b/pool', 'inception_4b/pool_proj',
'inception_4b/output', 'inception_4c/1x1', 'inception_4c/3x3_reduce',
'inception_4c/3x3', 'inception_4c/5x5_reduce', 'inception_4c/5x5', 'inception_4c/pool', 'inception_4c/pool_proj',
'inception_4c/output', 'inception_4d/1x1', 'inception_4d/3x3_reduce',
'inception_4d/3x3', 'inception_4d/5x5_reduce', 'inception_4d/5x5', 'inception_4d/pool', 'inception_4d/pool_proj',
'inception_4d/output', 'inception_4e/1x1', 'inception_4e/3x3_reduce',
'inception_4e/3x3', 'inception_4e/5x5_reduce', 'inception_4e/5x5', 'inception_4e/pool', 'inception_4e/pool_proj',
'inception_4e/output', 'pool4/3x3_s2', 'inception_5a/1x1', 'inception_5a/3x3_reduce',
'inception_5a/3x3', 'inception_5a/5x5_reduce', 'inception_5a/5x5', 'inception_5a/pool', 'inception_5a/pool_proj',
'inception_5a/output', 'inception_5b/1x1', 'inception_5b/3x3_reduce',
'inception_5b/3x3', 'inception_5b/5x5_reduce', 'inception_5b/5x5', 'inception_5b/pool', 'inception_5b/pool_proj',
'inception_5b/output', 'pool5/7x7_s1']

def split(layer, modelName='googlenet'):
	model = os.path.join(caffe_dir, 'models/bvlc_' + modelName, '/bvlc_' + modelName + '.caffemodel')
	deploy = os.path.join(caffe_dir, 'models/bvlc_' + modelName, 'deploy.prototxt')
	net = caffe.Net(deploy, model, caffe.TEST)

	cutoff_pt = layers.index(layer)
	start1 = time.time()
	ana_out = net.forward(end=layers[cutoff_pt - 1])
	end1 = time.time()
	time1 = end1 - start1
	np.save(os.path.join(caffe_dir, 'split_' + layer + '.npy'), ana_out)

	net.blobs[layers[cutoff_pt - 1]].data = ana_out
	start2 = time.time()
	res = net.forward(start=layer)
	end2 = time.time()
	time2 = end2 - start2
	return time1, time2

print split('conv2/norm2')

