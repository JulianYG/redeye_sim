import caffe
import setup
import os
import numpy as np
import h5py
import time

caffe_dir = setup.config().getLoc('caffeRoot')
model = os.path.join(caffe_dir, 'models/bvlc_googlenet/bvlc_googlenet.caffemodel')
deploy = os.path.join(caffe_dir, 'models/bvlc_googlenet/deploy.prototxt')
caffe.set_mode_gpu()
#net = caffe.Classifier(deploy, model)
net = caffe.Net(deploy, model, caffe.TEST)

start1 = time.time()
ana_out = net.forward(end='pool1/norm1')
end1 = time.time()
time1 = end1 - start1
print time1
np.save(os.path.join(caffe_dir, 'split_conv2red.npy'), ana_out)

net.blobs['pool1/norm1'].data = ana_out
start2 = time.time()
res = net.forward(start='conv2/3x3_reduce')
end2 = time.time()
time2 = end2 - start2
print time2
