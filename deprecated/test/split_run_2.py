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
ana_out = net.forward(end='conv1/7x7_s2')
end1 = time.time()
time1 = end1 - start1
print time1
np.save(os.path.join(caffe_dir, 'split_pool1s2.npy'), ana_out)

net.blobs['conv1/7x7_s2'].data = ana_out
start2 = time.time()
res = net.forward(start='pool1/3x3_s2')
end2 = time.time()
time2 = end2 - start2
print time2
