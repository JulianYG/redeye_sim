import numpy as np
import sys
#import setup
import caffe

#croot = setup.config().getLoc('caffeRoot')
croot = 'home/roblkw/caffe_exp/caffe'
caffe.set_mode_cpu()

solver = caffe.SGDSolver('/home/roblkw/caffe_exp/caffe/googletab/solver.prototxt')
#solver = caffe.SGDSolver('/home/roblkw/caffe_exp/caffe/alextab/solver.prototxt')
solver.net.copy_from('/home/roblkw/Work/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel')
#solver.net.copy_from('/home/roblkw/Work/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel')
solver.step(1500)

solver.net.save('/home/roblkw/Work/caffe/models/bvlc_googlenet/simple_train.caffemodel')