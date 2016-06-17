from heapq import nlargest
from matplotlib import pyplot as plt
from time import sleep
import os
import numpy as np
import sys
croot = '/home/roblkw/Work/caffe/'
sys.path.insert(0, croot + 'python')
import caffe

# croot = '/home/julian/caffe/'
# train = '/home/julian/caffe/deg_train'
# val = '/home/julian/caffe/deg_val'
# test = '/home/julian/caffe/test'
# val = '/home/julian/caffe/deg_val'
# test = '/home/julian/caffe/test'

class ModelComparator():
	"""
	Test and compare the accuracy between models. Initialization nputs:
	the full model names for comparison, the full model type, including
	the bvlc_ string at the beginning. This module tests absolute accuracy
	while the 'testmodule' tests for relative accuracy.
	"""
	def __init__(self, baseModelName, degModelName, noisyModelName, modelType='bvlc_googlenet'):

		caffe.set_mode_cpu()
		self._baseNet = caffe.Net(os.path.join(croot, 'models', modelType, 'train_val_prob.prototxt'),
			os.path.join(croot, 'models', modelType, baseModelName), caffe.TEST)
		self._degNet = caffe.Net(os.path.join(croot, 'models', modelType, 'train_val.prototxt'),
			os.path.join(croot, 'models', modelType, degModelName), caffe.TEST)
		self._noisyNet = caffe.Net(os.path.join(croot, 'models', modelType, 'train_val.prototxt'),
			os.path.join(croot, 'models', modelType, noisyModelName), caffe.TEST)

	def run_absolute(self, niter):
		"""
		In order to run the test, the "label" layer should be appended to train_val.prototxt.
		"""
		base_avg_1, base_avg_5 = 0.0, 0.0
		deg_avg_1, deg_avg_5 = 0.0, 0.0
		noisy_avg_1, noisy_avg_5 = 0.0, 0.0
		for i in range(niter):
			base_tup = self.__predict__(self._baseNet)[0]
			deg_tup = self.__predict__(self._degNet)[0]
			noisy_tup = self.__predict__(self._noisyNet)[0]
			base_avg_1 += base_tup[0]
			base_avg_5 += base_tup[1]
			deg_avg_1 += deg_tup[0]
			deg_avg_5 += deg_tup[1]
			noisy_avg_1 += noisy_tup[0]
			noisy_avg_5 += noisy_tup[1]
		base_avg = tuple((base_avg_1/niter, base_avg_5/niter))
		deg_avg = tuple((deg_avg_1/niter, deg_avg_5/niter))
		noisy_avg = tuple((noisy_avg_1/niter, noisy_avg_5/niter))
		print '\nPerformance:\n\tbase: ' + str(base_avg) + '\n\tdeg: ' + str(deg_avg) + '\n\tnoisy: ' + str(noisy_avg)

	def run_relative(self, niter):
		deg_hit_1, deg_hit_5 = 0.0, 0.0
		noisy_hit_1, noisy_hit_5 = 0.0, 0.0
		for i in range(niter):
			base_line_1, base_line_5 = self.__predict__(self._baseNet)[1]
			deg_line_1, deg_line_5 = self.__predict__(self._degNet)[1]
			noisy_line_1, noisy_line_5 = self.__predict__(self._noisyNet)[1]
			# print base_line_1, base_line_5
			# print deg_line_1, deg_line_5
			# print noisy_line_1, noisy_line_5
			deg_hit_1 += self.__matchVec__(base_line_1, deg_line_1)
			deg_hit_5 += self.__matchVec__(base_line_5, deg_line_5)
			noisy_hit_1 += self.__matchVec__(base_line_1, noisy_line_1)
			noisy_hit_5 += self.__matchVec__(base_line_5, noisy_line_5)

		deg_hit_1 /= niter
		deg_hit_5 /= niter
		noisy_hit_1 /= niter
		noisy_hit_5 /= niter	

		print deg_hit_1, deg_hit_5, noisy_hit_1, noisy_hit_5
		data_1 = [deg_hit_1, deg_hit_5]
		data_2 = [noisy_hit_1, noisy_hit_5]
		ind = np.arange(2)
		width = 0.15
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.set_xlabel('Accuracy Types')
		ax.set_ylabel('Accuracy')

		rect1 = ax.bar(ind, data_1, width, color='r', label='top1')
		rect2 = ax.bar(ind + width, data_2, width, color='b', label='top5')
		ax.legend(loc='best', fontsize=10, fancybox=True, framealpha=0.4)
		ax.set_xticklabels(['deg', 'noisy'])
		ax.set_ylim(0,1)
		fig.show()
		sleep(20)
		plt.close()

	def __predict__(self, net):
		"""
		Predictions are in form of list of lists for both top1 and top5.
		"""
		res = net.forward()
		prob = res['prob']
		label = []
		for l in net.blobs['label'].data:
			label.append([l])
		
		# has checked nets all have same set of batches
		pred_top1, pred_top5 = [], []
		for p in prob:
			pred_top1.append([p.argmax()])
			pred_top5.append(nlargest(5, range(1000), key=lambda i:p[i]))
		top1 = self.__matchVec__(label, pred_top1)
		top5 = self.__matchVec__(label, pred_top5)
		return (top1, top5), (pred_top1, pred_top5)

	def __matchVec__(self, ndarray, B):
		top_acc = 0.0
		if len(ndarray) != len(B):
			raise AssertionError('Vector dimension mismatch')
		# top 1 case
		if len(B[0]) == 1:
			for i in range(len(ndarray)):
				if B[i] == ndarray[i]:
					top_acc += 1
		# top 5 case
		else:	
			for i in range(len(ndarray)):
				top_acc += len(set(ndarray[i]) & set(B[i])) / 5.0
		top_acc /= 10
		return top_acc

comp = ModelComparator('bvlc_googlenet.caffemodel',
	'deg_qual_pytrained_50%_10000.caffemodel', 'deg_qual_cppn_50%_10000.caffemodel')
comp.run_absolute(1)
#comp.run_relative(1)
