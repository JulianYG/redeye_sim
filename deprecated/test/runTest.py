import setup
import time
from testmodule import CaffeTester
from stats import getTopStats
import os

paths = setup.config()
caffeRoot = paths.getLoc('caffeRoot')
iroot = paths.getLoc('valDataPath')
writeOut = paths.getLoc('vecSavePath')
numBatchesPerRun = 1
numImagesInBatchFolder = 100

def old_run(caffeRoot, iroot, writeRoot, numOfBatch, numOfImg, degType, degModelName, 
	noisyModelName, init, step, inc, topN, thresh, debug=False, run=0):
	"""
	Runs the test in comparison, generate both human readable and raw data, as well as plots
	Input arguments: 
		init, step, Inc: initial value, number of steps and step size of degradation parameter.
		topN: N values and indices to choose from length 1000 class prediction vector.
		degModelName: the caffemodel trained by noisy image inputs
		noisyModelName: the caffemodel trained by noisy image inputs and noise injection 
		between layers
		thresh: the threshold percentage to filter out outliers in the class prediction vector.
		run: which run is this
	"""
	t_base = CaffeTester(caffeRoot, iroot, writeRoot, 'base_' + degType.lower() + '_model_' + str(numOfImg * numOfBatch), '', 
		numOfBatch, numOfImg, 'base', debug)

	t_base.degrade([degType], init, step, inc, topN, run, thresh)

	t_deg = CaffeTester(caffeRoot, iroot, writeRoot, 'deg_' + degType.lower() + '_model_' + str(numOfImg * numOfBatch), 
		degModelName, numOfBatch, numOfImg, 'deg', debug)
	t_deg.degrade([degType], init, step, inc, topN, run, thresh)

	t_noisy = CaffeTester(caffeRoot, iroot, writeRoot, 'noisy_' + degType.lower() + '_model_' + str(numOfImg * numOfBatch), 
		noisyModelName, numOfBatch, numOfImg, 'noisy', debug)
	t_noisy.degrade([degType], init, step, inc, topN, run, thresh)

def new_run(protoSuf, fileName, numOfBatch,optional_trainedfile=None,optional_modelfile=None,optional_imagemean=None):
	new = CaffeTester(caffeRoot, iroot, writeOut, 'logname', '', numBatchesPerRun, 
	#	numImagesInBatchFolder, modelType='googletab', protoSuffix=protoSuf)
		numImagesInBatchFolder, modelType='alextab', protoSuffix=protoSuf,optional_trainedfile=None,optional_modelfile=None,optional_imagemean=None)
	for i in range(numOfBatch):
		new.setImages(i);
		new.degrade(['clean'], 0, 1, 0, 0, i, 0.5, folderName=fileName)
	return getTopStats(os.path.join(writeOut, 'classvec_2013', fileName), 100, numOfBatch)

def goog_run(protoSuf, fileName, numOfBatch,optional_trainedfile=None,optional_modelfile=None,optional_imagemean=None):
	new = CaffeTester(caffeRoot, iroot, writeOut, 'logname', '', numBatchesPerRun, 
		numImagesInBatchFolder, modelType='googletab', protoSuffix=protoSuf,optional_trainedfile=None,optional_modelfile=None,optional_imagemean=None)
	for i in range(numOfBatch):
		new.setImages(i);
		new.degrade(['clean'], 0, 1, 0, 0, i, 0.5, folderName=fileName)
	return getTopStats(os.path.join(writeOut, 'classvec_2013', fileName), 100, numOfBatch)
	
#start1 = time.time()
# caffeRoot, iroot, dataExportPath, logName, modelName, 
# numOfBatch, numPerBatch, modelType='googlenet', dataSuffix='', debug=False):
# new1 = CaffeTester(caffeRoot, iroot, writeOut, 'logname', '', 
#  	numBatchesPerRun, numImagesInBatchFolder, modelType='googlenet')
# for i in range(500):
# 	new1.setImages(i);
# 	new1.degrade(['gaussian'], 10e-7, 10, 1, 5, i, 0.5, folderName='goog_gaussian_500')
# end1 = time.time()

# start2 = time.time()
# new2 = CaffeTester(caffeRoot, iroot, 'logname', '', 0, 
# 	numBatchesPerRun, numImagesInBatchFolder, modelType='alexnet')
# for i in range(500):
# 	new2.setImages(i);
# 	new2.degrade(['Gaussian'], 10e-7, 10, 1, 5, i, 0.5, folderName='alex_gaussian_500')
# end2 = time.time()

# start3 = time.time()
# new3 = CaffeTester(caffeRoot, iroot, writeOut, 'logname', '', 
# 	numBatchesPerRun, numImagesInBatchFolder, modelType='googlenet', protoSuffix='')
# for i in range(500):
# 	new3.setImages(i);
# 	new3.degrade(['quality'], 0, 19, 5, 5, i, 0.5, folderName='goog_qual_500')
# end3 = time.time()

#start3 = time.time()

# new3 = CaffeTester(caffeRoot, iroot, writeOut, 'logname', '', 
# 			numBatchesPerRun, numImagesInBatchFolder, modelType='googlenet', protoSuffix='g1q5')
# for i in range(500):
# 	new3.setImages(i);
# 	new3.degrade(['clean'], 0, 1, 5, 5, i, 0.5, folderName='goog_gaussian_500_g1q5')
#end3 = time.time()

def digit_run(protoSuf ='', fileName='digit', numOfBatch=1):
	new = CaffeTester(caffeRoot, iroot, writeOut, 'logname', '', numBatchesPerRun, 
		numImagesInBatchFolder, modelType='googlenet_digit', protoSuffix=protoSuf)
	for i in range(numOfBatch):
		new.setImages(i);
		new.degrade(['clean'], 0, 1, 0, 0, i, 0.5, folderName=fileName)
	return getTopStats(os.path.join(writeOut, 'classvec_2013', fileName), 100, numOfBatch)
# start4 = time.time()	
# new4 = CaffeTester(caffeRoot, iroot, 'logname', '', 0, 
# 	numBatchesPerRun, numImagesInBatchFolder, modelType='alexnet')
# for i in range(500):
# 	new4.setImages(i);
# 	new4.degrade(['quality'], 0, 19, 5, 5, i, 0.5, folderName='alex_qual_500')
# end4 = time.time()

# print "time lapse: " + str(end1 - start1) + '\n' + str(end2 - start2) + '\n' + str(end3 - start3) + '\n' + str(end4 - start4)
# time lapse: 1507.00775981
# 22177.335659
# 35129.268419
# 30288.0969162
