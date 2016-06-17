import csv
import numpy as np
import glob
import os
from sys import argv
import argparse

def getTopStats(foldername, numImagesPerBatch, batches):

	gt = np.loadtxt(os.path.join(os.path.dirname(os.getcwd()),
		'MakeStat',"Ground_Truth_Caffe.txt"))

	def f(a,N):
		return np.argsort(a)[::-1][:N]

	match5 = 0;
	match1 = 0;

	for batch in range(0,batches):
		compare1 = np.load(str(foldername)+"/logname_0_0_"+str(batch)+".npy")
		
		for img in range(0,int(numImagesPerBatch)): 

			top_5 = f(compare1[img,:],5)
			top_1 = np.argmax(compare1[img,:])

			if gt[img + 100*batch] == top_1:
				match1 += 1

		 	for num in range(0,5): 
				if gt[img + 100*batch] == top_5[num]:
					match5 += 1

	match1 = match1/float(int(numImagesPerBatch*batches))
	match5 = match5/float(int(numImagesPerBatch*batches))

	return match5, match1
