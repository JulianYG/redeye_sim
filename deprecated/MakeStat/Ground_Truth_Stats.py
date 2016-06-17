#Instructions for running the script:
#This script runs on terminal and takes in 6 arguments (5 if you don't include the file name)
#The arguments are as follows: 
	# 1) filename.py --> Ground_Truth_Stats.py
	# 2) num_batches --> Number of batches being analyzed (corresponds to largest first/third number of numpy array file names)
	# 3) max_deg_level --> The maximum degredation level (corresponds to largest second number of numpy array file names)
	# 4) img_per_batch --> number of images per batch
	# 5) deg_folder_name --> name of folder containing numpy arrays
	# 6) log_name --> name of files (before _0_0_0, etc)
	# 7) toprint --> Whether to print results; either true or false
# Example:
# miapolansky$ python Ground_Truth_Stats.py 500 0 100 goog_gaussian_500 logname True

import csv
import numpy as np
import glob
import os
from sys import argv
import argparse

script, num_batches, max_deg_level, img_per_batch, deg_folder_name, log_name, toprint = argv

gt = np.loadtxt("Ground_Truth_Caffe.txt")

def f(a,N):
	return np.argsort(a)[::-1][:N]

#top_1 = np.argmax(gt)

#max_deg_level = 1
#num_batches = 500
#img_per_batch = 100

match5 = np.zeros(int(max_deg_level)+1)
match1 = np.zeros(int(max_deg_level)+1)

#for deg_lvl in range(0,int(max_deg_level)+1): #+1?
for batch in range(0,int(num_batches)):
	for deg_lvl in range(0,int(max_deg_level)+1):
		#compare1 = np.load("/Users/miapolansky/caffe/caffe_exp/classvec_2013/"+deg_folder_name+"/"+log_name+"_0_"+str(deg_lvl)+"_"+str(batch)+".npy")
		compare1 = np.load("/home/bigbox/SSD/classvec_2013/"+deg_folder_name+"/"+log_name+"_0_"+str(deg_lvl)+"_"+str(batch)+".npy")
		#compare1 = np.load("/home/roblkw/SSD/classvec_2013/"+deg_folder_name+"/logname_0_"+str(deg_lvl)+"_"+str(batch)+".npy")
		# if batch>30:
		# 	compare1 = np.load("/Users/miapolansky/caffe/caffe_exp/classvec_2013/Gaussian/googlenet_"+str(batch)+"_"+str(deg_lvl)+"_"+str(batch)+".npy")
		# else:
		# 	compare1 = np.load("/Users/miapolansky/caffe/caffe_exp/classvec_2013/gaussian_30/logname_0_"+str(deg_lvl)+"_"+str(batch)+".npy")
		#print compare1.shape
		for img in range(0,int(img_per_batch)): 

			top_5 =  f(compare1[img,:],5)
			#print top_5
			top_1 = np.argmax(compare1[img,:])

			if gt[img+100*batch] == top_1:

				match1[deg_lvl] += 1

		 	for num in range(0,5): 
		 		
				if gt[img+100*batch] == top_5[num]:
					match5[deg_lvl] += 1

match1 = match1/float(int(img_per_batch)*int(num_batches))
match5 = match5/float(int(img_per_batch)*int(num_batches))

results = np.vstack((match1.T,match5.T))
results = results.T

#np.savetxt("/home/roblkw/Work/julian-work/caffe_exp/SaveStat/for_gnu_hit_rate_['"+deg_folder_name+"']_"+max_deg_level+".csv", results, delimiter=",")
#np.savetxt("/Users/miapolansky/caffe/caffe_exp/SaveStat/for_gnu_hit_rate_['"+deg_folder_name+"']_"+max_deg_level+".csv", results, delimiter=",")
np.savetxt("/home/bigbox/caffe_exp/SaveStat/for_gnu_hit_rate_['"+deg_folder_name+"']_"+max_deg_level+".csv", results, delimiter=",")

# np.savetxt("top_5_hit_rate_['"+deg_type+"']_"+max_deg_level+".csv", match5/float(500), delimiter=",")
# np.savetxt("top_1_hit_rate_['"+deg_type+"']_"+max_deg_level+".csv", match1/float(500), delimiter=",")

#print "top 5:", match5/float(int(img_per_batch)*int(num_batches)), '\n', "top 1:", match1/float(int(img_per_batch)*int(num_batches))

if toprint.lower() == "true":
	print "top 5:", match5, '\n', "top 1:", match1


