# Example:
# miapolansky$ python Ground_Truth_Stats.py 500 0 100 goog_gaussian_500 logname True

import csv
import numpy as np
import glob
import os
from sys import argv
import argparse


def createTop1Top5(num_batches, max_deg_level, deg_folder_name):

	log_name = "logname"
	toprint = "true"
	img_per_batch = 100

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
			compare1 = np.load("/Users/miapolansky/caffe_exp/classvec_2013/"+deg_folder_name+"/"+log_name+"_0_"+str(deg_lvl)+"_"+str(batch)+".npy")
			#compare1 = np.load("/home/bigbox/SSD/classvec_2013/"+deg_folder_name+"/"+log_name+"_0_"+str(deg_lvl)+"_"+str(batch)+".npy")
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

	return results

	#np.savetxt("/home/roblkw/Work/julian-work/caffe_exp/SaveStat/for_gnu_hit_rate_['"+deg_folder_name+"']_"+max_deg_level+".csv", results, delimiter=",")
	#np.savetxt("/Users/miapolansky/caffe/caffe_exp/SaveStat/for_gnu_hit_rate_['"+deg_folder_name+"']_"+max_deg_level+".csv", results, delimiter=",")
	#np.savetxt("/home/bigbox/caffe_exp/SaveStat/for_gnu_hit_rate_['"+deg_folder_name+"']_"+max_deg_level+".csv", results, delimiter=",")

	if toprint.lower() == "true":
		print "top 5:", match5, '\n', "top 1:", match1


s1 = np.zeros(shape=(0,2))
s2 = np.zeros(shape=(0,2))
s3 = np.zeros(shape=(0,2))

max_deg_level = 18


keep1 = createTop1Top5(500, max_deg_level, "goog_qual_500")
s1 =  np.vstack((s1,keep1)) 

keep2 = createTop1Top5(38, max_deg_level, "goog_qual_500_240")
s2 =  np.vstack((s2,keep2)) 

keep3 = createTop1Top5(49, max_deg_level, "goog_qual_500_120")
s3 =  np.vstack((s3,keep3)) 

data = np.hstack((s1,s2,s3))

print data

np.savetxt("/Users/miapolansky/caffe_exp/SaveStat/for_gnu_hit_rate_480_240_120", data, delimiter=",")
