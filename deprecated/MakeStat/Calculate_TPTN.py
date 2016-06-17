#Instructions for running the script:
#This script runs on terminal and takes in 6 arguments (5 if you don't include the file name)
#The arguments are as follows: 
	# 1) filename.py --> Calculate_TPTN.py
	# 2) num_batches --> Number of batches being analyzed (corresponds to largest first/third number of numpy array file names)
	# 3) max_deg_level --> The maximum degredation level (corresponds to largest second number of numpy array file names)
	# 4) img_per_batch --> number of images per batch
	# 5) deg_folder_name --> name of degredation forder
	# 6) log_name --> name of files in folder (before _0_0_0 etc)
	# 7) thresh --> at what value to threshold the label vectors (10e-5 recommended)
# Example:
# miapolansky$ python Calculate_TPTN.py 10 9 100 gaussian logname goog_qual 10e-5

import csv
import numpy as np
import glob
import os
from sys import argv
import argparse

script, num_batches, max_deg_level, img_per_batch, deg_folder_name, log_name, folder_to_save_in, thresh = argv

num_batches = int(num_batches)
max_deg_level = int(max_deg_level)
img_per_batch = int(img_per_batch)
thresh = float(thresh)

def threshold(thresh, vector):
	vector_out = np.zeros((img_per_batch,1000))
	for j in range(0,img_per_batch):
		for i in range(0, 1000):
			if vector[j,i] > thresh:
				vector_out[j,i] = 1
			else: 
				vector_out[j,i] = 0
	#print vector
	np.savetxt("testest1.csv", vector, delimiter=",")
	return vector_out


def i_threshold(thresh, vector):
	vector_out = np.zeros((img_per_batch,1000))
	for j in range(0,img_per_batch):
		for i in range(0, 1000):
			if vector[j,i] > thresh:
				vector_out[j,i] = 0
			else: 
				vector_out[j,i] = 1
	#print vector
	return vector_out

def TP(vector1, vector2):
	vec1 = threshold(thresh,vector1)
	vec2 = threshold(thresh,vector2)
	length = np.sum(vec1, axis = 1)
	store = np.zeros((img_per_batch,1))
	for i in range(0,img_per_batch):
		anded = np.logical_and(vec1[i,:],vec2[i,:])
		store[i,0] = np.sum(anded)

	store = np.transpose(store)
	TP = np.divide(store,length)
	TP = np.sum(TP)
	TP = TP/float(img_per_batch)
	return TP

def FP(vector1, vector2):
	vec1 = i_threshold(thresh,vector1)
	vec2 = threshold(thresh,vector2)
	length = np.sum(vec1, axis = 1)
	store = np.zeros((img_per_batch,1))
	for i in range(0,img_per_batch):
		anded = np.logical_and(vec1[i,:],vec2[i,:])
		store[i,0] = np.sum(anded)
	
	store = np.transpose(store)
	FP = np.divide(store,length)
	FP = np.sum(FP)
	FP = FP/float(img_per_batch)
	return FP

def FN(vector1, vector2):
	vec1 = threshold(thresh,vector1)
	vec2 = i_threshold(thresh,vector2)
	length = np.sum(vec1, axis = 1)
	store = np.zeros((img_per_batch,1))
	for i in range(0,img_per_batch):
		anded = np.logical_and(vec1[i,:],vec2[i,:])
		store[i,0] = np.sum(anded)
	
	store = np.transpose(store)
	FN = np.divide(store,length)
	FN = np.sum(FN)
	FN = FN/float(img_per_batch)
	return FN
	#print vec1, vec2

def TN(vector1, vector2):
	vec1 = i_threshold(thresh,vector1)
	vec2 = i_threshold(thresh,vector2)
	length = np.sum(vec1, axis = 1)
	store = np.zeros((img_per_batch,1))
	for i in range(0,img_per_batch):
		anded = np.logical_and(vec1[i,:],vec2[i,:])
		store[i,0] = np.sum(anded)
	
	store = np.transpose(store)
	TN = np.divide(store,length)
	TN = np.sum(TN)
	TN = TN/float(img_per_batch)
	return TN

# compare1 = np.load("/Users/miapolansky/caffe/caffe_exp/classvec_2013/Alexnet_Gaussian/a_gaussian_0_1_0.npy")
# compare2 = np.load("/Users/miapolansky/caffe/caffe_exp/classvec_2013/Alexnet_Gaussian/a_gaussian_0_2_0.npy")

# compare1 = np.load("/home/roblkw/Work/julian-work/caffe_exp/classvec_2013/Gaussian/googlenet_50_0_50.npy")
# compare2 = np.load("/home/roblkw/Work/julian-work/caffe_exp/classvec_2013/Gaussian/googlenet_50_9_50.npy")

# print compare1 - compare2

# TP(compare1,compare2)
# FP(compare1,compare2)
# FN(compare1,compare2)
# TN(compare1,compare2)

def RunTPTN(deg_folder_name,num_batches,max_deg_level):
	

	for batch in range(0,num_batches):

		file_save = np.zeros((max_deg_level+1,5))

		for deg_lvl in range(0,max_deg_level+1):
			compare1 = np.load("/Users/miapolansky/caffe/caffe_exp/classvec_2013/goog_clean/googlenet_"+str(batch)+"_0_"+str(batch)+".npy")
			compare2 = np.load("/Users/miapolansky/caffe/caffe_exp/classvec_2013/"+deg_folder_name+"/"+log_name+"_0_"+str(deg_lvl)+"_"+str(batch)+".npy")
			
			# compare1 = np.load("/home/roblkw/Work/julian-work/caffe_exp//classvec_2013/['clean']/googlenet_"+str(batch)+"_0_"+str(batch)+".npy")
			# compare2 = np.load("/home/roblkw/Work/julian-work/caffe_exp//classvec_2013/"+deg_type+"/googlenet_"+str(batch)+"_"+str(deg_lvl)+"_"+str(batch)+".npy")
			

			file_save[deg_lvl, 0] = deg_lvl

			file_save[deg_lvl, 1] = TP(compare1,compare2)
			file_save[deg_lvl, 2] = TN(compare1,compare2)
			file_save[deg_lvl, 3] = FP(compare1,compare2)
			file_save[deg_lvl, 4] = FN(compare1,compare2)

		np.savetxt("/Users/miapolansky/caffe/caffe_exp/SaveStat/"+deg_folder_name+"/TPTNFPFN_batch_"+str(batch)+"_deg_['"+deg_folder_name+"'].csv", file_save, header="deg_lvl, TP, TN, FP, FN", delimiter=",")


RunTPTN(deg_folder_name,num_batches,max_deg_level,folder_to_save_in)



