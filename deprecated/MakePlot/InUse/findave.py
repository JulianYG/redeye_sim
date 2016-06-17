import csv
import numpy as np
import glob
import os

def findave(deg_folder_name,levels,num_batches): #levels is total number of degrade levels 
	#N = len(glob.glob('stat/rates/*'))/6 #change root, divide by 2 if necessary 

	f_all = np.zeros(shape=(0,5)) #(0,number of columns)

	for num in range(0,num_batches):
         a = np.genfromtxt("/Users/miapolansky/caffe/caffe_exp/SaveStat/"+deg_folder_name+"/TPTNFPFN_batch_"+str(num)+"_deg_['"+deg_folder_name+"'].csv",skip_header=1,delimiter=",") #change root
    	 f_all = np.vstack((f_all,a)) 

	f_add = np.zeros(shape=(levels,4)) #depends on dimensions of excel

	for num2 in range(0,num_batches): 
		 f_add = np.add(f_add,f_all[levels*num2:levels*num2+levels,1:5]) #depends on dimensions of excel and where data is

	f_add = f_add*(float(1)/num_batches)

	np.savetxt("/Users/miapolansky/caffe/caffe_exp/SaveStat/TPTNFPFN/TPTNFPFN_ave_"+deg_folder_name+"_"+str(levels)+"_"+str(num_batches)+".csv", f_add, delimiter=",")

	print f_add
    

findave("goog_qual_500", 18,10)

