import caffe
import os
import landscape
import datetime
from sys import argv
import argparse
import numpy as np

script, gintvl_val, qintvl_val, depth_level, numbatch, name = argv
 
depth_level = int(depth_level)
gintvl_val = int(gintvl_val)
qintvl_val = int(qintvl_val)
numbatch_val = int(numbatch)

def build_plane(numOfBatch=2, gintvl=3, qintvl=3, depth = 1):
	#k = open('/home/bigbox/caffe_exp/hyperplane.txt', 'w')
	terrain, g_std, SNRg, SNRq = landscape.create_map(gintvl=gintvl, qintvl=qintvl, 
		numOfBatch=numOfBatch, method='grid',depth=depth)

	for points in terrain:
			p = points.get_param()
			p_size = len(p)

	now = datetime.datetime.now()
	file_name = 'hypercube-' + str(p_size-1) + 'g1q-d'+str(depth_level)+'-b'+str(numOfBatch)+'-' +str(now.month) + "_" + str(now.day) +"_" + str(now.year) + str(name) + ".csv"
	#file_name = 'hypercube-' + str(p_size-1) + 'g1q-d'+str(depth_level)+'-b'+str(numOfBatch)+'-' +str(now.month) + "_" + str(now.day) +"_" + str(now.year) + str(name) + ".csv"
	k = open('../hypercubes/' + file_name,'w')
	plane = {}
	k.write('name,top5,top1,energy,g0,g1,q\n') #FIX HEADER 
	k.flush();
	i=0
	j=0
	m=0
	for points in terrain:
#		k = open('../hypercubes/' + file_name,'a')
		s = points.get_score()
		e = points.get_energy_loss()
		n = points.get_name()
		p = points.get_param()
		plane[n] = (s, e)
		k.write(str(s[0]) + ',' + str(s[0]) + ',' + str(s[1]) + ',' + str(e)+',')
		k.write(str(g_std[i]) + ',')
		k.write(','.join(str(x) for x in p) + ',')
		k.write(str(SNRg[j]) + ',')
		k.write(str(SNRq[m]) + '\n')
		k.flush();
		if i < gintvl - 1:
				i = i + 1
		else:
				i = 0	
		if j < gintvl - 1:
			j = j+1;
		else:
			j=0
		if m < gintvl - 1:
			m = m;
		else:
			m=m+1;

	k.close()
	return plane


build_plane(numOfBatch=numbatch_val, gintvl=gintvl_val, qintvl=qintvl_val, depth = depth_level);





