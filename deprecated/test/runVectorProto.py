import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import caffe
import os
import landscape
import datetime
from sys import argv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Top 1 and Top 5 prediction using given SNRg and SNRq vectors')
parser.add_argument('depth_level',metavar='depth_level',type=int,help='depth of the convolutional network')
parser.add_argument('numbatch_val',metavar='numbatch_val',type=int,help='number of batches')
parser.add_argument('name',metavar='name',help='name of file/experiment')
parser.add_argument('SNRg',metavar='SNRg',help='SNRg vector')
parser.add_argument('SNRq',metavar='SNRq',help='SNRq vector')
depth_level = parser.parse_args().depth_level
numbatch_val = parser.parse_args().numbatch_val
name = parser.parse_args().name
SNRg = list(map(float,parser.parse_args().SNRg.split(',')))
SNRq = list(map(float,parser.parse_args().SNRq.split(',')))

def build_fixed_plane(SNRg,SNRq,numOfBatch=2, depth = 1):
	terrain, g_std, SNRg, SNRq = landscape.create_map(method='vector',depth=depth,SNRg=SNRg,SNRq=SNRq,numOfBatch=numOfBatch)

	for points in terrain:
			p = points.get_param()
			p_size = len(p)

	now = datetime.datetime.now()
	file_name = 'SNR_Accuracy' + str(p_size-1) + 'g1q-d'+str(depth_level)+'-b'+str(numOfBatch)+'-' +str(now.month) + "_" + str(now.day) +"_" + str(now.year) + str(name) + ".csv"
	k = open('../hypercubes/' + file_name,'w')
	plane = {}
	k.write('name,top5,top1,energy,g0,g1,q\n') #FIX HEADER 
	k.flush();
	i=0
	j=0
	m=0
	for points in terrain:
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
		if i>=len(g_std)-1: break
		if j>=len(SNRg)-1: break
		if m>=len(SNRq)-1: break
		i=i+1
		j=j+1
		m=m+1

	k.close()
	return plane

build_fixed_plane(numOfBatch=numbatch_val,depth = depth_level,SNRg=SNRg,SNRq=SNRq);
