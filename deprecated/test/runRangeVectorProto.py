import caffe
import os
import landscape
import datetime
from sys import argv
import argparse
import numpy as np
import matplotlib.pyplot as plt

"""
Generate a CSV including accuracy of given SNRg and SNRq and a graph
Required arguments:
depth level
number of batch
name: included in the name of output file
SNRg: [lower bound],[upper bound],[number of data points needed]
SNRq: [lower bound],[upper bound],[number of data points needed]

Optional arguments:(default to caffe and imagenet 2012)
optional_trainedfile
optional_modelfile
optional_imagemean
"""

parser = argparse.ArgumentParser(description='Top 1 and Top 5 prediction using given SNRg and SNRq vectors')
parser.add_argument('depth_level',metavar='depth_level',type=int,help='depth of the convolutional network')
parser.add_argument('numbatch_val',metavar='numbatch_val',type=int,help='number of batches')
parser.add_argument('name',metavar='name',help='name of file/experiment')
parser.add_argument('SNRg',metavar='SNRg',help='SNRg range and step')
parser.add_argument('SNRq',metavar='SNRq',help='SNRq range and step')
parser.add_argument('optional_trainedfile',nargs='?',help='Optional trained file')
parser.add_argument('optional_modelfile',nargs='?',help='Optional model file')
parser.add_argument('Optional_imagemean',nargs='?',help='Optional image mean')
depth_level = parser.parse_args().depth_level
numbatch_val = parser.parse_args().numbatch_val
try:
	optional_trainedfile = parser.parse_args().optional_trainedfile
except AttributeError:
	optional_trainedfile = None
try:
	optional_modelfile = parser.parse_args().optional_modelfile
except AttributeError:
	optional_modelfile = None
try:
	optional_imagemean = parser.parse_args().optional_imagemean
except AttributeError:
	optional_imagemean = None
name = parser.parse_args().name
SNRg_v = list(map(float,parser.parse_args().SNRg.split(',')))
SNRq_v = list(map(float,parser.parse_args().SNRq.split(',')))

SNRg = np.linspace(SNRg_v[0],SNRg_v[1],SNRg_v[2])
SNRq = np.linspace(SNRq_v[0],SNRq_v[1],SNRq_v[2])

def build_fixed_plane(SNRg,SNRq,numOfBatch=2, depth = 1,optional_trainedfile=None,optional_modelfile=None,optional_imagemean=None):
	terrain, g_std, SNRg, SNRq = landscape.create_map(method='vector',depth=depth,SNRg=SNRg,SNRq=SNRq,numOfBatch=numOfBatch)

	for points in terrain:
			p = points.get_param()
			p_size = len(p)

	now = datetime.datetime.now()
	file_name = 'SNR_Accuracy' +str(now.month) + "_" + str(now.day) +"_" + str(now.year) + str(name)+".csv"
	k = open('../SNRgraph/' + file_name,'a')
	plane = {}
	#k.write('name,top5,top1,energy,g0,g1,q\n') #FIX HEADER 
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
		k.write(str(SNRq[m]) + ',' + str(s[0]) + ',' + str(s[1]) + '\n')
#		k.write(str(s[0]) + ',' + str(s[0]) + ',' + str(s[1]) + ',' + str(e)+',')
#		k.write(str(g_std[i]) + ',')
#		k.write(','.join(str(x) for x in p) + ',')
#		k.write(str(SNRg[j]) + ',')
#		k.write(str(SNRq[m]) + '\n')
		k.flush()
		if i>=len(g_std)-1: break
		if j>=len(SNRg)-1: break
		if m>=len(SNRq)-1: break
		i=i+1
		j=j+1
		m=m+1

	k.close()
	return plane

now = datetime.datetime.now()
file_name = 'SNR_Accuracy' +str(now.month) + "_" + str(now.day) +"_" + str(now.year) + str(name) + ".csv"
k = open('../SNRgraph/' + file_name,'a')
plane = {}
k.write('q,top5,top1\n') #FIX HEADER 
k.flush();
#g = SNRg[0]

for g in xrange(len(SNRq)):
#	for g in xrange(len(SNRg))
		SNRg_val = np.empty(1)
		SNRg_val.fill(SNRg[g])
		SNRq_val = np.empty(1)
		SNRq_val.fill(SNRq[g])
		build_fixed_plane(numOfBatch=numbatch_val,depth = depth_level,SNRg=SNRg_val,SNRq=SNRq_val,optional_trainedfile=optional_trainedfile,optional_modelfile=optional_modelfile,optional_imagemean=optional_imagemean);

data = np.genfromtxt('../SNRgraph/' + file_name,delimiter=',',skip_header=1,names=['SNR','top5','top1'])
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("Accuracy vs SNR")
ax1.set_xlabel('SNR')
ax1.set_ylabel('Accuracy')
ax1.plot(data['SNR'],data['top5'],color='r',label='top5')
ax1.plot(data['SNR'],data['top1'],color='b',label='top1')
fig.savefig('/home/newbox/caffe_exp/SNRgraph/SNRg_'+str(SNRg_v[0])+'_'+str(SNRg_v[1])+'_SNRq_'+str(SNRq_v[0])+'_'+str(SNRq_v[1])+'.png')
