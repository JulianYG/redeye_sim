import caffe
import os
import landscape

def build_plane(numOfBatch=1, gintvl=3, qintvl=3):
	k = open('/home/bigbox/caffe_exp/hyperplane.txt', 'w')
	plane = {}
	terrain = landscape.create_map(gintvl=gintvl, qintvl=qintvl, 
		numOfBatch=numOfBatch, method='grid')
	k.write('name\t(top5, top1)\tscore\tg0\tg1\tg2\tq\n')
	for points in terrain:
		s = points.get_score()
		e = points.get_energy_loss()
		n = points.get_name()
		p = points.get_param()
		plane[n] = (s, e)
		k.write(n + '\t' + str(s) + '\t' + str(e)+'\t')
		k.write(str(p[0]) + '\t' + str(p[1]) + '\t' + str(p[2]) + '\t' + str(p[3]) + '\n')
	k.close()
	return plane

build_plane(numOfBatch=20, gintvl=15, qintvl=30)
