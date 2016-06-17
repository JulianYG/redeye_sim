import os
import sys
from collections import OrderedDict
sys.path.append(os.path.join(os.path.dirname(os.getcwd()),'Tools'))
import addNoiseDeploy as adder
from point import protoPoint
import numpy as np
import math

net_type = "goog_net"

# val_range = {"data": (6.63771, 65.3683), 
# 	"pool1/norm1": (34.5659, 33.6041), 
# 	"conv2/3x3_reduce": (21.4679, 34.9004), 
# 	"pool2/3x3_s2": (23.7876, 33.9384), 
# 	"inception_3a/3x3_reduce": (21.6651, 35.4639), 
# 	"inception_3a/5x5_reduce": (29.0692, 41.1394), 
# 	"inception_3a/pool": (51.2998, 40.5487), 
# 	"inception_3a/output": (22.1338, 42.3537)}

if (net_type=="goog_net"):
	val_range = {"data": (1.35859,57.3365), 
	"conv1/7x7_s2": (33.2551,79.99), 
	"pool1/3x3_s2": (80.9825,124.475), 
	"pool1/norm1": (36.5085,35.0324), 
	"conv2/3x3_reduce": (22.5373,37.5824), 
	"conv2/3x3": (10.1438,31.7746), 
	"conv2/norm2": (8.102,22.4789), 
	"pool2/3x3_s2": (26.8907,35.6255), 
	"pool2/3x3_s2_pool2/3x3_s2_0_split_0": (26.8907,35.6255), 
	"pool2/3x3_s2_pool2/3x3_s2_0_split_1": (26.8907,35.6255), 
	"pool2/3x3_s2_pool2/3x3_s2_0_split_2": (26.8907,35.6255), 
	"pool2/3x3_s2_pool2/3x3_s2_0_split_3": (26.8907,35.6255), 
	"inception_3a/1x1": (28.4413,46.6613), 
	"inception_3a/3x3_reduce": (20.0826,33.2641), 
	"inception_3a/3x3": (19.2969,40.8052), 
	"inception_3a/5x5_reduce": (36.9519,46.4113), 
	"inception_3a/5x5": (24.1693,52.7146), 
	"inception_3a/pool": (57.1692,39.8291), 
	"inception_3a/pool_proj": (16.048,33.654), 
	"inception_3a/output": (21.7859,43.4403), 
	"inception_3a/output_inception_3a/output_0_split_0": (21.7859,43.4403), 
	"inception_3a/output_inception_3a/output_0_split_1": (21.7859,43.4403), 
	"inception_3a/output_inception_3a/output_0_split_2": (21.7859,43.4403), 
	"inception_3a/output_inception_3a/output_0_split_3": (21.7859,43.4403), 
	"inception_3b/1x1": (6.14397,20.3674), 
	"inception_3b/3x3_reduce": (18.9365,35.232), 
	"inception_3b/3x3": (3.82471,18.9279), 
	"inception_3b/5x5_reduce": (26.3726,42.29), 
	"inception_3b/5x5": (4.32468,21.2872), 
	"inception_3b/pool": (59.3904,64.5851), 
	"inception_3b/pool_proj": (5.95173,22.9322), 
	"inception_3b/output": (4.82678,20.3859), 
	"pool3/3x3_s2": (16.4972,37.1505), 
	"pool3/3x3_s2_pool3/3x3_s2_0_split_0": (16.4972,37.1505), 
	"pool3/3x3_s2_pool3/3x3_s2_0_split_1": (16.4972,37.1505), 
	"pool3/3x3_s2_pool3/3x3_s2_0_split_2": (16.4972,37.1505), 
	"pool3/3x3_s2_pool3/3x3_s2_0_split_3": (16.4972,37.1505), 
	"inception_4a/1x1": (6.02054,21.9338), 
	"inception_4a/3x3_reduce": (26.1381,48.2152), 
	"inception_4a/3x3": (5.92894,31.1104), 
	"inception_4a/5x5_reduce": (39.8718,55.03), 
	"inception_4a/5x5": (7.27394,24.4478), 
	"inception_4a/pool": (43.8489,56.0495), 
	"inception_4a/pool_proj": (4.72613,17.841), 
	"inception_4a/output": (5.93904,25.8848), 
	"inception_4a/output_inception_4a/output_0_split_0": (5.93904,25.8848), 
	"inception_4a/output_inception_4a/output_0_split_1": (5.93904,25.8848), 
	"inception_4a/output_inception_4a/output_0_split_2": (5.93904,25.8848), 
	"inception_4a/output_inception_4a/output_0_split_3": (5.93904,25.8848), 
	"inception_4b/1x1": (10.417,21.7549), 
	"inception_4b/3x3_reduce": (12.4262,27.6068), 
	"inception_4b/3x3": (9.11738,23.6623), 
	"inception_4b/5x5_reduce": (23.2033,40.2695), 
	"inception_4b/5x5": (14.3745,35.946), 
	"inception_4b/pool": (21.5434,52.2785), 
	"inception_4b/pool_proj": (10.3854,25.0922), 
	"inception_4b/output": (10.3391,25.2142), 
	"inception_4b/output_inception_4b/output_0_split_0": (10.3391,25.2142), 
	"inception_4b/output_inception_4b/output_0_split_1": (10.3391,25.2142), 
	"inception_4b/output_inception_4b/output_0_split_2": (10.3391,25.2142), 
	"inception_4b/output_inception_4b/output_0_split_3": (10.3391,25.2142), 
	"inception_4c/1x1": (10.8527,22.9401), 
	"inception_4c/3x3_reduce": (12.7254,28.0336), 
	"inception_4c/3x3": (6.2007,20.1999), 
	"inception_4c/5x5_reduce": (18.1213,43.2494), 
	"inception_4c/5x5": (10.7771,33.1033), 
	"inception_4c/pool": (27.7548,42.8135), 
	"inception_4c/pool_proj": (5.43729,17.4683), 
	"inception_4c/output": (7.84033,22.7162), 
	"inception_4c/output_inception_4c/output_0_split_0": (7.84033,22.7162), 
	"inception_4c/output_inception_4c/output_0_split_1": (7.84033,22.7162), 
	"inception_4c/output_inception_4c/output_0_split_2": (7.84033,22.7162), 
	"inception_4c/output_inception_4c/output_0_split_3": (7.84033,22.7162), 
	"inception_4d/1x1": (3.36203,13.3551), 
	"inception_4d/3x3_reduce": (7.2005,21.6373), 
	"inception_4d/3x3": (2.58944,11.9204), 
	"inception_4d/5x5_reduce": (11.9598,23.811), 
	"inception_4d/5x5": (2.42323,10.972), 
	"inception_4d/pool": (21.2095,39.5745), 
	"inception_4d/pool_proj": (3.90064,20.731), 
	"inception_4d/output": (2.89212,13.4974), 
	"inception_4d/output_inception_4d/output_0_split_0": (2.89212,13.4974), 
	"inception_4d/output_inception_4d/output_0_split_1": (2.89212,13.4974), 
	"inception_4d/output_inception_4d/output_0_split_2": (2.89212,13.4974), 
	"inception_4d/output_inception_4d/output_0_split_3": (2.89212,13.4974), 
	"inception_4e/1x1": (1.18363,6.52526), 
	"inception_4e/3x3_reduce": (4.19119,8.75011), 
	"inception_4e/3x3": (0.631327,3.8169), 
	"inception_4e/5x5_reduce": (9.07047,15.721), 
	"inception_4e/5x5": (1.1345,6.04016), 
	"inception_4e/pool": (8.80436,24.7152), 
	"inception_4e/pool_proj": (1.98087,8.51487), 
	"inception_4e/output": (1.0863,5.97449), 
	"pool4/3x3_s2": (3.1512,10.7446), 
	"pool4/3x3_s2_pool4/3x3_s2_0_split_0": (3.1512,10.7446), 
	"pool4/3x3_s2_pool4/3x3_s2_0_split_1": (3.1512,10.7446), 
	"pool4/3x3_s2_pool4/3x3_s2_0_split_2": (3.1512,10.7446), 
	"pool4/3x3_s2_pool4/3x3_s2_0_split_3": (3.1512,10.7446), 
	"inception_5a/1x1": (2.15512,6.64944), 
	"inception_5a/3x3_reduce": (4.01401,8.57983), 
	"inception_5a/3x3": (0.93758,3.60327), 
	"inception_5a/5x5_reduce": (6.53003,13.3659), 
	"inception_5a/5x5": (1.66267,5.14779), 
	"inception_5a/pool": (8.64457,17.7683), 
	"inception_5a/pool_proj": (2.19886,7.93829), 
	"inception_5a/output": (1.6178,5.71785), 
	"inception_5a/output_inception_5a/output_0_split_0": (1.6178,5.71785), 
	"inception_5a/output_inception_5a/output_0_split_1": (1.6178,5.71785), 
	"inception_5a/output_inception_5a/output_0_split_2": (1.6178,5.71785), 
	"inception_5a/output_inception_5a/output_0_split_3": (1.6178,5.71785), 
	"inception_5b/1x1": (0.535557,2.03443), 
	"inception_5b/3x3_reduce": (0.926386,3.34919), 
	"inception_5b/3x3": (0.70965,2.29487), 
	"inception_5b/5x5_reduce": (1.47796,3.8881), 
	"inception_5b/5x5": (0.515573,1.63341), 
	"inception_5b/pool": (4.35053,9.57875), 
	"inception_5b/pool_proj": (0.461159,2.20305), 
	"inception_5b/output": (0.589045,2.11551), 
	"pool5/7x7_s1": (0.589045,1.0716), 
	"loss3/classifier": (-0.00156424,2.44757), 
	"prob": (0.001,0.0181702), 
	}

	level_control = None
	control_param = {1: 5e-3, 2: 5e-3, 3: 5e-3, 4: 5e-3, 5: 5e-3, 6: 5e-3}

else:
	val_range = {"data": (1.3554,57.3337), 
	"conv1": (23.0153,58.2871), 
	"norm1": (15.5033,29.1486), 
	"pool1": (40.0731,37.6784), 
	"conv2": (7.43466,25.2039), 
	"norm2": (6.30801,19.0461), 
	"pool2": (17.3211,29.4393), 
	"conv3": (11.7791,25.573), 
	"conv4": (7.23071,16.875), 
	"conv5": (1.73583,9.62729), 
	"pool5": (6.82439,20.0219), 
	"fc6": (1.80326,6.02226), 
	"fc7": (0.406519,1.46838), 
	"fc8": (-6.37913e-05,3.1094), 
	"prob": (0.001,0.0126779), 
	}

	level_control = None
	control_param = {1: 5e-3, 2: 5e-3, 3: 5e-3, 4: 5e-3, 5: 5e-3, 6: 5e-3}


def scale_param(lname, val):
	"""
	lname indicates the name of the layer of parameter;
	"""
	#scale_norm = val_range[lname][0] + val_range[lname][1] * 4
	scale_norm = val_range[lname][1] * 4
	return val * scale_norm

def scale_val(lname):
	"""
	lname indicates the name of the layer of parameter;
	"""
	#scale_norm = val_range[lname][0] + val_range[lname][1] * 4
	return val_range[lname][1] * 4
	
def _is_neighbor(p1, p2):
	cnt = 0
	for i in range(len(p1.get_param()) - 2):
		if p1.get_param()[i] != p2.get_param()[i]:
			cnt += 1
			if cnt > 1:
				return False
	# handle case for min_u and max_u separately
	if p1.get_param()[-1] != p2.get_param()[-1] or p1.get_param()[-2] != p2.get_param()[-2]:
		cnt += 1
	if cnt == 1:
		return True
	else:
		return False

def create_map(method='grid',optional_trainedfile=None,optional_modelfile=None,optional_imagemean=None,**kwargs):
	"""
	The map is represented by a dictionary that has form of {point: [neighbors]}.
	Each point is a representation of a prototxt file with preset noise parameters.
	Neighbors are defined as the points with only one layer's param changed.
	Inputs: 
		intvl: the discreteness of the data points.
		range: indicates the start and end point of param interval.
	"""
	pmap = []

	if method == 'vector':
		SNRg=kwargs['SNRg']
		SNRq=kwargs['SNRq']
		if (net_type=="goog_net"):
			name_list, param_list, u_param, g_std = _vector_search_goog(depth = kwargs['depth'],SNRg=kwargs['SNRg'],SNRq=kwargs['SNRq'])
		else:
			name_list, param_list, u_param, g_std = _vector_search_alex(depth = kwargs['depth'],SNRg=kwargs['SNRg'],SNRq=kwargs['SNRq'])
	elif method == 'grid':
		if (net_type=="goog_net"):
			name_list, param_list, u_param, g_std, SNRg, SNRq = _grid_search_goog(q_intvl=kwargs['qintvl'],
				g_intvl=kwargs['gintvl'], g_range=(2e-4, .15), q_range=(0.0156, 0.5), depth = kwargs['depth'])
		else:
			name_list, param_list, u_param, g_std, SNRg, SNRq = _grid_search_alex(q_intvl=kwargs['qintvl'],
				g_intvl=kwargs['gintvl'], g_range=(2e-4, .15), q_range=(0.0156, 0.5), depth = kwargs['depth'])
	else:
		name_list, param_list, u_param, g_std, SNRg, SNRq = [], [], (0,0)
	for i in range(len(name_list)):
		p = protoPoint(param_list[i], u_param[i], name_list[i], kwargs['numOfBatch'],kwargs['depth'],optional_trainedfile,optional_modelfile,optional_imagemean)		
		pmap.append(p)
	return pmap, g_std, SNRg, SNRq

def _vector_search_alex(SNRg,SNRq,depth = 1):
	name_list = []
	param_list = []
	u_param = []
	
	g_std = np.zeros(len(SNRg));
	for i in range(len(SNRg)):
		expon = (-SNRg[i]+10*math.log10(.5))/10;
		expon2 = math.pow(10, expon);
		g_std[i] = math.sqrt(expon2);
		
	q_std = np.zeros(len(SNRq));
	for i in range(len(SNRq)):
		expon = (-SNRq[i]+10*math.log10(.5))/10;
		expon2 = math.pow(10, expon);
		q_std[i] = math.sqrt(expon2);

	g_std= np.array(g_std);
	q_std= np.array(q_std);

	######UNCOMMENT BELOW FOR MESH GRID (NEED TO CHANGE LAYERS FOR ALEXNET)
	# n0v = g_std*scale_val("data");
	# n1v = g_std*scale_val("conv1/7x7_s2");
	# n2v = g_std*scale_val("pool1/norm1");
	# n3v = g_std*scale_val("conv2/3x3_reduce");
	# n4v = g_std*scale_val("conv2/3x3");
	# n5v = g_std*scale_val("conv2/norm2");
	# u1v = q_std*scale_val("pool1/3x3_s2");
	# u2v = q_std*scale_val("conv2/3x3_reduce");
	# u3v = q_std*scale_val("pool2/3x3_s2");
	# #####

	g_std_f = g_std
	q_std_f = q_std


	n0v = g_std_f*scale_val("data");
	n1v = g_std_f*scale_val("conv1");
	n2v = g_std_f*scale_val("norm1");
	
	n3v = g_std_f*scale_val("conv2");
	n4v = g_std_f*scale_val("norm2");

	n5v = g_std_f*scale_val("conv3");

	n6v = g_std_f*scale_val("conv4");

	n7v = g_std_f*scale_val("conv5");


	u1v = q_std_f*scale_val("pool1");
	u2v = q_std_f*scale_val("pool2");
	u3v = q_std_f*scale_val("conv3");
	u4v = q_std_f*scale_val("conv4");
	u5v = q_std_f*scale_val("pool5");
	#

	######UNCOMMENT BELOW FOR MESH GRID
	# if (depth == 1):
	# 	n0,n1,u1 = np.meshgrid(n0v,n1v,u1v)
	# 	n0=n0.flatten();
	# 	n1=n1.flatten();
	# 	u1=u1.flatten();
	# 	for i in range(len(n0)):
	# 		param_list.append([n0[i], n1[i]])
	# 		name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_q2_" + str(u1[i]))
	# 		u_param.append(u1[i]);
	#####

	if (depth == 1):
		n0=n0v
		n1=n1v
		n2=n2v
		u1=u1v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" + str(n2[i]) +"_q2_" + str(u1[i]))
			u_param.append(u1[i]);

	######UNCOMMENT BELOW FOR MESH GRID
	# if (depth == 2):
	# 	n0,n1,n2,n3,u2 = np.meshgrid(n0v,n1v,n2v,n3v,u2v)
	# 	n0=n0.flatten();
	# 	n1=n1.flatten();
	# 	n2=n2.flatten();
	# 	n3=n3.flatten();
	# 	u2=u2.flatten();
	# 	for i in range(len(n0)):
	# 		param_list.append([n0[i], n1[i], n2[i], n3[i]])
	# 		name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i]) + "_q2_" + str(u2[i]))
	# 		u_param.append(u2[i]);
	######

	if (depth == 2):
		n0=n0v
		n1=n1v
		n2=n2v
		n3=n3v
		n4=n4v

		u2=u2v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i])
			+ "_g4_" + str(n4[i]) + "_q_" + str(u2[i]))
			u_param.append(u2[i]);

	######UNCOMMENT BELOW FOR MESH GRID		
	# if (depth == 3):
	# 	n0,n1,n2,n3,n4,n5,u3 = np.meshgrid(n0v,n1v,n2v,n3v,n4v,n5v,u3v)
	# 	n0=n0.flatten();
	# 	n1=n1.flatten();
	# 	n2=n2.flatten();
	# 	n3=n3.flatten();
	# 	n4=n4.flatten();
	# 	n5=n5.flatten();
	# 	u3=u3.flatten();
	# 	for i in range(len(n0)):
	# 		param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i], n5[i]])
	# 		name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i]) + "_g4_" 
	# 			+ str(n4[i]) + "_g5_" + str(n5[i]) + "_q2_" + str(u3[i]))
	# 		u_param.append(u3[i])
	#######

	if (depth == 3):
		n0=n0v
		n1=n1v
		n2=n2v
		n3=n3v
		n4=n4v
		n5=n5v

		u3=u3v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i], n5[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i]) + "_g4_" 
				+ str(n4[i]) + "_g5_" + str(n5[i]) + "_q_" + str(u3[i]))
			u_param.append(u3[i]);

	
	if (depth == 4):
		n0=n0v
		n1=n1v
		n2=n2v
		n3=n3v
		n4=n4v
		n5=n5v
		n6=n6v

		u4=u4v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i], n5[i], n6[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i]) + "_g4_" 
				+ str(n4[i]) + "_g5_" + str(n5[i]) + "_g6_" + str(n6[i]) + "_q_" + str(u4[i]))
			u_param.append(u4[i]);
	
	if (depth == 5):
		n0=n0v
		n1=n1v
		n2=n2v
		n3=n3v
		n4=n4v
		n5=n5v
		n6=n6v
		n7=n7v

		u5=u5v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i],n5[i],n6[i],n7[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i]) + "_g4_" 
				+ str(n4[i]) + "_g5_" + str(n5[i])+ "_g6_" + str(n6[i])+ "_g7_" + str(n7[i])+ "_q_" + str(u5[i]))
			u_param.append(u5[i]);

	SNRg = SNR;
	SNRq = SNR;

	
	return name_list, param_list, u_param, g_std

def _vector_search_goog(SNRg,SNRq,depth = 1):
	
	#1D SNR
	name_list = []
	param_list = []
	u_param = []
	
	g_std = np.zeros(len(SNRg));
	for i in range(len(SNRg)):
		expon = (-SNRg[i]+10*math.log10(.5))/10;
		expon2 = math.pow(10, expon);
		g_std[i] = math.sqrt(expon2);
		
	q_std = np.zeros(len(SNRq));
	for i in range(len(SNRq)):
		expon = (-SNRq[i]+10*math.log10(.5))/10;
		expon2 = math.pow(10, expon);
		q_std[i] = math.sqrt(expon2);

	g_std= np.array(g_std);
	q_std= np.array(q_std);

	g_std_f = g_std
	q_std_f = q_std
#<<<<<<< HEAD
	# ## /1D

	# ##FOR NON "SNR"
	g_std_v,q_std_v = np.meshgrid(g_std,q_std)
	g_std_f = g_std_v.flatten()
	q_std_f = q_std_v.flatten()

	## 10/05/2015 fix by Cary: SNR should not be equal to std
#	SNRg = g_std;
#	SNRq = q_std;
	# ##
#=======
	## /1D

	# ##FOR NON "SNR"
	# g_std_v,q_std_v = np.meshgrid(g_std,q_std)
	# g_std_f = g_std_v.flatten()
	# q_std_f = q_std_v.flatten()
	# SNRg = g_std;
	# SNRq = q_std;
	# # ##
#>>>>>>> b795f5b572296d8789a14372ba97a49b67633176

	n0v = g_std_f*scale_val("data");
	n1v = g_std_f*scale_val("conv1/7x7_s2");
	n2v = g_std_f*scale_val("pool1/norm1");


	n3v = g_std_f*scale_val("conv2/3x3_reduce");
	n4v = g_std_f*scale_val("conv2/3x3");
	n5v = g_std_f*scale_val("conv2/norm2");
	
	n6v = g_std_f*scale_val("inception_3a/1x1");
	n7v = g_std_f*scale_val("inception_3a/3x3_reduce");
	n8v = g_std_f*scale_val("inception_3a/3x3");
	n9v = g_std_f*scale_val("inception_3a/5x5_reduce");
	n10v = g_std_f*scale_val("inception_3a/5x5");
	n11v = g_std_f*scale_val("inception_3a/pool_proj");
	n12v = g_std_f*scale_val("inception_3b/1x1");
	n13v = g_std_f*scale_val("inception_3b/3x3_reduce");
	n14v = g_std_f*scale_val("inception_3b/3x3");
	n15v = g_std_f*scale_val("inception_3b/5x5_reduce");
	n16v = g_std_f*scale_val("inception_3b/5x5");
	n17v = g_std_f*scale_val("inception_3b/pool_proj");

	n18v = g_std_f*scale_val("inception_4a/1x1");
	n19v = g_std_f*scale_val("inception_4a/3x3_reduce");
	n20v = g_std_f*scale_val("inception_4a/3x3");
	n21v = g_std_f*scale_val("inception_4a/5x5_reduce");
	n22v = g_std_f*scale_val("inception_4a/5x5");
	n23v = g_std_f*scale_val("inception_4a/pool_proj");
	n24v = g_std_f*scale_val("inception_4b/1x1");
	n25v = g_std_f*scale_val("inception_4b/3x3_reduce");
	n26v = g_std_f*scale_val("inception_4b/3x3");
	n27v = g_std_f*scale_val("inception_4b/5x5_reduce");
	n28v = g_std_f*scale_val("inception_4b/5x5");
	n29v = g_std_f*scale_val("inception_4b/pool_proj");

	u1v = q_std_f*scale_val("pool1/norm1");
	u2v = q_std_f*scale_val("pool2/3x3_s2");
	un3v = q_std_f*scale_val("inception_3a/output");
	u3v = q_std_f*scale_val("pool3/3x3_s2");
	u4v = q_std_f*scale_val("inception_4a/output");
	u5v = q_std_f*scale_val("inception_4b/output");
	#

	######UNCOMMENT BELOW FOR MESH GRID
	# if (depth == 1):
	# 	n0,n1,u1 = np.meshgrid(n0v,n1v,u1v)
	# 	n0=n0.flatten();
	# 	n1=n1.flatten();
	# 	u1=u1.flatten();
	# 	for i in range(len(n0)):
	# 		param_list.append([n0[i], n1[i]])
	# 		name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_q2_" + str(u1[i]))
	# 		u_param.append(u1[i]);
	#####

	if (depth == 1):
		n0=n0v
		n1=n1v
		n2=n2v
		u1=u1v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" + str(n2[i]) + "_q2_" + str(u1[i]))
			u_param.append(u1[i]);

	######UNCOMMENT BELOW FOR MESH GRID
	# if (depth == 2):
	# 	n0,n1,n2,n3,u2 = np.meshgrid(n0v,n1v,n2v,n3v,u2v)
	# 	n0=n0.flatten();
	# 	n1=n1.flatten();
	# 	n2=n2.flatten();
	# 	n3=n3.flatten();
	# 	u2=u2.flatten();
	# 	for i in range(len(n0)):
	# 		param_list.append([n0[i], n1[i], n2[i], n3[i]])
	# 		name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i]) + "_q2_" + str(u2[i]))
	# 		u_param.append(u2[i]);
	######

	if (depth == 2):
		n0=n0v
		n1=n1v
		n2=n2v
		n3=n3v
		n4=n4v
		n5=n5v
		u2=u2v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i], n5[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i])+ "_g5_" + str(n5[i]) + "_q_" + str(u2[i]))
			u_param.append(u2[i]);

	######UNCOMMENT BELOW FOR MESH GRID		
	# if (depth == 3):
	# 	n0,n1,n2,n3,n4,n5,u3 = np.meshgrid(n0v,n1v,n2v,n3v,n4v,n5v,u3v)
	# 	n0=n0.flatten();
	# 	n1=n1.flatten();
	# 	n2=n2.flatten();
	# 	n3=n3.flatten();
	# 	n4=n4.flatten();
	# 	n5=n5.flatten();
	# 	u3=u3.flatten();
	# 	for i in range(len(n0)):
	# 		param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i], n5[i]])
	# 		name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i]) + "_g4_" 
	# 			+ str(n4[i]) + "_g5_" + str(n5[i]) + "_q2_" + str(u3[i]))
	# 		u_param.append(u3[i])
	#######

	if (depth == -3):
		n0=n0v
		n1=n1v
		n2=n2v
		n3=n3v
		n4=n4v
		n5=n5v
		n6=n6v
		n7=n7v
		n8=n8v
		n9=n9v
		n10=n10v
		n11=n11v
		u3=un3v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i], n5[i],n6[i],n7[i],n8[i],
				n9[i],n10[i],n11[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i]) + "_q_" + str(u3[i]))
			u_param.append(u3[i]);


	if (depth == 3):
		n0=n0v
		n1=n1v
		n2=n2v
		n3=n3v
		n4=n4v
		n5=n5v
		n6=n6v
		n7=n7v
		n8=n8v
		n9=n9v
		n10=n10v
		n11=n11v
		n12=n12v
		n13=n13v
		n14=n14v
		n15=n15v
		n16=n16v
		n17=n17v
		u3=u3v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i], n5[i],n6[i],n7[i],n8[i],
				n9[i],n10[i],n11[i],n12[i],n13[i],n12[i],n15[i],n16[i],n17[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i]) + "_q_" + str(u3[i]))
			u_param.append(u3[i]);

	if (depth == 4):
		n0=n0v
		n1=n1v
		n2=n2v
		n3=n3v
		n4=n4v
		n5=n5v
		n6=n6v
		n7=n7v
		n8=n8v
		n9=n9v
		n10=n10v
		n11=n11v
		n12=n12v
		n13=n13v
		n14=n14v
		n15=n15v
		n16=n16v
		n17=n17v
		n18=n18v
		n19=n19v
		n20=n20v
		n21=n21v
		n22=n22v
		n23=n23v
		u4=u4v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i], n5[i],n6[i],n7[i],n8[i],
				n9[i],n10[i],n11[i],n12[i],n13[i],n12[i],n15[i],n16[i],n17[i],n18[i],n19[i],n20[i],n21[i],n22[i],n23[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_q_" + str(u4[i]))
			u_param.append(u4[i]);	

	if (depth == 5):
		n0=n0v
		n1=n1v
		n2=n2v
		n3=n3v
		n4=n4v
		n5=n5v
		n6=n6v
		n7=n7v
		n8=n8v
		n9=n9v
		n10=n10v
		n11=n11v
		n12=n12v
		n13=n13v
		n14=n14v
		n15=n15v
		n16=n16v
		n17=n17v
		n18=n18v
		n19=n19v
		n20=n20v
		n21=n21v
		n22=n22v
		n23=n23v
		n24=n24v
		n25=n25v
		n26=n26v
		n27=n27v
		n28=n28v
		n29=n29v
		u5=u5v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i], n5[i],n6[i],n7[i],n8[i],
				n9[i],n10[i],n11[i],n12[i],n13[i],n12[i],n15[i],n16[i],n17[i],n18[i],n19[i],n20[i],n21[i],n22[i],n23[i],n24[i],
				n25[i],n26[i],n27[i],n28[i],n29[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_q_" + str(u5[i]))
			u_param.append(u5[i]);					

	
	return name_list, param_list, u_param, g_std

############################################################################################################################################################

def _grid_search_alex(g_intvl=1, g_range=(5e-4, 5e-2), q_intvl=1, q_range=(3e-3, 5e-2), depth = 1):
	# g_inc = (g_range[1] - g_range[0]) / g_intvl
	# q_inc = (q_range[1] - q_range[0]) / q_intvl
	# g_std, q_std = [], []
	# for i in range(g_intvl):
	# 	g_std.append(g_range[0] + i * g_inc)
	# for j in range(q_intvl):
	# 	q_std.append(q_range[0] + j * q_inc)
	name_list = []
	param_list = []
	u_param = []
	
	if q_intvl ==1:
		SNR_min=38
	else:
		SNR_min=40;
	expon_minq = (-SNR_min+10*math.log10(1.5))/10;
	expon2_minq = math.pow(10, expon_minq);
	q_min = math.sqrt(expon2_minq)
	
	expon_ming = (-SNR_min+10*math.log10(.5))/10;
	expon2_ming = math.pow(10, expon_ming);
	g_min = math.sqrt(expon2_ming)
 

	if q_intvl == 1:
		SNR = np.linspace(10,60, num=g_intvl);
		g_std = np.zeros(len(SNR));
		for i in range(len(SNR)):
			expon = (-SNR[i]+10*math.log10(.5))/10;
			expon2 = math.pow(10, expon);
			g_std[i] = math.sqrt(expon2);
		
		q_std = []
		for j in range(len(g_std)):
			q_std.append(q_min)

	else:
		SNR = np.linspace(10,60, num=q_intvl);
		q_std = np.zeros(len(SNR));
		for i in range(len(SNR)):
			expon = (-SNR[i]+10*math.log10(1.5))/10;
			expon2 = math.pow(10, expon);
			q_std[i] = math.sqrt(expon2);

		g_std = []
		for j in range(len(q_std)):
			g_std.append(g_min)


	# g_inc = (g_range[1] - g_range[0]) / g_intvl
	# q_inc = (q_range[1] - q_range[0]) / q_intvl
	# g_std, q_std = [], []
	# for i in range(g_intvl):
	# 	g_std.append(g_range[0] + i * g_inc)
	# for j in range(q_intvl):
	# 	q_std.append(q_range[0] + j * q_inc)
	# name_list = []
	# param_list = []
	# u_param = []

	g_std= np.array(g_std);
	q_std= np.array(q_std);

	######UNCOMMENT BELOW FOR MESH GRID (NEED TO CHANGE LAYERS FOR ALEXNET)
	# n0v = g_std*scale_val("data");
	# n1v = g_std*scale_val("conv1/7x7_s2");
	# n2v = g_std*scale_val("pool1/norm1");
	# n3v = g_std*scale_val("conv2/3x3_reduce");
	# n4v = g_std*scale_val("conv2/3x3");
	# n5v = g_std*scale_val("conv2/norm2");
	# u1v = q_std*scale_val("pool1/3x3_s2");
	# u2v = q_std*scale_val("conv2/3x3_reduce");
	# u3v = q_std*scale_val("pool2/3x3_s2");
	# #####


	#g_std_v,q_std_v = np.meshgrid(g_std,q_std)
	# g_std_f = g_std_v.flatten()
	# q_std_f = q_std_v.flatten()
	g_std_f = g_std
	q_std_f = q_std


	n0v = g_std_f*scale_val("data");
	n1v = g_std_f*scale_val("conv1");
	n2v = g_std_f*scale_val("norm1");
	
	n3v = g_std_f*scale_val("conv2");
	n4v = g_std_f*scale_val("norm2");

	n5v = g_std_f*scale_val("conv3");

	n6v = g_std_f*scale_val("conv4");

	n7v = g_std_f*scale_val("conv5");


	u1v = q_std_f*scale_val("pool1");
	u2v = q_std_f*scale_val("pool2");
	u3v = q_std_f*scale_val("conv3");
	u4v = q_std_f*scale_val("conv4");
	u5v = q_std_f*scale_val("pool5");
	#

	######UNCOMMENT BELOW FOR MESH GRID
	# if (depth == 1):
	# 	n0,n1,u1 = np.meshgrid(n0v,n1v,u1v)
	# 	n0=n0.flatten();
	# 	n1=n1.flatten();
	# 	u1=u1.flatten();
	# 	for i in range(len(n0)):
	# 		param_list.append([n0[i], n1[i]])
	# 		name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_q2_" + str(u1[i]))
	# 		u_param.append(u1[i]);
	#####

	if (depth == 1):
		n0=n0v
		n1=n1v
		n2=n2v
		u1=u1v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" + str(n2[i]) +"_q2_" + str(u1[i]))
			u_param.append(u1[i]);

	######UNCOMMENT BELOW FOR MESH GRID
	# if (depth == 2):
	# 	n0,n1,n2,n3,u2 = np.meshgrid(n0v,n1v,n2v,n3v,u2v)
	# 	n0=n0.flatten();
	# 	n1=n1.flatten();
	# 	n2=n2.flatten();
	# 	n3=n3.flatten();
	# 	u2=u2.flatten();
	# 	for i in range(len(n0)):
	# 		param_list.append([n0[i], n1[i], n2[i], n3[i]])
	# 		name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i]) + "_q2_" + str(u2[i]))
	# 		u_param.append(u2[i]);
	######

	if (depth == 2):
		n0=n0v
		n1=n1v
		n2=n2v
		n3=n3v
		n4=n4v

		u2=u2v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i])
			+ "_g4_" + str(n4[i]) + "_q_" + str(u2[i]))
			u_param.append(u2[i]);

	######UNCOMMENT BELOW FOR MESH GRID		
	# if (depth == 3):
	# 	n0,n1,n2,n3,n4,n5,u3 = np.meshgrid(n0v,n1v,n2v,n3v,n4v,n5v,u3v)
	# 	n0=n0.flatten();
	# 	n1=n1.flatten();
	# 	n2=n2.flatten();
	# 	n3=n3.flatten();
	# 	n4=n4.flatten();
	# 	n5=n5.flatten();
	# 	u3=u3.flatten();
	# 	for i in range(len(n0)):
	# 		param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i], n5[i]])
	# 		name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i]) + "_g4_" 
	# 			+ str(n4[i]) + "_g5_" + str(n5[i]) + "_q2_" + str(u3[i]))
	# 		u_param.append(u3[i])
	#######

	if (depth == 3):
		n0=n0v
		n1=n1v
		n2=n2v
		n3=n3v
		n4=n4v
		n5=n5v

		u3=u3v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i], n5[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i]) + "_g4_" 
				+ str(n4[i]) + "_g5_" + str(n5[i]) + "_q_" + str(u3[i]))
			u_param.append(u3[i]);

	
	if (depth == 4):
		n0=n0v
		n1=n1v
		n2=n2v
		n3=n3v
		n4=n4v
		n5=n5v
		n6=n6v

		u4=u4v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i], n5[i], n6[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i]) + "_g4_" 
				+ str(n4[i]) + "_g5_" + str(n5[i]) + "_g6_" + str(n6[i]) + "_q_" + str(u4[i]))
			u_param.append(u4[i]);
	
	if (depth == 5):
		n0=n0v
		n1=n1v
		n2=n2v
		n3=n3v
		n4=n4v
		n5=n5v
		n6=n6v
		n7=n7v

		u5=u5v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i],n5[i],n6[i],n7[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i]) + "_g4_" 
				+ str(n4[i]) + "_g5_" + str(n5[i])+ "_g6_" + str(n6[i])+ "_g7_" + str(n7[i])+ "_q_" + str(u5[i]))
			u_param.append(u5[i]);

	SNRg = SNR;
	SNRq = SNR;

	
	return name_list, param_list, u_param, g_std, SNRg, SNRq

def _grid_search_goog(g_intvl=1, g_range=(5e-4, 5e-2), q_intvl=1, q_range=(3e-3, 5e-2), depth = 1):
	# g_inc = (g_range[1] - g_range[0]) / g_intvl
	# q_inc = (q_range[1] - q_range[0]) / q_intvl
	# g_std, q_std = [], []
	# for i in range(g_intvl):
	# 	g_std.append(g_range[0] + i * g_inc)
	# for j in range(q_intvl):
	# 	q_std.append(q_range[0] + j * q_inc)
	# name_list = []
	# param_list = []
	# u_param = []


	# ##MESH GRID SNR
	# name_list = []
	# param_list = []
	# u_param = []

	# SNRg = np.linspace(40,60, num=7);
	# g_std = np.zeros(len(SNRg));
	# for i in range(len(SNRg)):
	# 	expon = (-SNRg[i]+10*math.log10(.5))/10;
	# 	expon2 = math.pow(10, expon);
	# 	g_std[i] = math.sqrt(expon2);
	
	# SNRq = [20, 26, 32, 38, 44, 50, 56]
	# q_std = np.zeros(len(SNRq));
	# for j in range(len(SNRq)):
	# 	expon = (-SNRq[j]+10*math.log10(1.5))/10;
	# 	expon2 = math.pow(10, expon);
	# 	q_std[j] = math.sqrt(expon2);

	# g_std_v,q_std_v = np.meshgrid(g_std,q_std)
	# g_std_f = g_std_v.flatten()
	# q_std_f = q_std_v.flatten()
	# ###/MESH GRID SNR

	
	#1D SNR
	name_list = []
	param_list = []
	u_param = []

	if q_intvl ==1:
		SNR_min=38
	else:
		SNR_min=40

	expon_minq = (-SNR_min+10*math.log10(1.5))/10;
	expon2_minq = math.pow(10, expon_minq);
	q_min = math.sqrt(expon2_minq)
	
	expon_ming = (-SNR_min+10*math.log10(.5))/10;
	expon2_ming = math.pow(10, expon_ming);
	g_min = math.sqrt(expon2_ming)
 

	if q_intvl == 1:
		SNR = np.linspace(10,60, num=g_intvl);
		g_std = np.zeros(len(SNR));
		for i in range(len(SNR)):
			expon = (-SNR[i]+10*math.log10(.5))/10;
			expon2 = math.pow(10, expon);
			g_std[i] = math.sqrt(expon2);
		
		q_std = []
		for j in range(len(g_std)):
			q_std.append(q_min)

	else:
		SNR = np.linspace(10,60, num=q_intvl);
		q_std = np.zeros(len(SNR));
		for i in range(len(SNR)):
			expon = (-SNR[i]+10*math.log10(1.5))/10;
			expon2 = math.pow(10, expon);
			q_std[i] = math.sqrt(expon2);

		g_std = []
		for j in range(len(q_std)):
			g_std.append(g_min)

	SNRg = SNR;
	SNRq = SNR;

	g_std= np.array(g_std);
	q_std= np.array(q_std);

	g_std_f = g_std
	q_std_f = q_std
#<<<<<<< HEAD
	# ## /1D

	# ##FOR NON "SNR"
	g_std_v,q_std_v = np.meshgrid(g_std,q_std)
	g_std_f = g_std_v.flatten()
	q_std_f = q_std_v.flatten()

	## 10/05/2015 fix by Cary: SNR should not be equal to std
#	SNRg = g_std;
#	SNRq = q_std;
	# ##
#=======
	## /1D

	# ##FOR NON "SNR"
	# g_std_v,q_std_v = np.meshgrid(g_std,q_std)
	# g_std_f = g_std_v.flatten()
	# q_std_f = q_std_v.flatten()
	# SNRg = g_std;
	# SNRq = q_std;
	# # ##
#>>>>>>> b795f5b572296d8789a14372ba97a49b67633176

	n0v = g_std_f*scale_val("data");
	n1v = g_std_f*scale_val("conv1/7x7_s2");
	n2v = g_std_f*scale_val("pool1/norm1");


	n3v = g_std_f*scale_val("conv2/3x3_reduce");
	n4v = g_std_f*scale_val("conv2/3x3");
	n5v = g_std_f*scale_val("conv2/norm2");
	
	n6v = g_std_f*scale_val("inception_3a/1x1");
	n7v = g_std_f*scale_val("inception_3a/3x3_reduce");
	n8v = g_std_f*scale_val("inception_3a/3x3");
	n9v = g_std_f*scale_val("inception_3a/5x5_reduce");
	n10v = g_std_f*scale_val("inception_3a/5x5");
	n11v = g_std_f*scale_val("inception_3a/pool_proj");
	n12v = g_std_f*scale_val("inception_3b/1x1");
	n13v = g_std_f*scale_val("inception_3b/3x3_reduce");
	n14v = g_std_f*scale_val("inception_3b/3x3");
	n15v = g_std_f*scale_val("inception_3b/5x5_reduce");
	n16v = g_std_f*scale_val("inception_3b/5x5");
	n17v = g_std_f*scale_val("inception_3b/pool_proj");

	n18v = g_std_f*scale_val("inception_4a/1x1");
	n19v = g_std_f*scale_val("inception_4a/3x3_reduce");
	n20v = g_std_f*scale_val("inception_4a/3x3");
	n21v = g_std_f*scale_val("inception_4a/5x5_reduce");
	n22v = g_std_f*scale_val("inception_4a/5x5");
	n23v = g_std_f*scale_val("inception_4a/pool_proj");
	n24v = g_std_f*scale_val("inception_4b/1x1");
	n25v = g_std_f*scale_val("inception_4b/3x3_reduce");
	n26v = g_std_f*scale_val("inception_4b/3x3");
	n27v = g_std_f*scale_val("inception_4b/5x5_reduce");
	n28v = g_std_f*scale_val("inception_4b/5x5");
	n29v = g_std_f*scale_val("inception_4b/pool_proj");

	u1v = q_std_f*scale_val("pool1/norm1");
	u2v = q_std_f*scale_val("pool2/3x3_s2");
	un3v = q_std_f*scale_val("inception_3a/output");
	u3v = q_std_f*scale_val("pool3/3x3_s2");
	u4v = q_std_f*scale_val("inception_4a/output");
	u5v = q_std_f*scale_val("inception_4b/output");
	#

	######UNCOMMENT BELOW FOR MESH GRID
	# if (depth == 1):
	# 	n0,n1,u1 = np.meshgrid(n0v,n1v,u1v)
	# 	n0=n0.flatten();
	# 	n1=n1.flatten();
	# 	u1=u1.flatten();
	# 	for i in range(len(n0)):
	# 		param_list.append([n0[i], n1[i]])
	# 		name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_q2_" + str(u1[i]))
	# 		u_param.append(u1[i]);
	#####

	if (depth == 1):
		n0=n0v
		n1=n1v
		n2=n2v
		u1=u1v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" + str(n2[i]) + "_q2_" + str(u1[i]))
			u_param.append(u1[i]);

	######UNCOMMENT BELOW FOR MESH GRID
	# if (depth == 2):
	# 	n0,n1,n2,n3,u2 = np.meshgrid(n0v,n1v,n2v,n3v,u2v)
	# 	n0=n0.flatten();
	# 	n1=n1.flatten();
	# 	n2=n2.flatten();
	# 	n3=n3.flatten();
	# 	u2=u2.flatten();
	# 	for i in range(len(n0)):
	# 		param_list.append([n0[i], n1[i], n2[i], n3[i]])
	# 		name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i]) + "_q2_" + str(u2[i]))
	# 		u_param.append(u2[i]);
	######

	if (depth == 2):
		n0=n0v
		n1=n1v
		n2=n2v
		n3=n3v
		n4=n4v
		n5=n5v
		u2=u2v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i], n5[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i])+ "_g5_" + str(n5[i]) + "_q_" + str(u2[i]))
			u_param.append(u2[i]);

	######UNCOMMENT BELOW FOR MESH GRID		
	# if (depth == 3):
	# 	n0,n1,n2,n3,n4,n5,u3 = np.meshgrid(n0v,n1v,n2v,n3v,n4v,n5v,u3v)
	# 	n0=n0.flatten();
	# 	n1=n1.flatten();
	# 	n2=n2.flatten();
	# 	n3=n3.flatten();
	# 	n4=n4.flatten();
	# 	n5=n5.flatten();
	# 	u3=u3.flatten();
	# 	for i in range(len(n0)):
	# 		param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i], n5[i]])
	# 		name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i]) + "_g4_" 
	# 			+ str(n4[i]) + "_g5_" + str(n5[i]) + "_q2_" + str(u3[i]))
	# 		u_param.append(u3[i])
	#######

	if (depth == -3):
		n0=n0v
		n1=n1v
		n2=n2v
		n3=n3v
		n4=n4v
		n5=n5v
		n6=n6v
		n7=n7v
		n8=n8v
		n9=n9v
		n10=n10v
		n11=n11v
		u3=un3v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i], n5[i],n6[i],n7[i],n8[i],
				n9[i],n10[i],n11[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i]) + "_q_" + str(u3[i]))
			u_param.append(u3[i]);


	if (depth == 3):
		n0=n0v
		n1=n1v
		n2=n2v
		n3=n3v
		n4=n4v
		n5=n5v
		n6=n6v
		n7=n7v
		n8=n8v
		n9=n9v
		n10=n10v
		n11=n11v
		n12=n12v
		n13=n13v
		n14=n14v
		n15=n15v
		n16=n16v
		n17=n17v
		u3=u3v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i], n5[i],n6[i],n7[i],n8[i],
				n9[i],n10[i],n11[i],n12[i],n13[i],n12[i],n15[i],n16[i],n17[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_g3_" + str(n3[i]) + "_q_" + str(u3[i]))
			u_param.append(u3[i]);

	if (depth == 4):
		n0=n0v
		n1=n1v
		n2=n2v
		n3=n3v
		n4=n4v
		n5=n5v
		n6=n6v
		n7=n7v
		n8=n8v
		n9=n9v
		n10=n10v
		n11=n11v
		n12=n12v
		n13=n13v
		n14=n14v
		n15=n15v
		n16=n16v
		n17=n17v
		n18=n18v
		n19=n19v
		n20=n20v
		n21=n21v
		n22=n22v
		n23=n23v
		u4=u4v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i], n5[i],n6[i],n7[i],n8[i],
				n9[i],n10[i],n11[i],n12[i],n13[i],n12[i],n15[i],n16[i],n17[i],n18[i],n19[i],n20[i],n21[i],n22[i],n23[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_q_" + str(u4[i]))
			u_param.append(u4[i]);	

	if (depth == 5):
		n0=n0v
		n1=n1v
		n2=n2v
		n3=n3v
		n4=n4v
		n5=n5v
		n6=n6v
		n7=n7v
		n8=n8v
		n9=n9v
		n10=n10v
		n11=n11v
		n12=n12v
		n13=n13v
		n14=n14v
		n15=n15v
		n16=n16v
		n17=n17v
		n18=n18v
		n19=n19v
		n20=n20v
		n21=n21v
		n22=n22v
		n23=n23v
		n24=n24v
		n25=n25v
		n26=n26v
		n27=n27v
		n28=n28v
		n29=n29v
		u5=u5v

		for i in range(len(n0)):
			param_list.append([n0[i], n1[i], n2[i], n3[i], n4[i], n5[i],n6[i],n7[i],n8[i],
				n9[i],n10[i],n11[i],n12[i],n13[i],n12[i],n15[i],n16[i],n17[i],n18[i],n19[i],n20[i],n21[i],n22[i],n23[i],n24[i],
				n25[i],n26[i],n27[i],n28[i],n29[i]])
			name_list.append("g0_" + str(n0[i]) + "_g1_" + str(n1[i]) + "_g2_" +str(n2[i]) + "_q_" + str(u5[i]))
			u_param.append(u5[i]);					

	
	return name_list, param_list, u_param, g_std, SNRg, SNRq
