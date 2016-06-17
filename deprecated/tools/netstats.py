import caffe;
import numpy as np;
import os;
net = caffe.Net('../../caffe_exp/caffe/models/bvlc_googlenet/deploy.prototxt','../../caffe_exp/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel',caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load('../../caffe_exp/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
net.blobs['data'].reshape(1,3,224,224)
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image('../../caffe_exp/caffe/examples/images/cat.jpg'))

out=net.forward();

if not os.path.exists('../net_stats'):
    os.makedirs('../net_stats')

k = open('../net_stats/' + 'googlenet_data_stats.txt','w')
for key in net.blobs.keys():
	name = key;
	min = net.blobs[name].data[0].min();
	max = net.blobs[name].data[0].max();
	mean = net.blobs[name].data[0].mean();
	std = net.blobs[name].data[0].std();
	try:
		shape = net.params[name][0].data[0].shape;
		k.write(name+','+str(min)+','+str(max)+','+str(mean)+','+str(std)+','+str(shape)+'\n')	
	except:
		pass
	
k.close()
k = open('../net_stats/' + 'googlenet_data_stats_val_range.txt','w')

k.write("val_range = {");
for key in net.blobs.keys():
	name = key;
	min = net.blobs[name].data[0].min();
	max = net.blobs[name].data[0].max();
	mean = net.blobs[name].data[0].mean();
	std = net.blobs[name].data[0].std();
	shape = net.blobs[name].data[0].shape;
	k.write("\""+name+"\": ("+str(mean)+','+str(std)+'), \n');

k.write("}");
k.close();
