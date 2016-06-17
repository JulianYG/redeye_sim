from scipy import misc
import os
import numpy as np
import sys
import random
croot = '/home/roblkw/Work/caffe/'
sys.path.insert(0, croot + 'python')
sys.path.insert(0, croot + 'distribute')
sys.path.insert(0, croot + 'distribute/python')
sys.path.insert(0, croot + 'distribute/python/caffe')
import caffe

# croot = '/home/julian/caffe/'
# train = '/home/julian/caffe/train'
# val = '/home/julian/caffe/val'
croot = '/home/roblkw/Work/caffe'
val = '/home/roblkw/Work/caffe/ILSVRC2013_DET_val'
train = '/home/roblkw/Work/caffe/ILSVRC2013_DETextra_train'

def degradeTrain(start, numOfBatch, caffeRoot, trainRoot, deg=[], params=[]):
    """
    A simple function to perform degradation operation on the training image set.
    Input arguments:
        start: the starting number of image folder, subsetxxx
        numOfBatch: number of folders to train
        trainRoot: the parent folder that contains all subsetxxx folders
        deg, params: degradation type and parameters
    Output:
        A folder named 'deg_train' under caffe root containing all degraded images.
    """
    train_img = {}
    deg_train = os.path.join(caffeRoot, 'deg_train_13')
    if not os.path.exists(deg_train):
        os.makedirs(deg_train)

    for trainDir in os.listdir(trainRoot)[start : start + numOfBatch]:
        # exclude meta-data files
        if trainDir[0] != '.':
            degDir = os.path.join(deg_train, trainDir)
            if not os.path.exists(degDir):
                os.makedirs(degDir)
            for imgName in os.listdir(os.path.join(trainRoot, trainDir)):
                if os.path.isfile(os.path.join(trainRoot, trainDir, imgName)):
                    img = caffe.io.load_image(os.path.join(trainRoot, trainDir, imgName))
                    degTrainImgPath = os.path.join(degDir, imgName)
                    train_img[degTrainImgPath] = img

    if degtype.lower() == 'fpn':
        noise_cover = noiseGenerator(params[0], params[1])

    for imgPath, img in train_img.items():
        i = -1
        for degtype in deg:
            i += 1
            if degtype.lower() == 'gaussian':
                dirty_img = addGNoise(img, params[i])
            elif degtype.lower() == 'resize':
                dirty_img = resize(img, params[i])
            elif degtype.lower() == 'poisson':
                dirty_img = addPNoise(img, params[i])
            elif degtype.lower() == 'fpn':
                dirty_img += noise_cover
            misc.imsave(imgPath, dirty_img)

def degradeVal(start, numOfImg, caffeRoot, valRoot, deg=[], params=[]):
    """
    A simple function to perform degradation operation on the validation image set.
    Input arguments:
        start: the starting number of image folder, subsetxxx
        numOfBatch: number of folders to train
        valRoot: the parent folder that contains all subsetxxx folders
        deg, params: degradation type and parameters
    Output:
        A folder named 'deg_val' under caffe root containing all degraded images.
    """
    val_img = {}
    deg_val = os.path.join(caffeRoot, 'deg_val_13') 
    if not os.path.exists(deg_val):
        os.makedirs(deg_val)
    
    for valImgName in os.listdir(valRoot)[start : start + numOfImg]:
        valImgPath = os.path.join(valRoot, valImgName)
        if os.path.isfile(valImgPath):
            val_img[valImgName] = caffe.io.load_image(valImgPath)

    if degtype.lower() == 'fpn':
        noise_cover = noiseGenerator(params[0], params[1])

    for imgName, img in val_img.items():
        j = -1
        for degtype in deg:
            j += 1
            if degtype.lower() == 'gaussian':
                dirty_img = addGNoise(img, params[j])
            elif degtype.lower() == 'resize':
                dirty_img = resize(img, params[j])
            elif degtype.lower() == 'poisson':
                dirty_img = addPNoise(onimg, params[j])
            elif degtype.lower() == 'fpn':
                dirty_img += noise_cover
            misc.imsave(os.path.join(deg_val, imgName), dirty_img)

def addGNoise(img, stddev):
    """
    Assumes gaussian noise mean is always zero.
    """
    gaussianNoise = np.random.normal(0, gammaFunction(float(stddev)), img.shape).astype(float)
    dirty_img = img + gaussianNoise
    return dirty_img

def noiseGenerator(sd_range, img_shape):
    """
    Given a tuple indicating range of gaussian standard deviation, generate random noise
    for each single pixel with sd within that range
    """
    noise_skin = np.empty(img_shape)
    for x_axis in img_shape[0]:
        for y_axis in img_shape[1]:
            for channel in img_shape[2]:
                noise_skin[(x_axis, y_axis, channel)] = 
                    random.gauss(0, random.uniform(sd_range[0], sd_range[1]))
    return noise_skin

def addPNoise(img, lam):
    poissonNoise = np.random.poisson(gammaFunction(float(lam)), img.shape)
    dirty_img = img + poissonNoise
    return dirty_img

def resize(img, quality):
    crop = 1.0 - quality
    temp_img = caffe.io.resize(img, (int(np.ceil(crop * img.shape[0])), int(np.ceil(crop * img.shape[1]))))
    dirty_img = caffe.io.resize(temp_img, (img.shape[0], img.shape[1]))
    return dirty_img

def gammaFunction(pixel):
    gamma = 2.2
    intensity = (pixel / float(255)) ** (1/float(gamma))
    return intensity

# print "Running degradation on validation dataset."
# for i in range(100):
#     degradeVal(i * 500, 500, croot, val, ['resize'], [0.7])
#     if (i % 3 == 0):
#         print '.'

# print "Running degradation on training dataset."
# for k in range(40):    
#     degradeTrain(k * 3, 3, croot, train, ['resize'], [0.7])
#     if (k % 4 == 0):
#         print '.'

# print "Running degradation on validation dataset."
# for i in range(57):
#     degradeVal(i * 353, 353, croot, val, ['resize'], [0.75])
#     if (i % 3 == 0):
#         print '.'

# print "Running degradation on training dataset."
# for k in range(40):    
#     degradeTrain(k * 3, 3, croot, train, ['resize'], [0.75])
#     if (k % 4 == 0):
#         print '.'

print "Running degradation on training dataset."
for k in range(40):    
    degradeTrain(k * 3, 3, croot, train, ['fpn'], [(0, 5.1), (256,256,3)])
    if (k % 4 == 0):
        print '.'

# from caffe import layers as L
# from caffe import params as P

# def googlenet(lmdb, batch_size):
    # """
    # A helper function to manually create layers prototxt for solver.
    # """
#     n = caffe.NetSpec()
#     n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
#         transform_param=dict(scale=1./255), ntop=2)
#     # initialize the net

#     n.conv1_7x7_s2 = L.Convolution(n.data, kernel_size=7, num_output=64, 
#         weight_filler=dict(type='xavier', std=0.1), bias_filler=dict(type='constant',value=0.2))
#     n.conv1_relu_7x7 = L.ReLU(n.conv1_7x7_s2)
#     n.pool1_3x3_s2 = L.Pooling(n.conv1_7x7_s2, kernel_size=3, stride=2, pool=P.Pooling.MAX)

#     return n.to_proto()

caffe.set_mode_gpu()
# solver = caffe.SGDSolver('/home/julian/caffe/models/bvlc_googlenet/solver.prototxt')
# print [(k, v.data.shape) for k, v in solver.net.blobs.items()]
# dimensions for blobs: k represents name, shape is represented by
# blob dimension number(batch size) * channel * height * width

# solver.net.forward()
# solver.test_nets[0].forward()
# print solver.net.blobs['data'].data[:8,0].transpose(1,0,2).shape
# misc.imshow(solver.net.blobs['data'].data[:8,0].transpose(1,0,2).reshape(224,8*224))#,cmap='gray')
# misc.imshow(solver.test_nets[0].blobs['data'].data[:8,0].transpose(1,0,2).reshape(224,8*224))
# print solver.net.blobs['label'].data[:8]
# print solver.test_nets[0].blobs['label'].data[:8]
# solver.step(1)
# print solver.net.params['conv1/7x7_s2']
# print solver.net.params['conv1/7x7_s2'][0].diff[:,0].shape

# should be the shape of the first layer filters
# misc.imshow(solver.net.params['conv1/7x7_s2'][0].diff[:,0].reshape(8,8,7,7).transpose(0,2,1,3).reshape(8*8,7*7))

def custom_train(solverprototxt, logtxt, modelName, batchSize, niter, test_interval, 
    injectionLayer=[], forwardDegParam=[], backwardDegParam=[]):
    """
    Customize the training process by flexibly tuning number of iterations and test intervals, 
    creating log files to keep record of additional analytics, modifying net in the loop to change
    the solving process. Takes input as path name of solver.prototxt, path name for log file,
    number of images to run on, type and level of noise to be added, number of iterations to run
    on training, and the test interval for running a full test on model. 
    Requirement: niter must be greater than or equal to test_interval, otherwise raise error.
    Input Arguments:
        solverprototxt: the full path name of the solver.prototxt file used for training
        logtxt: the full path name of the output log file to keep record of training data
        modelName: the full path name of the output caffemodel
        batchSize: number of images per batch defined as in train_val.prototxt data layer
        niter: number of iterations for training
        test_interval: after how many iterations run a test on validation set and get data;
        if set to 0 or negative numbers, no test runs
    """
    if len(degParam) != len(injectionLayer):
        raise AssertionError('Noise injection dimension mismatch!')
    if niter < test_interval:
        raise AssertionError('Number of iterations must >= test interval!')
    solver = caffe.SGDSolver(solverprototxt)
    train_loss = np.zeros(niter)
    if test_interval > 0:
        test_accuracy = np.zeros(int(np.ceil(niter/test_interval)))
    output = np.zeros((niter, batchSize, 1000))

    # the last layer with name "loss3/classifier" of type "InnerProduct" has 1000 outputs
    # the main solver loop
    for i in range(niter):
        # adding noise to the layer params for each iteration to simulate analog sampling
        for layers, params in zip(injectionLayer, forwardDegParam):
            layer_forward = solver.net.blobs[layers].data
            layer_forward += np.random.normal(0, gammaFunction(float(params)), layer_forward.shape)
        for layers, params in zip(injectionLayer, backwardDegParam):
            layer_backward = solver.net.blobs[layers].diff
            layer_backward += np.random.normal(0, gammaFunction(float(params)) layer_backward.shape)

            # weight_filler = solver.net.params[layers][0].data
            # weight_filler += np.random.normal(0, float(params) / 255, weight_filler.shape)
            # if len(solver.net.params[layers]) == 2:
            #     bias_filler = solver.net.params[layers][1].data
            #     bias_filler += np.random.normal(0, float(params) / 255, bias_filler.shape)

        solver.step(1)  # run iteration by a step size of 1
        train_loss[i] = solver.net.blobs['loss3/loss3'].data    
        #layer with name "loss3/loss3" is of type "SoftmaxWithLoss", which stores loss value
        
        solver.test_nets[0].forward(start='conv1/7x7_s2')

        output[i] = solver.test_nets[0].blobs['loss3/classifier'].data[:batchSize].reshape(batchSize, 1000)
        #output[i] = solver.test_nets[0].blobs['loss3/classifier'].data[:batchSize]

        # performing test on validation dataset
        if test_interval > 0:
            if i % test_interval == 0:
                print 'Iteration', i, 'testing...'
                correct = 0
                # run on the full size of deg_val folder
                for test_it in range(100 / batchSize):
                    # this is calculated by number of val images/ test batch_size,
                    # which is recorded in train_val.prototxt, in this case 50
                    solver.test_nets[0].forward()

                    # correct += sum(solver.test_nets[0].blobs['loss3/classifier'].data.argmax(1)
                    #     == solver.test_nets[0].blobs['label'].data)
                    correct += sum(solver.test_nets[0].blobs['loss3/classifier'].data.reshape(batchSize, 1000).argmax(1)
                        == solver.test_nets[0].blobs['label'].data.flatten())

                test_accuracy[i // test_interval] = correct / 100

    with open(logtxt, 'w') as f:
        f.write('Solving results: \n Train loss: \n\t')
        f.write(str(train_loss))
        if test_interval > 0:
            f.write('\nTesting accuracy: \n\t')
            f.write(str(test_accuracy))

    solver.net.save(modelName)
    return output

# layers = caffe.SGDSolver('/home/julian/caffe/models/bvlc_googlenet/solver.prototxt').net.blobs.keys()[5:]
#layers = caffe.SGDSolver('/home/roblkw/Work/caffe/models/bvlc_googlenet/solver.prototxt').net.blobs.keys()[5:]
layers = caffe.SGDSolver('/home/roblkw/Work/caffe/models/bvlc_googlenet/solver.prototxt').net.params.keys()

# custom_train('/home/julian/caffe/models/bvlc_googlenet/solver.prototxt', 
#     '/home/julian/caffe/train.txt',
#     '/home/julian/caffe/models/bvlc_googlenet/blvlc_googlenet_deg_pytrained.caffemodel', 
#     2, 3, 0, layers[:3], [6, 6, 6], [6, 6, 6])



# outshape = custom_train('/home/roblkw/Work/caffe/models/bvlc_googlenet/solver.prototxt', 
#    '/home/roblkw/Work/caffe/train_40000.txt',
#    '/home/roblkw/Work/caffe/models/bvlc_googlenet/bvlc_googlenet_deg_pytrained_40000.caffemodel', 
#    10, 40000, 4000, layers[:3], [6, 6, 6], [4, 4, 4])

# with open('/home/roblkw/Work/caffe/seeOut.log', 'w') as f:
#     f.write(str(outshape))



# layers = ['conv1/7x7_s2', 'pool1/3x3_s2', 'pool1/norm1', 'conv2/3x3_reduce', 'conv2/3x3', 
# 'conv2/norm2', 'pool2/3x3_s2', 'pool2/3x3_s2_pool2/3x3_s2_0_split_0', 
# 'pool2/3x3_s2_pool2/3x3_s2_0_split_1', 'pool2/3x3_s2_pool2/3x3_s2_0_split_2', 
# 'pool2/3x3_s2_pool2/3x3_s2_0_split_3', 'inception_3a/1x1', 'inception_3a/3x3_reduce', 
# 'inception_3a/3x3', 'inception_3a/5x5_reduce', 'inception_3a/5x5', 'inception_3a/pool', 
# 'inception_3a/pool_proj', 'inception_3a/output', 'inception_3a/output_inception_3a/output_0_split_0', 
# 'inception_3a/output_inception_3a/output_0_split_1', 'inception_3a/output_inception_3a/output_0_split_2', 
# 'inception_3a/output_inception_3a/output_0_split_3', 'inception_3b/1x1', 'inception_3b/3x3_reduce', 
# 'inception_3b/3x3', 'inception_3b/5x5_reduce', 'inception_3b/5x5', 'inception_3b/pool', 
# 'inception_3b/pool_proj', 'inception_3b/output', 'pool3/3x3_s2', 'pool3/3x3_s2_pool3/3x3_s2_0_split_0', 
# 'pool3/3x3_s2_pool3/3x3_s2_0_split_1', 'pool3/3x3_s2_pool3/3x3_s2_0_split_2', 
# 'pool3/3x3_s2_pool3/3x3_s2_0_split_3', 'inception_4a/1x1', 'inception_4a/3x3_reduce', 
# 'inception_4a/3x3', 'inception_4a/5x5_reduce', 'inception_4a/5x5', 'inception_4a/pool', 
# 'inception_4a/pool_proj', 'inception_4a/output', 'inception_4a/output_inception_4a/output_0_split_0', 
# 'inception_4a/output_inception_4a/output_0_split_1', 'inception_4a/output_inception_4a/output_0_split_2', 
# 'inception_4a/output_inception_4a/output_0_split_3', 'inception_4a/output_inception_4a/output_0_split_4', 
# 'loss1/ave_pool', 'loss1/conv', 'loss1/fc', 'loss1/classifier', 'loss1/loss1', 'inception_4b/1x1', 
# 'inception_4b/3x3_reduce', 'inception_4b/3x3', 'inception_4b/5x5_reduce', 'inception_4b/5x5', 
# 'inception_4b/pool', 'inception_4b/pool_proj', 'inception_4b/output', 'inception_4b/output_inception_4b/output_0_split_0', 
# 'inception_4b/output_inception_4b/output_0_split_1', 
# 'inception_4b/output_inception_4b/output_0_split_2', 
# 'inception_4b/output_inception_4b/output_0_split_3', 'inception_4c/1x1', 'inception_4c/3x3_reduce', 
# 'inception_4c/3x3', 'inception_4c/5x5_reduce', 'inception_4c/5x5', 'inception_4c/pool', 
# 'inception_4c/pool_proj', 'inception_4c/output', 'inception_4c/output_inception_4c/output_0_split_0', 
# 'inception_4c/output_inception_4c/output_0_split_1', 'inception_4c/output_inception_4c/output_0_split_2', 
# 'inception_4c/output_inception_4c/output_0_split_3', 'inception_4d/1x1', 'inception_4d/3x3_reduce', 
# 'inception_4d/3x3', 'inception_4d/5x5_reduce', 'inception_4d/5x5', 'inception_4d/pool', 
# 'inception_4d/pool_proj', 'inception_4d/output', 'inception_4d/output_inception_4d/output_0_split_0', 
# 'inception_4d/output_inception_4d/output_0_split_1', 'inception_4d/output_inception_4d/output_0_split_2', 
# 'inception_4d/output_inception_4d/output_0_split_3', 'inception_4d/output_inception_4d/output_0_split_4', 
# 'loss2/ave_pool', 'loss2/conv', 'loss2/fc', 'loss2/classifier', 'loss2/loss1', 'inception_4e/1x1', 
# 'inception_4e/3x3_reduce', 'inception_4e/3x3', 'inception_4e/5x5_reduce', 'inception_4e/5x5', 
# 'inception_4e/pool', 'inception_4e/pool_proj', 'inception_4e/output', 'pool4/3x3_s2', 
# 'pool4/3x3_s2_pool4/3x3_s2_0_split_0', 'pool4/3x3_s2_pool4/3x3_s2_0_split_1', 
# 'pool4/3x3_s2_pool4/3x3_s2_0_split_2', 'pool4/3x3_s2_pool4/3x3_s2_0_split_3', 'inception_5a/1x1', 
# 'inception_5a/3x3_reduce', 'inception_5a/3x3', 'inception_5a/5x5_reduce', 'inception_5a/5x5', 
# 'inception_5a/pool', 'inception_5a/pool_proj', 'inception_5a/output', 
# 'inception_5a/output_inception_5a/output_0_split_0', 'inception_5a/output_inception_5a/output_0_split_1', 
# 'inception_5a/output_inception_5a/output_0_split_2', 'inception_5a/output_inception_5a/output_0_split_3', 
# 'inception_5b/1x1', 'inception_5b/3x3_reduce', 'inception_5b/3x3', 'inception_5b/5x5_reduce', 
# 'inception_5b/5x5', 'inception_5b/pool', 'inception_5b/pool_proj', 'inception_5b/output', 
# 'pool5/7x7_s1', 'loss3/classifier', 'loss3/loss3']

