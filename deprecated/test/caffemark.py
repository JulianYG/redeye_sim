import numpy as np
import random
import os
from scipy import misc
import heapq
import caffe
import Image
import setup

paths = setup.config()
croot = paths.getLoc('caffeRoot')

class Caffemark:
    """
    A benchmark module to evaluate the degradation tolerance of image
    qualities. Requires input as image files, degradation parameters,
    and the caffe model for testing. Generates statistics for given 
    degradation parameters.
    """
    def __init__(self, croot, start, num, prototxt, model, modelSuffix, protoSuffix, ifOversample=False, debug=True,optional_trainedfile=None,optional_modelfile=None,optional_imagemean=None):
        """
        Initialize the files for testing.
        Input arguments: 
            croot, iroot: caffe root and image root folder. Future improvement:
                as the atomic class, iroot should be changed to a single image path?
            start: which image to start with inside iroot
            num: number of images to classify
            model: the model used for prediction
            modelSuffix: a string after 'bvlc_xxnet' to indicate which model to use.
            ifOversample: the default is to do 10 predictions, cropping the center and corners
                of the image as well as their mirrored versions, and average over the predictions.
                This is set to false for performance.
        """
        self._caffe_root = croot
        self._numOfClass = 1000
        self._protoSuffix = protoSuffix
        self._modelSuffix = modelSuffix
        self.numOfImages = num
        self.startIdx = start
        self._model_file, self._trained_file = prototxt, model
        if not('optional_modelfile'):
            self._model_file = optional_modelfile
        if not('optional_trainedfile'):
            self._trained_file =  optional_trainedfile
        self.ifOversample = ifOversample
        self.debugMode = debug
        caffe.set_mode_gpu()
        #caffe.set_mode_cpu()
        if not('optional_imagemean'):
                    self.net = caffe.Classifier(self._model_file,
                                    self._trained_file,
                                    mean=np.load(optional_imagemean).mean(1).mean(1),
                                    channel_swap=(2,1,0),
                                    raw_scale=255,
                                    image_dims=(256,256))
        else:  
                    self.net = caffe.Classifier(self._model_file,
                                    self._trained_file,
                                    mean=np.load(os.path.join(croot,'python/caffe/imagenet/ilsvrc_2012_mean.npy')).mean(1).mean(1),
                                    channel_swap=(2,1,0),
                                    raw_scale=255,
                                    image_dims=(256,256))   
        
    def __basePredict__(self):
        """
        Set the base line prediction for later comparison
        """
        prediction = self.net.predict(self.image_list, self.ifOversample)
        if self.debugMode:
            for pred in prediction:
                print 'predicted class: ', pred.argmax()
        return prediction
        
    def __parseModel__(self, modelName):
        name = modelName.lower()
        if name == 'googlenet':
            modelDir = os.path.join(self._caffe_root, 'models/googlenet', 'deploy_' + self._protoSuffix + '.prototxt')
            trainedDir = os.path.join(self._caffe_root, 'models/bvlc_googlenet', 'googlenet' + self._modelSuffix + '.caffemodel')
        elif name == 'googletab':
            modelDir = os.path.join(self._caffe_root, 'googletab', 'deploy_' + self._protoSuffix + '.prototxt')
            trainedDir = os.path.join(self._caffe_root, 'models/googlenet', 'bvlc_googlenet' + self._modelSuffix + '.caffemodel')
        elif name == 'googlenet_digit':
            modelDir = os.path.join(self._caffe_root, 'models/googlenet', 'deploy' + self._protoSuffix + '.prototxt')
            trainedDir = os.path.join(self._caffe_root, 'models/bvlc_googlenet_digit1', 'bvlc_googlenet_digit1' + self._modelSuffix + '.caffemodel')
        elif name == 'alextab':
            modelDir = os.path.join(self._caffe_root, 'alextab', 'deploy_' + self._protoSuffix + '.prototxt')
            trainedDir = os.path.join(self._caffe_root, 'models/bvlc_alexnet', 'bvlc_alexnet' + self._modelSuffix + '.caffemodel')
            # modelDir = os.path.join(self._caffe_root, 'alextab', 'deploy.prototxt')
            # trainedDir = os.path.join(self._caffe_root, 'models/bvlc_alexnet', 'bvlc_alexnet.caffemodel')
        elif name == 'alexnet':
            modelDir = os.path.join(self._caffe_root, 'models/bvlc_alexnet', 'deploy_' + self._protoSuffix + '.prototxt')
            trainedDir = os.path.join(self._caffe_root, 'models/bvlc_alexnet', 'bvlc_alexnet' + self._modelSuffix + '.caffemodel')
        elif name == 'reference_caffenet':
            modelDir = os.path.join(self._caffe_root, 'models/bvlc_reference_caffenet', 'deploy_' + self._protoSuffix + '.prototxt')
            trainedDir = os.path.join(self._caffe_root, 'models/bvlc_reference_caffenet', 'bvlc_reference_caffenet.caffemodel')
        elif name == 'reference_rcnn_ilsvrc13':
            modelDir = os.path.join(self._caffe_root, 'models/bvlc_reference_rcnn_ilsvrc13', 'deploy_' + self._protoSuffix + '.prototxt')
            trainedDir = os.path.join(self._caffe_root, 'models/bvlc_reference_rcnn_ilsvrc13', 'bvlc_reference_rcnn_ilsvrc13.caffemodel')
        
        return modelDir, trainedDir

    def __subset__(self, lst):
        """
        Pick N items from a list.
        """
        for img in lst[self.startIdx : self.startIdx + self.numOfImages]:
            # there might be less imgs than expected, but rarely
            if os.path.isfile(os.path.join(self._image_file, img)):
                self.image_list.append(caffe.io.load_image(os.path.join(self._image_file, img)))

    def set(self, iroot):
        self._image_file = iroot
        self.image_list = []
        self.deg_img_list = []
        self.__subset__(sorted(os.listdir(self._image_file)))
        self.baseline = self.__basePredict__()

    def reset(self):
        self.image_list = []
        prediction = []

    def clean(self):
        self.deg_img_list = []

    def setDegradeParam(self, deg=[], params=[]):
        """
        Setting the degradation arameters for the module. Parameters include
        types of degradation, and exact corresponding value for quality degradation.
        Currently the types supported are Gaussian noise and resizing.
        Provided parameters should be wrapped in a tuple, e.g, for Gaussian noise,
        the corresponding parameter should be (mean, std); for resizing image,
        the parameter could either be an integer, or a float number, or a tuple
        indicating the height and width of the output image.
        """
        if not deg or not params:
            for img in self.image_list:
                self.deg_img_list.append(img)
        else:
            for img in self.image_list:
                dirty_img = img + 0     # to avoid shallow copy?
                i = -1
                for degtype in deg:
                    i += 1
                    if 'gaussian' in degtype.lower():
                        dirty_img = self.__addGNoise__(dirty_img, params[i])
                    elif degtype.lower() == 'resize':
                        dirty_img = self.__resize__(dirty_img, params[i])
                    elif degtype.lower() == 'poisson':
                        dirty_img = self.__addPNoise__(dirty_img, params[i])
                    elif degtype.lower() == 'quality':
                        dirty_img = self.__quality__(dirty_img, params[i])
                    elif degtype.lower() == 'clean':
                        dirty_img = dirty_img + 0
                    else:
                        continue    # leave for now
                if self.debugMode:
                    misc.imshow(dirty_img)
                self.deg_img_list.append(dirty_img)               

    def __addGNoise__(self, img, stddev):
        gaussianNoise = np.random.normal(0, self.__gammaFunction__(float(stddev)), img.shape)
        return img + gaussianNoise

    def __addPNoise__(self, img, lam):
        poissonNoise = np.random.poisson(self.__gammaFunction__(float(lam)), img.shape)
        return img + poissonNoise

    def __resize__(self, img, quality):
        crop = 1.0 - quality
        temp_img = caffe.io.resize(img, (int(np.ceil(crop * img.shape[0])), int(np.ceil(crop * img.shape[1]))))
        dirty_img = caffe.io.resize(temp_img, (img.shape[0], img.shape[1]))
        return dirty_img
    
    def __quality__(self, img, quality):
        q = int(100 - quality)
        misc.imsave(os.path.join(croot, 'temp_qual.JPEG'), img)
        image = Image.open(os.path.join(croot, 'temp_qual.JPEG'))
        image.save(os.path.join(croot, 'temp_qual.JPEG'), 'JPEG', quality=q)
        dirty_img = caffe.io.load_image(os.path.join(croot, 'temp_qual.JPEG'))
        return dirty_img

    def runDegradation(self, topN, thresh):
        """
        Returns number of matches specified for one run as first output, 
        and similarity between prediction vectors as second output.
        """
        prediction = self.net.predict(self.deg_img_list, self.ifOversample)
        if self.debugMode:
            for pred in prediction:
                print 'deg_predicted class: ', pred.argmax()
 #       return self.__getStat__(topN, thresh)    
        return self.baseline, prediction

    def __normalize__(self, vector):
        return [(float(i) - min(vector))/(max(vector) - min(vector)) for i in vector]

    def __thresh__(self, vec, prc):
        threshold = max(vec) * prc
        for i in range(len(vec)):
            if vec[i] < threshold:
                vec[i] = 0
        return vec

    def __gammaFunction__(self, pixel):
         gamma = 2.2
         intensity = (pixel / float(255)) ** (1/float(gamma))
         return intensity

    def __getStat__(self, n, t):

        v_sim, r_sim = 0.0, 0.0
        TP, FP, TN, FN = 0.0, 0.0, 0.0, 0.0
        pos_denom, neg_denom = 0, 0

        for i in range(0, len(self.baseline)):
            base_predict = self.baseline[i]
            deg_predict = self.degraded_predict[i]
            norm_base_predict = self.__normalize__(base_predict)
            norm_deg_predict = self.__normalize__(deg_predict)
            top_base_idx = np.argsort(norm_base_predict)[-n:][::-1]
            top_deg_idx = np.argsort(norm_deg_predict)[-n:][::-1]

            threshed_base_pred = self.__thresh__(norm_base_predict, t)
            threshed_deg_pred = self.__thresh__(norm_deg_predict, t)

            threshed_base_idx = []
            threshed_deg_idx = []

            # get the indices of unremoved values
            for i in range(len(threshed_base_pred)):
                if threshed_base_pred[i] != 0:
                    threshed_base_idx.append(i)

            for j in range(len(threshed_deg_pred)):
                if threshed_deg_pred[j] != 0:
                    threshed_deg_idx.append(j)
            idx_pool = range(self._numOfClass)
        
            TP += len(set(threshed_base_idx) & set(threshed_deg_idx))/float(len(threshed_base_idx))
            FP += len(set(idx_pool).difference(set(threshed_base_idx)) & set(threshed_deg_idx))/float(len(set(idx_pool).difference(set(threshed_base_idx))))
            FN += len(set(threshed_base_idx) & set(idx_pool).difference(threshed_deg_idx))/float(len(threshed_base_idx))
            TN += len(set(idx_pool).difference(set(threshed_base_idx)) & set(idx_pool).difference(set(threshed_deg_idx)))/float(len(set(idx_pool).difference(set(threshed_base_idx))))
            pos_denom += len(threshed_base_idx)
            neg_denom += len(set(idx_pool).difference(set(threshed_base_idx)))
            # an intuitive naive way
            v_sim += float((len(set(top_base_idx) & set(top_deg_idx)))) / len(set(top_base_idx))
            if base_predict.argmax() == deg_predict.argmax():
                r_sim += 1

        TP /= len(self.baseline)
        FP /= len(self.baseline)
        TN /= len(self.baseline)
        FN /= len(self.baseline)
        r_sim /= len(self.baseline)
        v_sim /= len(self.baseline)
        
        return r_sim, v_sim, tuple([TP, TN, FP, FN, pos_denom, neg_denom])
