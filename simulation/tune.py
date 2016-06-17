# Simply fine tune the model based on given SNR
# Usage: python tune.py --iter i --g x --q y --d d

import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(os.getcwd())))
from lib import utils, setup
from proto_writer import _write_prototxt as wp
from google.protobuf import text_format as proto
from caffe.proto import caffe_pb2
import caffe

caffe.set_mode_gpu()
config = setup.config()
solver_prototxt_temp = config['solver_prototxt_template']
template = config['train_prototxt_template']
pretrained_model = config['pretrained_model']

parser = argparse.ArgumentParser(description='Tune model with given Gaussian and quantization noise')
parser.add_argument('--iter', type=int, help='number of iterations for fine tuning', required=True)
parser.add_argument('--g', type=int, help='gaussian SNR', required=True)
parser.add_argument('--q', type=int, help='quantization layer SNR', required=True)
parser.add_argument('--d', type=int, help='depth of noise contamination', required=True)

args = parser.parse_args()
iter_num = args.iter
g_snr, q_snr = args.g, args.q
depth = args.d

train_net = caffe_pb2.NetParameter()
proto.Merge((open(template).read()), train_net)
net = caffe.Net(template, pretrained_model, caffe.TRAIN)
net.forward()
param = [q_snr] + [g_snr]*depth
file_name = os.path.join('../prototxt/train/redeye/', 'goog_train' + str(param).replace(' ',
	'').replace('[','_').replace(']','').replace(',','_') + '.prototxt')

wp(train_net, param, file_name, depth, net.blobs)
del(net)
solver, _ = utils.write_solver_prototxt(solver_prototxt_temp,
	[file_name], iter_num, path='../prototxt/solver/redeye')

os.system('../caffe/build/tools/caffe train --solver='\
	+ solver[0] + ' -weights ' + pretrained_model)

