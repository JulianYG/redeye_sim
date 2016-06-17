import caffe
import os
import setup
import modelcheck
from dial import Dial

paths = setup.config()
model_path = paths.getLoc('pretrainedModel')
solver_path = paths.getLoc('tabSolverProto')
solver_path_b = paths.getLoc('googleNetSolverProto')
train_val_path = paths.getLoc('tabTrainValProto')
save_path = paths.getLoc('fineTuneModelSave')
weights_root_c = os.path.join(setup.config().getLoc('caffeExpRoot'), 'model_param_clean')
weights_root_n = os.path.join(setup.config().getLoc('caffeExpRoot'), 'model_param_noise')
data_root_c = os.path.join(setup.config().getLoc('caffeExpRoot'), 'blob_clean')
data_root_n = os.path.join(setup.config().getLoc('caffeExpRoot'), 'blob_noise')

class Tuner():
	"""
	A finetuner to control parameters for different layers in net
	"""
	def __init__(self, GPU=0):

		self.solver = caffe.SGDSolver(solver_path)
		self.solver.net.copy_from(model_path)
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		if GPU:
			caffe.set_mode_gpu()
		else:
			caffe.set_mode_cpu()
		with open(model_path, 'r') as n:
			self._net_bak = n.read()

	def test_tune(self, iters=0):
		"""
		This is a test tuning to check whether paremeters are copied
		from the pretrained model correctly that minimize loss
		"""
		base_solver = caffe.SGDSolver(solver_path_b)
		base_solver.net.copy_from(model_path)
		
		modelcheck.write_param_data(weights_root_c, base_solver.net)
		modelcheck.write_param_data(weights_root_n, self.solver.net)
		base_solver.net.forward()
		self.solver.net.forward()
		modelcheck.write_blob_data(data_root_c, base_solver.net)
		modelcheck.write_blob_data(data_root_n, self.solver.net)		

		base_solver.step(iters)
		self.solver.step(iters)

	def manual_tune(self, iters, modelname='manually_tuned.caffemodel', name=''):
		"""
		This tuning assumes the layer has been modified manually
		in the train_val.prototxt to freeze local rate
		Name is indicated to differentiate each run
		"""
		self.solver.step(iters)
		self.solver.net.save(os.path.join(save_path, modelname))
		modelcheck.write_param_data(os.path.join(weights_root_n, name), self.solver.net)

	def auto_tune(self, iters):
		"""
		This tuning method takes info about layers to modify and 
		automatically modifies the prototxt, then run training 
		"""
		Dial(train_val_path).processFile()
		self.solver.step(iters)
		self.solver.net.save(os.path.join(save_path,
			'auto_tuned_iter' + str(iters) + '.caffemodel'))
		modelcheck.write_param_data(weights_root_n, self.solver.net)

	def reset():
		with open(model_path, 'w') as n:
			n.write(self._net_bak)
