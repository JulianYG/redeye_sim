#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/energy_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void EnergyLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	// set default loss weight 0.4
	if (this->layer_param_.loss_weight_size() == 0) {
    	this->layer_param_.add_loss_weight(Dtype(0.4));
  	}
  	_alpha = this->layer_param_.energy_loss_param().alpha();
	_model = this->layer_param_.energy_loss_param().model();
	_param_types = vector<int>(bottom.size(), 0);
	// initialize the vector indicating noise type
	if (this->layer_param_.energy_loss_param().paramtype_size())
		for (int i = 0; i < bottom.size(); i++)
			_param_types[i] = this->layer_param_.energy_loss_param().paramtype(i);
}

template <typename Dtype>
void EnergyLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	
	vector<int> top_shape(0);
	top[0]->Reshape(top_shape);
	// initialize the loss value
	top[0]->mutable_gpu_data()[0] = (Dtype) 0.;
	LOG(INFO) << "Scaling parameter value: " << _alpha;
}

template <typename Dtype>
void EnergyLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	for (int bottom_id = 0; bottom_id < bottom.size(); bottom_id++) {
		const Dtype* noise_param = bottom[bottom_id]->gpu_data();
		const int param_size = bottom[bottom_id]->count();
		LOG(INFO) << "Noise parameters blob length: " << param_size;
		Dtype loss_val = (Dtype) 0.;
		switch (_param_types[bottom_id]) {
			case 0: {	// case for gaussian, default
				for (int i = 0; i < param_size; i++) {
					switch(_model) {
						case EnergyLossParameter_ModelType_LINEAR: {
							loss_val -= _alpha * noise_param[i];
						} 	break;
					    case EnergyLossParameter_ModelType_SQUARE_INVERSE: { 
					    	loss_val += _alpha / pow(noise_param[i], 2);
						} 	break;
						case EnergyLossParameter_ModelType_RESERVED: {}	break;
					}
				}
			}	break;
			case 1: {}	break;	// case for poisson, no energy loss
			case 2: {	// case for uniform additive noise
				for (int i = 0; i < param_size; i++) {
					switch(_model) {
						case EnergyLossParameter_ModelType_LINEAR: {
							loss_val -= _alpha * noise_param[i];
						} 	break;
					    case EnergyLossParameter_ModelType_SQUARE_INVERSE: { 
					    	loss_val += _alpha / pow(noise_param[i], 2);
						} 	break;
						case EnergyLossParameter_ModelType_RESERVED: {}	break;
					}
				}
			} 	break;
		}	// also needs normalization
		top[0]->mutable_gpu_data()[0] += loss_val / param_size;
		LOG(INFO) << "Energy loss value for this bottom blob: " << loss_val / param_size;
	}
	LOG(INFO) << "top loss output: " << top[0]->mutable_gpu_data()[0];
	LOG(INFO) << "example noise param: " << bottom[0]->gpu_data()[0];
}

template <typename Dtype>
void EnergyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(EnergyLossLayer);

}	// namespace caffe
