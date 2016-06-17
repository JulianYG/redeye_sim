#include <limits>
#include <algorithm>

#include "caffe/syncedmem.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/noise_layer.hpp"

namespace caffe {

template<typename Dtype>
void NoiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
	const vector<Blob<Dtype>*>& top) {
	// allocate memory space for random noise
	const int noise_size = bottom[0]->count();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	Dtype* rnoise = static_cast<Dtype*>(rand_noise_.mutable_gpu_data());
	switch(_noise_type) {
		case NoiseParameter_NoiseType_GAUSSIAN: {
			caffe_gpu_rng_gaussian(noise_size, _param1, _scale * _param2, rnoise);
		} 	break;
		case NoiseParameter_NoiseType_POISSON: {
			std::poisson_distribution<int> 
				_poisson_engine(_scale * _param1);
			for (int s = 0; s < noise_size; s++) {
				rnoise[s] = _poisson_engine(_generator) / _param2;
			}
		}	break;
		case NoiseParameter_NoiseType_UNIFORM: {
			caffe_gpu_rng_uniform(noise_size, (Dtype) 0., 
				_scale * (_param2 - _param1), rnoise);
		}	break;
	}
	caffe_gpu_add(noise_size, bottom_data, rnoise, top_data);
	caffe_copy(top[1]->count(), this->blobs_[0].get()->gpu_data(), top[1]->mutable_gpu_data());
}

template<typename Dtype>
void NoiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	
	if (_if_pass)
		return;

	vector<int> blob_shape_ = top[0]->shape();
	const int col_num_ = blob_shape_[2] * blob_shape_[3];

	const Dtype* top_diff = top[0]->gpu_diff();
	const Dtype* sd_weight = this->blobs_[0]->gpu_data();
	Dtype* sd_weight_diff = this->blobs_[0]->mutable_gpu_diff();

	if (propagate_down[0] && !_if_pass) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		// set bottom_diff = top_diff x sd_weight
		for (int n = 0; n < blob_shape_[0]; n++) {
			for (int c = 0; c < blob_shape_[1]; c++) {
				caffe_gpu_axpby<Dtype>(col_num_, this->blobs_[0]->data_at(n, c, 0, 0),
					top_diff + top[0]->offset(n, c), (Dtype)0., 
					bottom_diff + bottom[0]->offset(n, c));
			}
		}
		if (this->param_propagate_down_[0]) {
			// fill the weight_diff array with 0 first for computation
			caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), sd_weight_diff);
			// next update the noise parameters
			// weight_diff = top_diff x bottom_data
			for (int n = 0; n < blob_shape_[0]; n++) {
				for (int c = 0; c < blob_shape_[1]; c++) {
					caffe_gpu_dot(this->blobs_[0]->count(), 
						top_diff + top[0]->offset(n, c), bottom_data + bottom[0]->offset(n,c),
							sd_weight_diff + this->blobs_[0]->offset(n, c));				
				}
			}
			caffe_gpu_scal(this->blobs_[0]->count(), (const Dtype) _diff_scale, sd_weight_diff);
		}	
	}
	// else {
	// 	caffe_copy(top[0]->count(), top_diff, bottom[0]->mutable_gpu_diff());
	// //	caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
	// }
}

INSTANTIATE_LAYER_GPU_FUNCS(NoiseLayer);

}	// namespace caffe
