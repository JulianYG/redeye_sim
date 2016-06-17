#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/noise_layer.hpp"

namespace caffe {

template<typename Dtype> 
void NoiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {

	CHECK_EQ(bottom.size(), 1) << "Can only have one bottom blob input.";
	// checks top size in hpp files
	NoiseParameter noise_param = this->layer_param_.noise_param();
	// to store the noise parameters and layer parameters
	this->blobs_.resize(1);
	// different noise parameters for each input image and channel
	this->blobs_[0].reset(new Blob<Dtype>(bottom[0]->num(),
		bottom[0]->channels(), 1, 1));

	// compute the correct noise param size since shape is different
	const int noise_param_size = this->blobs_[0]->count();
	// get the pointer of noise parameter array
	Dtype* blob_ptr = this->blobs_[0].get()->mutable_cpu_data();
	_if_pass = noise_param.forward_only();
	_noise_type = noise_param.ntype();
	_diff_scale = noise_param.diff_scale();

	switch(_noise_type) {
		// fill the noise param blobs with preset numbers
		case NoiseParameter_NoiseType_GAUSSIAN: {
			_param1 = noise_param.gaussian_param().mean();
			_param2 = noise_param.gaussian_param().stddev();
			_min_sd = noise_param.gaussian_param().min_sd();
			_max_sd = noise_param.gaussian_param().max_sd();
			_scale = noise_param.gaussian_param().scale();
			caffe_set(noise_param_size, _param2, blob_ptr);
			LOG(INFO) << "NoiseLayer initialized as Gaussian.";
		} 	break;
		case NoiseParameter_NoiseType_POISSON: {
			_param1 = noise_param.poisson_param()._lambda();
			_param2 = noise_param.poisson_param().norm();
			_scale = noise_param.poisson_param().scale();
			caffe_set(noise_param_size, _param1, blob_ptr);
			LOG(INFO) << "NoiseLayer initialized as Poisson.";
		}	break;
		// now assigning uniform for quantization errors
		case NoiseParameter_NoiseType_UNIFORM: {
			_param1 = noise_param.uniform_param().min_u();
			_param2 = noise_param.uniform_param().max_u();
			_scale = noise_param.uniform_param().scale();
			caffe_set(noise_param_size, (_param2 - _param1), blob_ptr);
			LOG(INFO) << "NoiseLayer initialized as UNIFORM.";
		} 	break;	// setup parameters based on noise type
	}
	// Propagate gradients to the parameters (as directed by backward prop).
//	if (!_if_pass) {
  		this->param_propagate_down_.resize(this->blobs_.size(), true);
//	}
}

template<typename Dtype>
void NoiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top){
	// maybe check won't apply under some cases
	CHECK_LE(top[0]->count(), bottom[0]->count())
		<< "Top blob cannot produce more info than does bottom blob.";
	// setup the memory space for top layer
	top[0]->Reshape(bottom[0]->shape());
	// second top layer is to store weights
	top[1]->Reshape(this->blobs_[0]->shape());
	rand_noise_.Reshape(bottom[0]->num(), bottom[0]->channels(),
    	bottom[0]->height(), bottom[0]->width());
}

template<typename Dtype>
void NoiseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
	const vector<Blob<Dtype>*>& top) {

	const int noise_size = (bottom[0]-> height()) * (bottom[0]->width());
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_noisy_data = top[0]->mutable_cpu_data();
	Dtype* top_param_data = top[1]->mutable_cpu_data();

	for (int i = 0; i < bottom[0]->num(); i++) {
		for (int j = 0; j < bottom[0]->channels(); j++) {
			// allocate memory space for random noise
			Dtype* rnoise = (Dtype*) calloc(noise_size, sizeof(Dtype));
			switch(_noise_type) {
				case NoiseParameter_NoiseType_GAUSSIAN: {
					Dtype sd = this->blobs_[0]->data_at(i, j, 0, 0);
					Dtype abs_sd = std::abs(sd);
					if (abs_sd < _min_sd) {
						LOG(INFO) << "Standard deviation out of range: " << sd;
						caffe_set(1, _min_sd, this->blobs_[0]->mutable_cpu_data() + this->blobs_[0]->offset(i, j));
						caffe_rng_gaussian(noise_size, _param1, _scale * _min_sd, rnoise);		
					}
					else if (abs_sd > _max_sd) {
						LOG(INFO) << "Standard deviation out of range: " << sd;
						caffe_set(1, _max_sd, this->blobs_[0]->mutable_cpu_data() + this->blobs_[0]->offset(i, j));
						caffe_rng_gaussian(noise_size, _param1, _scale * _max_sd, rnoise);	
					}
					else {
						caffe_rng_gaussian(noise_size, _param1, _scale * abs_sd, rnoise);
					}
				} 	break;
				case NoiseParameter_NoiseType_POISSON: {
					std::poisson_distribution<int> 
						_poisson_engine(_scale * (this->blobs_[0]->data_at(i, j, 0, 0)));
					for (int s = 0; s < noise_size; s++) {
						rnoise[s] = _poisson_engine(_generator) / _param2;
					}
				}	break;
				case NoiseParameter_NoiseType_UNIFORM: {
					caffe_rng_uniform(noise_size, (Dtype) 0., 
						_scale * (this->blobs_[0]->data_at(i, j, 0, 0)), rnoise);
				}	break;
			}	// should care about noise seed problem
			caffe_add(noise_size, bottom_data + bottom[0]->offset(i, j), 
				rnoise, top_noisy_data + top[0]->offset(i, j));
			free(rnoise);
		}	// add different noise to different inputs and channels
	}	
	caffe_copy(top[1]->count(), this->blobs_[0].get()->cpu_data(), top_param_data);
	// pushing the learnable noise parameters to output for EnergyLossLayer
}

template<typename Dtype>
void NoiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	if (_if_pass)
		return;

	vector<int> blob_shape_ = top[0]->shape();
	const int col_num_ = blob_shape_[2] * blob_shape_[3];

//	const Dtype* top_data_diff = top[0]->cpu_diff();
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* top_energy_diff = top[1]->cpu_diff();
//	Dtype* scaled_data_diff = (Dtype*) calloc(top);
//	const Dtype* sd_weight = this->blobs_[0]->cpu_data();
	Dtype* sd_weight_diff = this->blobs_[0]->mutable_cpu_diff();

	if (propagate_down[0] && !_if_pass) {

		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		// set bottom_diff = top_diff x sd_weight
		for (int n = 0; n < blob_shape_[0]; n++) {
			for (int c = 0; c < blob_shape_[1]; c++) {
				caffe_cpu_axpby<Dtype>(col_num_, this->blobs_[0]->data_at(n, c, 0, 0),
					top_diff + top[0]->offset(n, c), (Dtype)0., 
					bottom_diff + bottom[0]->offset(n, c));
			}
		}
		if (this->param_propagate_down_[0]) {
			// fill the weight_diff array with 0 first for computation
			caffe_set(this->blobs_[0]->count(), Dtype(0), sd_weight_diff);
			// next update the noise parameters
			// weight_diff = top_diff x bottom_data
			for (int n = 0; n < blob_shape_[0]; n++) {
				for (int c = 0; c < blob_shape_[1]; c++) {
					Dtype weight_val = caffe_cpu_dot(this->blobs_[0]->count(), 
						top_diff + top[0]->offset(n, c), bottom_data + bottom[0]->offset(n,c));
					caffe_set(1, weight_val, sd_weight_diff + this->blobs_[0]->offset(n, c));
				}
			}
			caffe_scal(this->blobs_[0]->count(), (const Dtype) _diff_scale, sd_weight_diff);
		}	
	}
	// else {
	// //	caffe_copy(top[0]->count(), top_diff, bottom[0]->mutable_cpu_diff());
	// 	if (propagate_down[0])
 //        	caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
 //    	}
	// }
}

#ifdef CPU_ONLY
STUB_GPU(NoiseLayer);
#endif

INSTANTIATE_CLASS(NoiseLayer);
REGISTER_LAYER_CLASS(Noise);
}	// namespace caffe