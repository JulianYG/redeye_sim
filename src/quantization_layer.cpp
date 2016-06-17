#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/quantization_layer.hpp"

namespace caffe {

template<typename Dtype> 
void QuantizationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {

	CHECK_EQ(bottom.size(), 1) << "Can only have one bottom blob input.";
	// checks top size in hpp files
	QuantizationParameter quantization_param = this->layer_param_.quantization_param();
	_bit_num = quantization_param.bit_num();
}

template<typename Dtype>
void QuantizationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
	const vector<Blob<Dtype>*>& top) {

	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	
	Dtype mx = bottom_data[0];
	Dtype mn = bottom_data[0];

	for (int i = 0; i < bottom[0]->count(); ++i) {
		if (mn > bottom_data[i]) {
			mn = bottom_data[i];
		} else if (mx < bottom_data[i]) {
			mx = bottom_data[i];
		}
	}
	Dtype bin = (mx - mn) / Dtype(pow(2, _bit_num) - 1.);
	for (int i = 0; i < top[0]->count(); ++i) {
		top_data[i] = bin * floor((bottom_data[i] - mn)/bin) + mn;
	}
}

#ifdef CPU_ONLY
STUB_GPU(QuantizationLayer);
#endif

INSTANTIATE_CLASS(QuantizationLayer);
REGISTER_LAYER_CLASS(Quantization);
}	// namespace caffe