#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/layers/quantization_layer.hpp"

namespace caffe {

template<typename Dtype>
void QuantizationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
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
template<typename Dtype>
void QuantizationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    // DO NOTHING
}

INSTANTIATE_LAYER_GPU_FUNCS(QuantizationLayer);

}	// namespace caffe