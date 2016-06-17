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
	
	Dtype maxVal = bottom_data[0];
	Dtype minVal = bottom_data[0];

	for (int i = 0; i < bottom[0]->count(); ++i) {
		if (minVal > bottom_data[i]) {
			minVal = bottom_data[i];
		} else if (maxVal < bottom_data[i]) {
			maxVal = bottom_data[i];
		}
	}
	Dtype bin = (maxVal - minVal) / Dtype(pow(2, _bit_num) - 1.);
	
	// for (int i = 0; i < top[0]->count(); ++i) {
	// 	top_data[i] = bin * floor((bottom_data[i] - minVal)/bin) + minVal;
	// }

	// cublasHandle_t handle;
	// cublasCreate(&handle);
	// int minIdx, maxIdx;
	// cublasIsamax(handle, bottom[0]->count(), (const float *)bottom_data, 1, &maxIdx);
	// cublasIsamin(handle, bottom[0]->count(), (const float *)bottom_data, 1, &minIdx);
	// cublasDestroy(handle);

	// Dtype minVal = bottom[0]->cpu_data()[minIdx];
	// Dtype maxVal = bottom[0]->cpu_data()[maxIdx];
	// Dtype bin = (maxVal - minVal) / Dtype(pow(2, _bit_num) - 1.);
	// an array of minimums for calculation purpose
	Dtype* tmp_data = static_cast<Dtype*>(min_array_.mutable_cpu_data());	

	caffe_gpu_set(bottom[0]->count(), minVal, tmp_data);
	caffe_gpu_axpby(bottom[0]->count(), Dtype(1/bin), bottom_data, 
		- minVal / bin, tmp_data);
	caffe_gpu_set(bottom[0]->count(), Dtype(1.), top_data);
	// have to access cpu data for use of floor
	for (int i = 0; i < bottom[0]->count(); ++i) {
		tmp_data[i] = floor(tmp_data[i]);
	}
	LOG(INFO) << minVal;
	LOG(INFO) << maxVal;
	LOG(INFO) << bin;
	caffe_gpu_axpby(bottom[0]->count(), bin, tmp_data, minVal, top_data);
}
template<typename Dtype>
void QuantizationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    // DO NOTHING
}

INSTANTIATE_LAYER_GPU_FUNCS(QuantizationLayer);

}	// namespace caffe