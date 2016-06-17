#ifndef CAFFE_ENERGY_LOSS_LAYER_HPP_
#define CAFFE_ENERGY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
	
	/**
	 * Computes the energy loss based on the noise type and magnitude. Used
	 * to train caffe for a better combination of both prediction accuracy
	 * and noise tolerance, which reduces processing energy.
	 */

	template <typename Dtype>
	class EnergyLossLayer: public Layer<Dtype> {
	  public:
	    explicit EnergyLossLayer(const LayerParameter& param):Layer<Dtype>(param) {}
	    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top);
	    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top);
	    virtual inline int ExactNumTopBlobs() const { return 1; }
	    virtual inline const char* type() const { return "EnergyLoss"; }
	    virtual inline int MinBottomBlobs() const { return 1; }
	    virtual inline int ExactNumBottomBlobs() const { return -1; }

	  protected:
	    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top);
	    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top);

	    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	    EnergyLossParameter_ModelType _model;
	    Dtype _alpha;
	    vector<int> _param_types;
	};

}

#endif // CAFFE_ENERGY_LOSS_LAYER_HPP_
