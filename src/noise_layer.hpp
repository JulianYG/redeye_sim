#ifndef CAFFE_NOISE_LAYER_HPP_
#define CAFFE_NOISE_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>
#include <random>
#include <iostream>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/** 
 * @brief A parasitic layer that add noise to incoming data.
 *
 */
  template <typename Dtype>
  class NoiseLayer : public Layer<Dtype> {
   public:
    explicit NoiseLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
      const vector<Blob<Dtype>*>& top);
    // has to inherit Reshape() from layer.hpp
    virtual inline const char* type() const { return "Noise"; }
    virtual inline int MinBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 2; }

   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    bool _if_pass;
    NoiseParameter_NoiseType _noise_type;
    Blob<Dtype> rand_noise_;
    // parameters for noises
    Dtype _param1, _param2;
    Dtype _min_sd, _max_sd;
    // initialized default random number generator
    std::default_random_engine _generator;
    Dtype _scale, _diff_scale;
  };
}

#endif // CAFFE_NOISE_LAYER_HPP_