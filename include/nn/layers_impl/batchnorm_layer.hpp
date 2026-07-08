/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "nn/siso_layer.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

class BatchNormLayerImpl : public SISOLayerImpl {
private:
  size_t num_features_;
  float epsilon_;
  float momentum_;
  bool affine_;
  bool use_relu_;

  Tensor gamma_;
  Tensor beta_;
  Tensor grad_gamma_;
  Tensor grad_beta_;

  Tensor running_mean_;
  Tensor running_var_;
  Tensor grad_dummy_mean_;
  Tensor grad_dummy_var_;

  void init_impl() override;
  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  explicit BatchNormLayerImpl(size_t num_features, float epsilon = 1e-5f, float momentum = 0.1f,
                              bool affine = true, bool use_relu = false,
                              const std::string &name = "batchnorm");
  ~BatchNormLayerImpl() override;

  static constexpr const char *TYPE_NAME = "batchnorm";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  static std::shared_ptr<BatchNormLayerImpl> create_from_config(const LayerConfig &config);
  Vec<ParamDescriptor> param_descriptors() override {
    Vec<ParamDescriptor> descriptors;
    auto gamma_desc = ParamDescriptor{
        param_dtype_,
        {num_features_},
        &gamma_,
        &grad_gamma_,
    };
    descriptors.push_back(gamma_desc);
    auto beta_desc = ParamDescriptor{
        param_dtype_,
        {num_features_},
        &beta_,
        &grad_beta_,
    };
    descriptors.push_back(beta_desc);
    auto running_mean_desc = ParamDescriptor{
        param_dtype_,
        {num_features_},
        &running_mean_,
        &grad_dummy_mean_,
    };
    descriptors.push_back(running_mean_desc);
    auto running_var_desc = ParamDescriptor{
        param_dtype_,
        {num_features_},
        &running_var_,
        &grad_dummy_var_,
    };
    descriptors.push_back(running_var_desc);
    return descriptors;
  }
  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
};

class BatchNormLayer : public LayerRef<BatchNormLayerImpl> {
public:
  BatchNormLayer(size_t num_features, float epsilon = 1e-5f, float momentum = 0.1f,
                 bool affine = true, bool use_relu = false, const std::string &name = "batchnorm")
      : LayerRef(std::make_shared<BatchNormLayerImpl>(num_features, epsilon, momentum, affine,
                                                      use_relu, name)) {}

  using LayerRef<BatchNormLayerImpl>::LayerRef;
};

}  // namespace tunx
