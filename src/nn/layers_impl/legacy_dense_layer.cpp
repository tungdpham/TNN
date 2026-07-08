/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/legacy_dense_layer.hpp"

#include "nn/engines/iengine.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>


#include "tensor/tensor.hpp"
#include "tensor/tensor_ops.hpp"
#include "type/type.hpp"

namespace tunx {

LegacyDenseLayerImpl::LegacyDenseLayerImpl(size_t input_features, size_t output_features,
                                           bool use_bias, const std::string &name)
    : SISOLayerImpl(name),
      input_features_(input_features),
      output_features_(output_features),
      use_bias_(use_bias) {}

void LegacyDenseLayerImpl::init_impl() {
  float stddev = static_cast<float>(1.0 / std::sqrt(static_cast<double>(input_features_)));
  long long seed = this->use_seed_ ? this->srand_seed_
                                   : std::chrono::system_clock::now().time_since_epoch().count();

  fill_normal(weights_, 0, stddev, seed);

  if (use_bias_) {
    fill_normal(bias_, 0, stddev, seed);
  }

  fill(grad_weights_, 0.0f);
  if (use_bias_) {
    fill(grad_bias_, 0.0f);
  }
}

Tensor LegacyDenseLayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  const Vec<size_t> &in_shape = input.shape();
  size_t last_dim = in_shape.back();
  size_t batch_size = 1;
  for (size_t i = 0; i < in_shape.size() - 1; ++i) {
    batch_size *= in_shape[i];
  }

  if (last_dim != input_features_) {
    std::cerr << "Input last dim: " << last_dim << " features, expected: " << input_features_
              << " features" << std::endl;
    throw std::invalid_argument("Input feature size mismatch in LegacyDenseLayerImpl");
  }

  if (this->is_training_) {
    residuals["input"] = input;
  }

  Vec<size_t> out_shape = in_shape;
  out_shape.back() = output_features_;
  Tensor output = get_tensor(out_shape, io_dtype_);

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  engine_->legacy_dense_fwd(backend_handle_, input.data_as<void>(), weights_.data_as<void>(),
                            output.data_as<void>(), batch_size, input_features_, output_features_,
                            type_desc);

  if (use_bias_) {
    engine_->legacy_dense_add_bias(backend_handle_, output.data_as<void>(), bias_.data_as<void>(),
                                   batch_size, output_features_, type_desc);
  }

  return output;
}

Tensor LegacyDenseLayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  if (grad_output.shape().back() != output_features_) {
    throw std::invalid_argument("Gradient feature size mismatch in LegacyDenseLayerImpl");
  }
  const Tensor &input = residuals["input"];
  const Vec<size_t> &in_shape = input.shape();
  size_t batch_size = 1;
  for (size_t i = 0; i < in_shape.size() - 1; ++i) {
    batch_size *= in_shape[i];
  }

  Tensor grad_input = get_tensor(input.shape(), io_dtype_);

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  engine_->legacy_dense_wgrad(backend_handle_, input.data_as<void>(), grad_output.data_as<void>(),
                              grad_weights_.data_as<void>(), batch_size, input_features_,
                              output_features_, type_desc);

  if (use_bias_) {
    engine_->legacy_dense_bgrad(backend_handle_, grad_output.data_as<void>(),
                                grad_bias_.data_as<void>(), batch_size, output_features_,
                                type_desc);
  }

  engine_->legacy_dense_dgrad(backend_handle_, grad_output.data_as<void>(),
                              weights_.data_as<void>(), grad_input.data_as<void>(), batch_size,
                              input_features_, output_features_, type_desc);

  return grad_input;
}



LayerConfig LegacyDenseLayerImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("input_features", input_features_);
  config.set("output_features", output_features_);
  config.set("use_bias", use_bias_);
  return config;
}

Vec<size_t> LegacyDenseLayerImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  if (input_shape.empty()) {
    throw std::runtime_error("LegacyDenseLayerImpl::compute_output_shape: Input shape is empty.");
  }
  Vec<size_t> out_shape = input_shape;
  out_shape.back() = output_features_;
  return out_shape;
}

std::shared_ptr<LegacyDenseLayerImpl> LegacyDenseLayerImpl::create_from_config(
    const LayerConfig &config) {
  size_t input_features = config.get<size_t>("input_features");
  size_t output_features = config.get<size_t>("output_features");
  bool use_bias = config.get<bool>("use_bias");

  return std::make_shared<LegacyDenseLayerImpl>(input_features, output_features, use_bias,
                                                config.name);
}

}  // namespace tunx
