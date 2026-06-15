/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/sub_layer.hpp"

#include <stdexcept>

#include "ops/ops.hpp"
#include "type/type.hpp"

namespace synet {

Vec<Vec<size_t>> SubLayerImpl::output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
  if (input_shapes.size() != 2) {
    throw std::runtime_error("SubLayerImpl: expected exactly 2 inputs");
  }
  if (input_shapes[0] != input_shapes[1]) {
    throw std::runtime_error("SubLayerImpl: both inputs must have the same shape");
  }
  return {input_shapes[0]};
}

Vec<Tensor> SubLayerImpl::forward_impl(const Vec<Tensor> &inputs, Residuals &residuals) {
  if (inputs.size() != 2) {
    throw std::runtime_error("SubLayerImpl: expected exactly 2 inputs");
  }
  const Tensor &a = inputs[0];
  const Tensor &b = inputs[1];

  if (a.shape() != b.shape()) {
    throw std::runtime_error("SubLayerImpl: both inputs must have the same shape");
  }

  Tensor output = get_tensor(a.shape(), io_dtype_);
  size_t n = a.size();

  DISPATCH_DTYPE(a.data_type(), T, {
    ops::sub<T>(a.data_ptr(), b.data_ptr(), output.data_ptr(), n, this->flow_handle_);
  });

  return {output};
}

Vec<Tensor> SubLayerImpl::backward_impl(const Vec<Tensor> &grad_outputs, Residuals &residuals) {
  if (grad_outputs.size() != 1) {
    throw std::runtime_error("SubLayerImpl: expected exactly 1 grad output");
  }
  const Tensor &grad_out = grad_outputs[0];
  size_t n = grad_out.size();

  // grad_a = grad_out, grad_b = -grad_out
  Tensor grad_a = get_tensor(grad_out.shape(), this->io_dtype_);
  Tensor grad_b = get_tensor(grad_out.shape(), this->io_dtype_);

  DISPATCH_DTYPE(grad_out.data_type(), T, {
    ops::copy<T>(grad_out.data_ptr(), grad_a.data_ptr(), n, this->flow_handle_);
    ops::mul_scalar<T>(grad_out.data_ptr(), static_cast<T>(-1), grad_b.data_ptr(), n,
                       this->flow_handle_);
  });

  return {grad_a, grad_b};
}

LayerConfig SubLayerImpl::get_config() const {
  LayerConfig config;
  config.type = TYPE_NAME;
  config.name = this->name_;
  return config;
}

std::shared_ptr<SubLayerImpl> SubLayerImpl::create_from_config(const LayerConfig &config) {
  return std::make_shared<SubLayerImpl>(config.name.empty() ? "sub" : config.name);
}

}  // namespace synet
