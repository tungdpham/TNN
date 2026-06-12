/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/div_layer.hpp"

#include <stdexcept>

#include "ops/ops.hpp"
#include "type/type.hpp"

namespace synet {

Vec<Vec<size_t>> DivLayerImpl::output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
  if (input_shapes.size() != 2) {
    throw std::runtime_error("DivLayerImpl: expected exactly 2 inputs");
  }
  if (input_shapes[0] != input_shapes[1]) {
    throw std::runtime_error("DivLayerImpl: both inputs must have the same shape");
  }
  return {input_shapes[0]};
}

Vec<Tensor> DivLayerImpl::forward_impl(const Vec<Tensor> &inputs, size_t mb_id) {
  if (inputs.size() != 2) {
    throw std::runtime_error("DivLayerImpl: expected exactly 2 inputs");
  }
  const Tensor &a = inputs[0];
  const Tensor &b = inputs[1];

  if (a.shape() != b.shape()) {
    throw std::runtime_error("DivLayerImpl: both inputs must have the same shape");
  }

  Tensor output = get_tensor(a.shape(), io_dtype_);
  const size_t n = a.size();

  // Cache inputs for backward pass
  if (this->is_training_) {
    this->set_immutable_cache(mb_id, "a", a);
    this->set_immutable_cache(mb_id, "b", b);
  }

  DISPATCH_DTYPE(a.data_type(), T, {
    ops::div<T>(a.data_ptr(), b.data_ptr(), output.data_ptr(), n, this->flow_handle_);
  });

  return {output};
}

Vec<Tensor> DivLayerImpl::backward_impl(const Vec<Tensor> &grad_outputs, size_t mb_id) {
  if (grad_outputs.size() != 1) {
    throw std::runtime_error("DivLayerImpl: expected exactly 1 grad output");
  }
  const Tensor &grad_out = grad_outputs[0];
  const Tensor &a = this->get_immutable_cache(mb_id, "a");
  const Tensor &b = this->get_immutable_cache(mb_id, "b");
  const size_t n = grad_out.size();

  // grad_a = grad_out / b
  // grad_b = -(grad_out * a) / b^2
  Tensor grad_a = get_tensor(grad_out.shape(), this->io_dtype_);
  Tensor grad_b = get_tensor(grad_out.shape(), this->io_dtype_);

  DISPATCH_DTYPE(grad_out.data_type(), T, {
    // grad_a = grad_out / b
    ops::div<T>(grad_out.data_ptr(), b.data_ptr(), grad_a.data_ptr(), n, this->flow_handle_);

    // grad_b = -(grad_out * a) / b^2
    // Step 1: b_sq = b * b
    Tensor b_sq = get_tensor(grad_out.shape(), this->io_dtype_);
    ops::mul<T>(b.data_ptr(), b.data_ptr(), b_sq.data_ptr(), n, this->flow_handle_);

    // Step 2: numerator = grad_out * a
    Tensor numerator = get_tensor(grad_out.shape(), this->io_dtype_);
    ops::mul<T>(grad_out.data_ptr(), a.data_ptr(), numerator.data_ptr(), n, this->flow_handle_);

    // Step 3: grad_b = numerator / b_sq
    ops::div<T>(numerator.data_ptr(), b_sq.data_ptr(), grad_b.data_ptr(), n, this->flow_handle_);

    // Step 4: negate
    ops::mul_scalar<T>(grad_b.data_ptr(), static_cast<T>(-1), grad_b.data_ptr(), n,
                       this->flow_handle_);
  });

  return {grad_a, grad_b};
}

LayerConfig DivLayerImpl::get_config() const {
  LayerConfig config;
  config.type = TYPE_NAME;
  config.name = this->name_;
  return config;
}

std::shared_ptr<DivLayerImpl> DivLayerImpl::create_from_config(const LayerConfig &config) {
  return std::make_shared<DivLayerImpl>(config.name.empty() ? "div" : config.name);
}

}  // namespace synet
