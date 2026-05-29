/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/mul_layer.hpp"

#include <stdexcept>

#include "ops/ops.hpp"
#include "type/type.hpp"

namespace tnn {

Vec<Vec<size_t>> MulLayerImpl::output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
  if (input_shapes.size() != 2) {
    throw std::runtime_error("MulLayerImpl: expected exactly 2 inputs");
  }
  if (input_shapes[0] != input_shapes[1]) {
    throw std::runtime_error("MulLayerImpl: both inputs must have the same shape");
  }
  return {input_shapes[0]};
}

Vec<Tensor> MulLayerImpl::forward_impl(const Vec<ConstTensor> &inputs, size_t mb_id) {
  if (inputs.size() != 2) {
    throw std::runtime_error("MulLayerImpl: expected exactly 2 inputs");
  }
  const ConstTensor &a = inputs[0];
  const ConstTensor &b = inputs[1];

  if (a->shape() != b->shape()) {
    throw std::runtime_error("MulLayerImpl: both inputs must have the same shape");
  }

  Tensor output = get_output_tensor(a->shape());
  const size_t n = a->size();

  // Cache inputs for backward pass
  if (this->is_training_) {
    this->set_immutable_cache(mb_id, "a", a);
    this->set_immutable_cache(mb_id, "b", b);
  }

  DISPATCH_DTYPE(a->data_type(), T, {
    ops::mul<T>(a->data_ptr(), b->data_ptr(), output->data_ptr(), n, this->flow_handle_);
  });

  return {output};
}

Vec<Tensor> MulLayerImpl::backward_impl(const Vec<ConstTensor> &grad_outputs, size_t mb_id) {
  if (grad_outputs.size() != 1) {
    throw std::runtime_error("MulLayerImpl: expected exactly 1 grad output");
  }
  const ConstTensor &grad_out = grad_outputs[0];
  const ConstTensor &a = this->get_immutable_cache(mb_id, "a");
  const ConstTensor &b = this->get_immutable_cache(mb_id, "b");
  const size_t n = grad_out->size();

  // grad_a = grad_out * b,  grad_b = grad_out * a
  Tensor grad_a = get_workspace(grad_out->shape(), this->io_dtype_);
  Tensor grad_b = get_workspace(grad_out->shape(), this->io_dtype_);

  DISPATCH_DTYPE(grad_out->data_type(), T, {
    ops::mul<T>(grad_out->data_ptr(), b->data_ptr(), grad_a->data_ptr(), n, this->flow_handle_);
    ops::mul<T>(grad_out->data_ptr(), a->data_ptr(), grad_b->data_ptr(), n, this->flow_handle_);
  });

  return {grad_a, grad_b};
}

LayerConfig MulLayerImpl::get_config() const {
  LayerConfig config;
  config.type = TYPE_NAME;
  config.name = this->name_;
  return config;
}

std::unique_ptr<MulLayerImpl> MulLayerImpl::create_from_config(const LayerConfig &config) {
  return std::make_unique<MulLayerImpl>(config.name.empty() ? "mul" : config.name);
}

}  // namespace tnn
