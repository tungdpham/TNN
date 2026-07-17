/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/transpose_layer.hpp"

#include <cstddef>

#include "nn/stats/stats.hpp"

namespace tunx {

TransposeLayerImpl::TransposeLayerImpl(size_t dim0, size_t dim1, const std::string &name)
    : SISOLayerImpl(name),
      dim0_(dim0),
      dim1_(dim1) {}

Tensor TransposeLayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  if (dim0_ >= input.dims() || dim1_ >= input.dims()) {
    throw std::runtime_error("TransposeLayerImpl: dim0 or dim1 out of bounds");
  }

  Vec<size_t> out_shape = input.shape();
  std::swap(out_shape[dim0_], out_shape[dim1_]);
  Tensor output = get_tensor(out_shape, input.dtype());

  TransposeStats stats;
  stats.ndim = input.dims();
  stats.dim0 = dim0_;
  stats.dim1 = dim1_;
  for (size_t i = 0; i < input.dims(); ++i) stats.shape[i] = input.dim(i);

  DTypeDesc type_desc{io_dtype_, param_dtype_, compute_dtype_};

  WorkspaceReq ws_req = engine_->query_transpose_graph(this->backend_handle_, stats, type_desc);

  Tensor ws = get_tensor({ws_req.fwd_workspace}, io_dtype_);

  engine_->transpose(backend_handle_, stats, input.data_as<void>(), output.data_as<void>(),
                     ws.data_as<void>(), type_desc);

  return output;
}

Tensor TransposeLayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  if (dim0_ >= grad_output.dims() || dim1_ >= grad_output.dims()) {
    throw std::runtime_error("TransposeLayerImpl: dim0 or dim1 out of bounds");
  }

  Vec<size_t> in_shape = grad_output.shape();
  std::swap(in_shape[dim0_], in_shape[dim1_]);
  Tensor grad_input = get_tensor(in_shape, grad_output.dtype());

  TransposeStats stats;
  stats.ndim = grad_output.dims();
  stats.dim0 = dim0_;
  stats.dim1 = dim1_;
  for (size_t i = 0; i < grad_output.dims(); ++i) stats.shape[i] = grad_output.dim(i);

  DTypeDesc type_desc{io_dtype_, param_dtype_, compute_dtype_};

  WorkspaceReq ws_req = engine_->query_transpose_graph(this->backend_handle_, stats, type_desc);

  Tensor ws = get_tensor({ws_req.bwd_workspace}, io_dtype_);

  engine_->transpose(backend_handle_, stats, grad_output.data_as<void>(),
                     grad_input.data_as<void>(), ws.data_as<void>(), type_desc);

  return grad_input;
}

Vec<size_t> TransposeLayerImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  if (dim0_ >= input_shape.size() || dim1_ >= input_shape.size()) {
    throw std::runtime_error("TransposeLayerImpl: dim0 or dim1 out of bounds");
  }
  Vec<size_t> out_shape = input_shape;
  std::swap(out_shape[dim0_], out_shape[dim1_]);
  return out_shape;
}

LayerConfig TransposeLayerImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("dim0", dim0_);
  config.set("dim1", dim1_);
  return config;
}

std::shared_ptr<TransposeLayerImpl> TransposeLayerImpl::create_from_config(
    const LayerConfig &config) {
  size_t dim0;
  size_t dim1;
  dim0 = config.get<size_t>("dim0");
  dim1 = config.get<size_t>("dim1");
  return std::make_shared<TransposeLayerImpl>(dim0, dim1, config.name);
}

}  // namespace tunx
