/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/sdpa_layer.hpp"

#include <cmath>
#include <stdexcept>

namespace tunx {

SDPALayerImpl::SDPALayerImpl(float attn_scale, bool is_causal, const std::string &name)
    : attn_scale_(attn_scale),
      is_causal_(is_causal),
      is_training_(false) {
  this->name_ = name;
}

SDPALayerImpl::~SDPALayerImpl() = default;

LayerConfig SDPALayerImpl::get_config() const {
  LayerConfig config;
  config.type = TYPE_NAME;
  config.name = this->name_;
  config.set("attn_scale", attn_scale_);
  config.set("is_causal", is_causal_);
  return config;
}

Vec<Vec<size_t>> SDPALayerImpl::output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
  if (input_shapes.size() != 3) {
    throw std::runtime_error("SDPALayerImpl: expected exactly 3 inputs (Q, K, V)");
  }

  // All inputs should have same shape: (B, H, S, D)
  const auto &q_shape = input_shapes[0];
  const auto &k_shape = input_shapes[1];
  const auto &v_shape = input_shapes[2];

  if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4) {
    throw std::runtime_error("SDPALayerImpl: inputs must be 4D (B, H, S, D)");
  }

  if (q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0]) {
    throw std::runtime_error("SDPALayerImpl: batch size mismatch");
  }
  if (q_shape[1] != k_shape[1] || q_shape[1] != v_shape[1]) {
    throw std::runtime_error("SDPALayerImpl: number of heads mismatch");
  }
  if (q_shape[2] != k_shape[2] || q_shape[2] != v_shape[2]) {
    throw std::runtime_error("SDPALayerImpl: sequence length mismatch");
  }
  if (q_shape[3] != k_shape[3] || q_shape[3] != v_shape[3]) {
    throw std::runtime_error("SDPALayerImpl: head dim mismatch");
  }

  // Output shape same as Q: (B, H, S, D)
  return {q_shape};
}

Vec<Tensor> SDPALayerImpl::forward_impl(const Vec<Tensor> &inputs, Residuals &residuals) {
  if (inputs.size() != 3) {
    throw std::runtime_error("SDPALayerImpl: expected exactly 3 inputs (Q, K, V)");
  }

  const Tensor &q = inputs[0];
  const Tensor &k = inputs[1];
  const Tensor &v = inputs[2];

  if (q.dims() != 4) {
    throw std::runtime_error("SDPALayerImpl: Q must be 4D (B, H, S, D)");
  }

  const auto &q_shape = q.shape();
  size_t batch_size = q_shape[0];
  size_t num_heads = q_shape[1];
  size_t seq_len = q_shape[2];
  size_t head_dim = q_shape[3];

  // Validate K and V shapes
  {
    const auto &k_shape = k.shape();
    const auto &v_shape = v.shape();
    if (k_shape != q_shape || v_shape != q_shape) {
      throw std::runtime_error("SDPALayerImpl: Q, K, V must have same shape");
    }
  }

  Tensor output = get_tensor(q_shape, io_dtype_);

  if (this->is_training_) {
    residuals["q"] = q;
    residuals["k"] = k;
    residuals["v"] = v;
    residuals["output"] = output;
  }

  AttentionStats stats{
      .batch_size = batch_size,
      .num_heads = num_heads,
      .seq_len = seq_len,
      .head_dim = head_dim,
      .attn_scale = attn_scale_,
      .is_causal = is_causal_,
  };
  
  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  void* backend_handle = engine_->create_backend_handle();
  WorkspaceReq req = engine_->query_sdpa_graph(backend_handle, stats, type_desc);

  Tensor workspace = this->get_tensor({req.fwd_workspace}, DType_t::BYTE);
  
  // Use a stats_tensor to hold statistics for backward pass
  // CPU uses [B, H, S, S] float/double
  // CuDNN uses [B, H, S, 1] float32
  // We allocate conservatively.
  size_t stats_elements = std::max(batch_size * num_heads * seq_len * seq_len,
                                   batch_size * num_heads * seq_len * 1);
  Tensor stats_tensor = this->get_tensor({stats_elements}, io_dtype_);

  if (this->is_training_) {
    residuals["stats"] = stats_tensor;
  }

  engine_->sdpa_fwd(backend_handle, stats, q.data_as<void>(), k.data_as<void>(),
                    v.data_as<void>(), output.data_as<void>(), stats_tensor.data_as<void>(),
                    workspace.data_as<void>(), type_desc);

  return {output};
}

Vec<Tensor> SDPALayerImpl::backward_impl(const Vec<Tensor> &grad_outputs, Residuals &residuals) {
  if (grad_outputs.size() != 1) {
    throw std::runtime_error("SDPALayerImpl: expected exactly 1 grad output");
  }

  const Tensor &grad_output = grad_outputs[0];

  const Tensor &q = residuals["q"];
  const Tensor &k = residuals["k"];
  const Tensor &v = residuals["v"];
  const Tensor &output = residuals["output"];
  const Tensor &stats_tensor = residuals["stats"];

  const Vec<size_t> &q_shape = q.shape();
  size_t batch_size = q_shape[0];
  size_t num_heads = q_shape[1];
  size_t seq_len = q_shape[2];
  size_t head_dim = q_shape[3];

  Tensor grad_q = get_tensor(q_shape, this->io_dtype_);
  Tensor grad_k = get_tensor(q_shape, this->io_dtype_);
  Tensor grad_v = get_tensor(q_shape, this->io_dtype_);

  AttentionStats stats{
      .batch_size = batch_size,
      .num_heads = num_heads,
      .seq_len = seq_len,
      .head_dim = head_dim,
      .attn_scale = attn_scale_,
      .is_causal = is_causal_,
  };
  
  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  void* backend_handle = engine_->create_backend_handle();
  WorkspaceReq req = engine_->query_sdpa_graph(backend_handle, stats, type_desc);

  Tensor workspace = this->get_tensor({req.bwd_workspace}, DType_t::BYTE);

  engine_->sdpa_bwd(backend_handle, stats, q.data_as<void>(), k.data_as<void>(),
                    v.data_as<void>(), output.data_as<void>(), grad_output.data_as<void>(),
                    stats_tensor.data_as<void>(), grad_q.data_as<void>(), grad_k.data_as<void>(),
                    grad_v.data_as<void>(), workspace.data_as<void>(), type_desc);

  return {grad_q, grad_k, grad_v};
}

std::shared_ptr<SDPALayerImpl> SDPALayerImpl::create_from_config(const LayerConfig &config) {
  float attn_scale = config.get<float>("attn_scale", 1.0f);
  bool is_causal = config.get<bool>("is_causal", false);
  std::string name = config.name.empty() ? "sdpa" : config.name;

  return std::make_shared<SDPALayerImpl>(attn_scale, is_causal, name);
}

}  // namespace tunx
