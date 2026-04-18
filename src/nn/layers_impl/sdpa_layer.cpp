/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/sdpa_layer.hpp"

#include <cmath>
#include <stdexcept>

#include "device/task.hpp"
#include "nn/layers_impl/cpu/sdpa_ops.hpp"
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/sdpa_ops.hpp"
#endif
#ifdef USE_CUDNN
#include "cuda/cudnn/common.hpp"
#include "device/cuda/cuda_context.hpp"
#include "nn/blocks_impl/common/flash_attention.hpp"
#include "nn/blocks_impl/cuda/cudnn_flash_attention_ops.hpp"
#endif

namespace tnn {

SDPALayer::SDPALayer(float attn_scale, bool is_causal, const std::string &name)
    : attn_scale_(attn_scale),
      is_causal_(is_causal),
      is_training_(false) {
  this->name_ = name;
}

SDPALayer::~SDPALayer() {
#ifdef USE_CUDNN
  for (auto &kv : fe_handle_cache_) {
    if (kv.second) {
      cuda::cudnn_flash_attention::destroy_fe_handle(
          static_cast<cuda::cudnn_flash_attention::feHandle_t *>(kv.second));
    }
  }
  fe_handle_cache_.clear();
  for (auto &kv : stats_cache_) {
    if (kv.second) {
      delete static_cast<AttentionStats *>(kv.second);
    }
  }
  stats_cache_.clear();
#endif
}

LayerConfig SDPALayer::get_config() const {
  LayerConfig config;
  config.type = TYPE_NAME;
  config.name = this->name_;
  config.set("attn_scale", attn_scale_);
  config.set("is_causal", is_causal_);
  return config;
}

Vec<Vec<size_t>> SDPALayer::output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
  if (input_shapes.size() != 3) {
    throw std::runtime_error("SDPALayer: expected exactly 3 inputs (Q, K, V)");
  }

  // All inputs should have same shape: (B, H, S, D)
  const auto &q_shape = input_shapes[0];
  const auto &k_shape = input_shapes[1];
  const auto &v_shape = input_shapes[2];

  if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4) {
    throw std::runtime_error("SDPALayer: inputs must be 4D (B, H, S, D)");
  }

  if (q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0]) {
    throw std::runtime_error("SDPALayer: batch size mismatch");
  }
  if (q_shape[1] != k_shape[1] || q_shape[1] != v_shape[1]) {
    throw std::runtime_error("SDPALayer: number of heads mismatch");
  }
  if (q_shape[2] != k_shape[2] || q_shape[2] != v_shape[2]) {
    throw std::runtime_error("SDPALayer: sequence length mismatch");
  }
  if (q_shape[3] != k_shape[3] || q_shape[3] != v_shape[3]) {
    throw std::runtime_error("SDPALayer: head dimension mismatch");
  }

  // Output shape same as Q: (B, H, S, D)
  return {q_shape};
}

Vec<Tensor> SDPALayer::forward_impl(const Vec<ConstTensor> &inputs, size_t mb_id) {
  if (inputs.size() != 3) {
    throw std::runtime_error("SDPALayer: expected exactly 3 inputs (Q, K, V)");
  }

  const ConstTensor &q = inputs[0];
  const ConstTensor &k = inputs[1];
  const ConstTensor &v = inputs[2];

  if (q->dims() != 4) {
    throw std::runtime_error("SDPALayer: Q must be 4D (B, H, S, D)");
  }

  const auto &q_shape = q->shape();
  const size_t batch_size = q_shape[0];
  const size_t num_heads = q_shape[1];
  const size_t seq_len = q_shape[2];
  const size_t head_dim = q_shape[3];

  // Validate K and V shapes
  {
    const auto &k_shape = k->shape();
    const auto &v_shape = v->shape();
    if (k_shape != q_shape || v_shape != q_shape) {
      throw std::runtime_error("SDPALayer: Q, K, V must have same shape");
    }
  }

  Tensor output = get_output_tensor(q_shape);

  // Cache for backward
  if (this->is_training_) {
    micro_batch_q_shapes_[mb_id] = q_shape;
    micro_batch_q_cache_[mb_id] = q;
    micro_batch_k_cache_[mb_id] = k;
    micro_batch_v_cache_[mb_id] = v;
  }

#ifdef USE_CUDNN
  if (q->device_type() == DeviceType::GPU) {
    cudnn_forward(q, k, v, output, mb_id);
    return {output};
  }
#endif

  // CPU or fallback GPU implementation
  DISPATCH_IO_DTYPE(compute_sdpa_forward_impl, q, k, v, output, batch_size, num_heads, seq_len,
                    head_dim, this->flow_handle_, mb_id);
  return {output};
}

Vec<Tensor> SDPALayer::backward_impl(const Vec<ConstTensor> &grad_outputs, size_t mb_id) {
  if (grad_outputs.size() != 1) {
    throw std::runtime_error("SDPALayer: expected exactly 1 grad output");
  }

  const ConstTensor &grad_output = grad_outputs[0];

  // Retrieve cached forward pass data
  auto it_shapes = micro_batch_q_shapes_.find(mb_id);
  if (it_shapes == micro_batch_q_shapes_.end()) {
    throw std::runtime_error("SDPALayer: forward must be called before backward");
  }

  const auto &q_shape = it_shapes->second;
  const size_t batch_size = q_shape[0];
  const size_t num_heads = q_shape[1];
  const size_t seq_len = q_shape[2];
  const size_t head_dim = q_shape[3];

  const ConstTensor &q = micro_batch_q_cache_[mb_id];
  const ConstTensor &k = micro_batch_k_cache_[mb_id];
  const ConstTensor &v = micro_batch_v_cache_[mb_id];

  // Allocate gradient tensors
  Tensor grad_q = get_workspace(q_shape, this->io_dtype_);
  Tensor grad_k = get_workspace(q_shape, this->io_dtype_);
  Tensor grad_v = get_workspace(q_shape, this->io_dtype_);

  // We need the forward output for backward - reconstruct or cache it
  // For now, we'll compute it on-the-fly (can be optimized by caching)
  Tensor output = this->get_workspace(q_shape);

  // Run forward to get output for backward
  DISPATCH_IO_DTYPE(compute_sdpa_forward_impl, q, k, v, output, batch_size, num_heads, seq_len,
                    head_dim, this->flow_handle_, mb_id);

#ifdef USE_CUDNN
  if (grad_output->device_type() == DeviceType::GPU) {
    cudnn_backward(q, k, v, output, grad_output, grad_q, grad_k, grad_v, mb_id);
    // Clear cached data
    micro_batch_q_shapes_.erase(mb_id);
    micro_batch_q_cache_.erase(mb_id);
    micro_batch_k_cache_.erase(mb_id);
    micro_batch_v_cache_.erase(mb_id);
    micro_batch_stats_cache_.erase(mb_id);
    return {grad_q, grad_k, grad_v};
  }
#endif

  // CPU or fallback GPU implementation
  DISPATCH_IO_DTYPE(compute_sdpa_backward_impl, q, k, v, output, grad_output, grad_q, grad_k,
                    grad_v, batch_size, num_heads, seq_len, head_dim, this->flow_handle_, mb_id);

  // Clear cached data
  micro_batch_q_shapes_.erase(mb_id);
  micro_batch_q_cache_.erase(mb_id);
  micro_batch_k_cache_.erase(mb_id);
  micro_batch_v_cache_.erase(mb_id);
  micro_batch_stats_cache_.erase(mb_id);

  return {grad_q, grad_k, grad_v};
}

template <typename IO_T>
std::unique_ptr<Task> SDPALayer::compute_sdpa_forward_impl(
    const ConstTensor &q, const ConstTensor &k, const ConstTensor &v, const Tensor &output,
    size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim, flowHandle_t handle,
    size_t mb_id) const {
  if (q->data_type() != dtype_of<IO_T>() || k->data_type() != dtype_of<IO_T>() ||
      v->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("SDPALayer: data type mismatch in forward pass");
  }

  if (q->device_type() == DeviceType::CPU) {
    cpu::sdpa::run_forward<IO_T>(q->data_as<IO_T>(), k->data_as<IO_T>(), v->data_as<IO_T>(),
                                 output->data_as<IO_T>(), batch_size, num_heads, seq_len, head_dim,
                                 attn_scale_, is_causal_);
  }
#ifdef USE_CUDA
  else if (q->device_type() == DeviceType::GPU) {
    cuda::sdpa::run_forward<IO_T>(q->data_as<IO_T>(), k->data_as<IO_T>(), v->data_as<IO_T>(),
                                  output->data_as<IO_T>(), batch_size, num_heads, seq_len, head_dim,
                                  attn_scale_, is_causal_);
  }
#endif
  else {
    throw std::runtime_error("SDPALayer: unsupported device type");
  }

  return nullptr;
}

template <typename IO_T>
std::unique_ptr<Task> SDPALayer::compute_sdpa_backward_impl(
    const ConstTensor &q, const ConstTensor &k, const ConstTensor &v, const ConstTensor &output,
    const ConstTensor &grad_output, const Tensor &grad_q, const Tensor &grad_k,
    const Tensor &grad_v, size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim,
    flowHandle_t handle, size_t mb_id) const {
  if (q->data_type() != dtype_of<IO_T>() || grad_output->data_type() != dtype_of<IO_T>() ||
      grad_q->data_type() != dtype_of<IO_T>() || grad_k->data_type() != dtype_of<IO_T>() ||
      grad_v->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("SDPALayer: data type mismatch in backward pass");
  }

  if (grad_output->device_type() == DeviceType::CPU) {
    cpu::sdpa::run_backward<IO_T>(
        q->data_as<IO_T>(), k->data_as<IO_T>(), v->data_as<IO_T>(), output->data_as<IO_T>(),
        grad_output->data_as<IO_T>(), grad_q->data_as<IO_T>(), grad_k->data_as<IO_T>(),
        grad_v->data_as<IO_T>(), batch_size, num_heads, seq_len, head_dim, attn_scale_, is_causal_);
  }
#ifdef USE_CUDA
  else if (grad_output->device_type() == DeviceType::GPU) {
    cuda::sdpa::run_backward<IO_T>(
        q->data_as<IO_T>(), k->data_as<IO_T>(), v->data_as<IO_T>(), output->data_as<IO_T>(),
        grad_output->data_as<IO_T>(), grad_q->data_as<IO_T>(), grad_k->data_as<IO_T>(),
        grad_v->data_as<IO_T>(), batch_size, num_heads, seq_len, head_dim, attn_scale_, is_causal_);
  }
#endif
  else {
    throw std::runtime_error("SDPALayer: unsupported device type");
  }

  return nullptr;
}

#ifdef USE_CUDNN
void SDPALayer::cudnn_forward(const ConstTensor &q, const ConstTensor &k, const ConstTensor &v,
                              const Tensor &output, size_t mb_id) {
  const auto &q_shape = q->shape();
  size_t batch_size = q_shape[0];
  size_t num_heads = q_shape[1];
  size_t seq_len = q_shape[2];
  size_t head_dim = q_shape[3];

  // Create shape key for caching
  size_t shape_key = 0;
  size_t hash_val = batch_size ^ (num_heads << 8) ^ (seq_len << 16) ^ (head_dim << 24);
  shape_key = hash_val;

  // Initialize attention stats if not cached
  if (stats_cache_.find(shape_key) == stats_cache_.end()) {
    auto stats = new AttentionStats();
    init_attention_stats(*stats, batch_size, num_heads, seq_len, head_dim, attn_scale_, is_causal_);

    // Get cuDNN handle
    cudnnHandle_t cudnn_handle = CUDAContext::getCudnnHandle();

    // Convert dtype
    cudnnDataType_t io_data_type = cuda::cudnn::to_cudnn_datatype(q->data_type());
    cudnnDataType_t compute_data_type = cuda::cudnn::to_cudnn_datatype(this->compute_dtype_);

    // Initialize cuDNN flash attention handle
    auto fe_handle = cuda::cudnn_flash_attention::initialize_fe_handle(cudnn_handle, io_data_type,
                                                                       compute_data_type, *stats);

    fe_handle_cache_[shape_key] = fe_handle;
    stats_cache_[shape_key] = stats;
  }

  auto *fe_handle =
      static_cast<cuda::cudnn_flash_attention::feHandle_t *>(fe_handle_cache_[shape_key]);
  auto &stats = *static_cast<AttentionStats *>(stats_cache_[shape_key]);

  // Allocate workspace
  Tensor workspace = this->get_workspace({stats.fwd_workspace_size});

  // Allocate stats tensor
  Tensor stats_tensor = this->get_workspace({batch_size, num_heads, seq_len, 1});

  // Cache output for backward if training
  if (this->is_training_) {
    micro_batch_stats_cache_[mb_id] = stats_tensor;
  }

  // Call cuDNN flash attention forward
  create_cuda_task(this->flow_handle_, cuda::cudnn_flash_attention::run_forward, fe_handle, stats,
                   q->data(), k->data(), v->data(), output->data(), stats_tensor->data(),
                   workspace->data());
}

void SDPALayer::cudnn_backward(const ConstTensor &q, const ConstTensor &k, const ConstTensor &v,
                               const ConstTensor &output, const ConstTensor &grad_output,
                               const Tensor &grad_q, const Tensor &grad_k, const Tensor &grad_v,
                               size_t mb_id) {
  const auto &q_shape = q->shape();
  size_t batch_size = q_shape[0];
  size_t num_heads = q_shape[1];
  size_t seq_len = q_shape[2];
  size_t head_dim = q_shape[3];

  size_t shape_key = batch_size ^ (num_heads << 8) ^ (seq_len << 16) ^ (head_dim << 24);

  auto *fe_handle =
      static_cast<cuda::cudnn_flash_attention::feHandle_t *>(fe_handle_cache_[shape_key]);
  auto &stats = *static_cast<AttentionStats *>(stats_cache_[shape_key]);
  auto &stats_tensor = micro_batch_stats_cache_[mb_id];

  // Allocate workspace
  Tensor workspace = this->get_workspace({stats.bwd_workspace_size});

  // Call cuDNN flash attention backward
  create_cuda_task(this->flow_handle_, cuda::cudnn_flash_attention::run_backward, fe_handle, stats,
                   q->data(), k->data(), v->data(), output->data(), grad_output->data(),
                   stats_tensor->data(), grad_q->data(), grad_k->data(), grad_v->data(),
                   workspace->data());
}
#endif

std::unique_ptr<SDPALayer> SDPALayer::create_from_config(const LayerConfig &config) {
  float attn_scale = 1.0f;
  bool is_causal = false;
  std::string name = "sdpa";

  attn_scale = config.get<float>("attn_scale", 1.0f);
  is_causal = config.get<bool>("is_causal", false);
  name = config.name.empty() ? "sdpa" : config.name;

  return std::make_unique<SDPALayer>(attn_scale, is_causal, name);
}

}  // namespace tnn
