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

namespace synet {

namespace {

template <typename T>
constexpr DType_t sdpa_workspace_dtype() {
  using AccT = typename TypeTraits<T>::ComputePrecision;
  if constexpr (std::is_same_v<AccT, double>) {
    return DType_t::FP64;
  } else {
    return DType_t::FP32;
  }
}

}  // namespace

SDPALayerImpl::SDPALayerImpl(float attn_scale, bool is_causal, const std::string &name)
    : attn_scale_(attn_scale),
      is_causal_(is_causal),
      is_training_(false) {
  this->name_ = name;
}

SDPALayerImpl::~SDPALayerImpl() {
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
    throw std::runtime_error("SDPALayerImpl: head dimension mismatch");
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

#ifdef USE_CUDNN
  if (q.device_type() == DeviceType::GPU) {
    cudnn_forward(q, k, v, output, residuals);
    return {output};
  }
#endif

  Vec<size_t> attn_shape = {batch_size, num_heads, seq_len, seq_len};
  Tensor attn_weights = get_tensor(attn_shape, io_dtype_);

  if (this->is_training_) {
    residuals["attn_weights"] = attn_weights;
  }

  // CPU or fallback GPU implementation
  DISPATCH_DTYPE(this->io_dtype_, IO_T, {
    Tensor scores = get_tensor(attn_shape, sdpa_workspace_dtype<IO_T>());
    compute_sdpa_forward_impl<IO_T>(q, k, v, output, scores, attn_weights, batch_size, num_heads,
                                    seq_len, head_dim, this->flow_handle_, residuals);
  });
  return {output};
}

Vec<Tensor> SDPALayerImpl::backward_impl(const Vec<Tensor> &grad_outputs, Residuals &residuals) {
  if (grad_outputs.size() != 1) {
    throw std::runtime_error("SDPALayerImpl: expected exactly 1 grad output");
  }

  const Tensor &grad_output = grad_outputs[0];

  // Retrieve cached forward pass data

  const Tensor &q = residuals["q"];
  const Tensor &k = residuals["k"];
  const Tensor &v = residuals["v"];

  const Vec<size_t> &q_shape = q.shape();
  size_t batch_size = q_shape[0];
  size_t num_heads = q_shape[1];
  size_t seq_len = q_shape[2];
  size_t head_dim = q_shape[3];

  // Allocate gradient tensors
  Tensor grad_q = get_tensor(q_shape, this->io_dtype_);
  Tensor grad_k = get_tensor(q_shape, this->io_dtype_);
  Tensor grad_v = get_tensor(q_shape, this->io_dtype_);

#ifdef USE_CUDNN
  if (grad_output.device_type() == DeviceType::GPU) {
    Tensor &output = residuals["output"];
    cudnn_backward(q, k, v, output, grad_output, grad_q, grad_k, grad_v, residuals);
    return {grad_q, grad_k, grad_v};
  }
#endif

  const Tensor &attn_weights = residuals["attn_weights"];
  if (!attn_weights) {
    throw std::runtime_error("SDPALayerImpl: missing cached attention weights for backward");
  }

  Vec<size_t> attn_shape = {batch_size, num_heads, seq_len, seq_len};

  // CPU or fallback GPU implementation
  DISPATCH_DTYPE(this->io_dtype_, IO_T, {
    Tensor grad_scores = get_tensor(attn_shape, sdpa_workspace_dtype<IO_T>());
    compute_sdpa_backward_impl<IO_T>(q, k, v, attn_weights, grad_output, grad_scores, grad_q,
                                     grad_k, grad_v, batch_size, num_heads, seq_len, head_dim,
                                     this->flow_handle_, residuals);
  });

  return {grad_q, grad_k, grad_v};
}

template <typename IO_T>
std::unique_ptr<Task> SDPALayerImpl::compute_sdpa_forward_impl(
    const Tensor &q, const Tensor &k, const Tensor &v, Tensor &output, Tensor &scores,
    Tensor &attn_weights, size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim,
    flowHandle_t handle, Residuals &residuals) const {
  using AccT = typename TypeTraits<IO_T>::ComputePrecision;

  if (q.dtype() != dtype_of<IO_T>() || k.dtype() != dtype_of<IO_T>() ||
      v.dtype() != dtype_of<IO_T>() || output.dtype() != dtype_of<IO_T>() ||
      attn_weights.dtype() != dtype_of<IO_T>()) {
    throw std::runtime_error("SDPALayerImpl: data type mismatch in forward pass");
  }
  if (scores.dtype() != sdpa_workspace_dtype<IO_T>()) {
    throw std::runtime_error("SDPALayerImpl: score workspace dtype mismatch in forward pass");
  }

  if (q.device_type() == DeviceType::CPU) {
    return create_cpu_task(handle, cpu::sdpa::run_forward<IO_T>, q.data_as<IO_T>(),
                           k.data_as<IO_T>(), v.data_as<IO_T>(), output.data_as<IO_T>(),
                           scores.data_as<AccT>(), attn_weights.data_as<IO_T>(), batch_size,
                           num_heads, seq_len, head_dim, attn_scale_, is_causal_);
  }
#ifdef USE_CUDA
  else if (q.device_type() == DeviceType::GPU) {
    return create_cuda_task(handle, cuda::sdpa::run_forward<IO_T>, q.data_as<IO_T>(),
                            k.data_as<IO_T>(), v.data_as<IO_T>(), output.data_as<IO_T>(),
                            scores.data_as<AccT>(), attn_weights.data_as<IO_T>(), batch_size,
                            num_heads, seq_len, head_dim, attn_scale_, is_causal_);
  }
#endif
  else {
    throw std::runtime_error("SDPALayerImpl: unsupported device type");
  }
}

template <typename IO_T>
std::unique_ptr<Task> SDPALayerImpl::compute_sdpa_backward_impl(
    const Tensor &q, const Tensor &k, const Tensor &v, const Tensor &attn_weights,
    const Tensor &grad_output, Tensor &grad_scores, Tensor &grad_q, Tensor &grad_k, Tensor &grad_v,
    size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim, flowHandle_t handle,
    Residuals &residuals) const {
  using AccT = typename TypeTraits<IO_T>::ComputePrecision;

  if (q.dtype() != dtype_of<IO_T>() || grad_output.dtype() != dtype_of<IO_T>() ||
      attn_weights.dtype() != dtype_of<IO_T>() || grad_q.dtype() != dtype_of<IO_T>() ||
      grad_k.dtype() != dtype_of<IO_T>() || grad_v.dtype() != dtype_of<IO_T>()) {
    throw std::runtime_error("SDPALayerImpl: data type mismatch in backward pass");
  }
  if (grad_scores.dtype() != sdpa_workspace_dtype<IO_T>()) {
    throw std::runtime_error("SDPALayerImpl: grad-score workspace dtype mismatch in backward pass");
  }

  if (grad_output.device_type() == DeviceType::CPU) {
    return create_cpu_task(handle, cpu::sdpa::run_backward<IO_T>, q.data_as<IO_T>(),
                           k.data_as<IO_T>(), v.data_as<IO_T>(), attn_weights.data_as<IO_T>(),
                           grad_output.data_as<IO_T>(), grad_scores.data_as<AccT>(),
                           grad_q.data_as<IO_T>(), grad_k.data_as<IO_T>(), grad_v.data_as<IO_T>(),
                           batch_size, num_heads, seq_len, head_dim, attn_scale_, is_causal_);
  }
#ifdef USE_CUDA
  else if (grad_output.device_type() == DeviceType::GPU) {
    return create_cuda_task(handle, cuda::sdpa::run_backward<IO_T>, q.data_as<IO_T>(),
                            k.data_as<IO_T>(), v.data_as<IO_T>(), attn_weights.data_as<IO_T>(),
                            grad_output.data_as<IO_T>(), grad_scores.data_as<AccT>(),
                            grad_q.data_as<IO_T>(), grad_k.data_as<IO_T>(), grad_v.data_as<IO_T>(),
                            batch_size, num_heads, seq_len, head_dim, attn_scale_, is_causal_);
  }
#endif
  else {
    throw std::runtime_error("SDPALayerImpl: unsupported device type");
  }
}

#ifdef USE_CUDNN
void SDPALayerImpl::cudnn_forward(const Tensor &q, const Tensor &k, const Tensor &v, Tensor &output,
                                  Residuals &residuals) {
  const auto &q_shape = q.shape();
  size_t batch_size = q_shape[0];
  size_t num_heads = q_shape[1];
  size_t seq_len = q_shape[2];
  size_t head_dim = q_shape[3];

  size_t shape_key = 0;
  size_t hash_val = batch_size ^ (num_heads << 8) ^ (seq_len << 16) ^ (head_dim << 24);
  shape_key = hash_val;

  if (stats_cache_.find(shape_key) == stats_cache_.end()) {
    auto stats = new AttentionStats();
    init_attention_stats(*stats, batch_size, num_heads, seq_len, head_dim, attn_scale_, is_causal_);

    // Get cuDNN handle
    cudnnHandle_t cudnn_handle = CUDAContext::getCudnnHandle();

    // Convert dtype
    cudnnDataType_t io_dtype = cuda::cudnn::to_cudnn_datatype(q.dtype());
    cudnnDataType_t compute_dtype = cuda::cudnn::to_cudnn_datatype(this->compute_dtype_);

    // Initialize cuDNN flash attention handle
    auto fe_handle = cuda::cudnn_flash_attention::initialize_fe_handle(cudnn_handle, io_dtype,
                                                                       compute_dtype, *stats);

    fe_handle_cache_[shape_key] = fe_handle;
    stats_cache_[shape_key] = stats;
  }

  auto *fe_handle =
      static_cast<cuda::cudnn_flash_attention::feHandle_t *>(fe_handle_cache_[shape_key]);
  auto &stats = *static_cast<AttentionStats *>(stats_cache_[shape_key]);

  Tensor workspace = this->get_tensor({stats.fwd_workspace_size}, io_dtype_);

  Tensor stats_tensor = this->get_tensor({batch_size, num_heads, seq_len, 1}, io_dtype_);

  if (this->is_training_) {
    residuals["stats"] = stats_tensor;
  }

  create_cuda_task(this->flow_handle_, cuda::cudnn_flash_attention::run_forward, fe_handle, stats,
                   q.data_as<void>(), k.data_as<void>(), v.data_as<void>(), output.data_as<void>(),
                   stats_tensor.data_as<void>(), workspace.data_as<void>());
}

void SDPALayerImpl::cudnn_backward(const Tensor &q, const Tensor &k, const Tensor &v,
                                   const Tensor &output, const Tensor &grad_output, Tensor &grad_q,
                                   Tensor &grad_k, Tensor &grad_v, Residuals &residuals) {
  const auto &q_shape = q.shape();
  size_t batch_size = q_shape[0];
  size_t num_heads = q_shape[1];
  size_t seq_len = q_shape[2];
  size_t head_dim = q_shape[3];

  size_t shape_key = batch_size ^ (num_heads << 8) ^ (seq_len << 16) ^ (head_dim << 24);

  auto *fe_handle =
      static_cast<cuda::cudnn_flash_attention::feHandle_t *>(fe_handle_cache_[shape_key]);
  auto &stats = *static_cast<AttentionStats *>(stats_cache_[shape_key]);
  Tensor &stats_tensor = residuals["stats"];

  // Allocate workspace
  Tensor workspace = this->get_tensor({stats.bwd_workspace_size}, io_dtype_);

  // Call cuDNN flash attention backward
  create_cuda_task(this->flow_handle_, cuda::cudnn_flash_attention::run_backward, fe_handle, stats,
                   q.data_as<void>(), k.data_as<void>(), v.data_as<void>(), output.data_as<void>(),
                   grad_output.data_as<void>(), stats_tensor.data_as<void>(),
                   grad_q.data_as<void>(), grad_k.data_as<void>(), grad_v.data_as<void>(),
                   workspace.data_as<void>());
}
#endif

std::shared_ptr<SDPALayerImpl> SDPALayerImpl::create_from_config(const LayerConfig &config) {
  float attn_scale = 1.0f;
  bool is_causal = false;
  std::string name = "sdpa";

  attn_scale = config.get<float>("attn_scale", 1.0f);
  is_causal = config.get<bool>("is_causal", false);
  name = config.name.empty() ? "sdpa" : config.name;

  return std::make_shared<SDPALayerImpl>(attn_scale, is_causal, name);
}

}  // namespace synet
