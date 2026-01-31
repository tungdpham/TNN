/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/blocks_impl/flash_attention_block.hpp"

#include "device/cuda/cuda_context.hpp"
#include "device/device_type.hpp"
#include "device/task.hpp"
#include "nn/blocks_impl/common/flash_attention.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"
#ifdef USE_CUDA
#include "nn/blocks_impl/cuda/permute_heads.hpp"
#endif
#ifdef USE_CUDNN
#include "cuda/cudnn/common.hpp"
#include "nn/blocks_impl/cuda/cudnn_flash_attention_ops.hpp"
#endif
#include <cmath>
#include <stdexcept>
#include <string>

#include "type/type.hpp"

namespace tnn {

// Constructor
FlashAttentionBlock::FlashAttentionBlock(size_t embed_dim, size_t num_heads, bool is_causal,
                                         const std::string &name)
    : ParameterizedLayer(name),
      embed_dim_(embed_dim),
      num_heads_(num_heads),
      is_causal_(is_causal) {
  if (embed_dim % num_heads != 0) {
    throw std::invalid_argument("embed_dim must be divisible by num_heads");
  }
  head_dim_ = embed_dim / num_heads;

  q_proj_ = std::make_unique<DenseLayer>(embed_dim, embed_dim, true, name + "_q");
  k_proj_ = std::make_unique<DenseLayer>(embed_dim, embed_dim, true, name + "_k");
  v_proj_ = std::make_unique<DenseLayer>(embed_dim, embed_dim, true, name + "_v");
  out_proj_ = std::make_unique<DenseLayer>(embed_dim, embed_dim, true, name + "_out");
}

FlashAttentionBlock::~FlashAttentionBlock() {
#ifdef USE_CUDNN
  for (auto &kv : fe_handle_cache) {
    if (kv.second) {
      cuda::cudnn_flash_attention::destroy_fe_handle(kv.second);
    }
  }
  fe_handle_cache.clear();
#endif
}

void FlashAttentionBlock::init_params() {
  q_proj_->init();
  k_proj_->init();
  v_proj_->init();
  out_proj_->init();
}

void FlashAttentionBlock::on_set_io_dtype(DType_t dtype) {
  q_proj_->set_io_dtype(dtype);
  k_proj_->set_io_dtype(dtype);
  v_proj_->set_io_dtype(dtype);
  out_proj_->set_io_dtype(dtype);
}

void FlashAttentionBlock::on_set_param_dtype(DType_t dtype) {
  q_proj_->set_param_dtype(dtype);
  k_proj_->set_param_dtype(dtype);
  v_proj_->set_param_dtype(dtype);
  out_proj_->set_param_dtype(dtype);
}

void FlashAttentionBlock::on_set_device(const Device &device) {
  q_proj_->set_device(device);
  k_proj_->set_device(device);
  v_proj_->set_device(device);
  out_proj_->set_device(device);
}

size_t FlashAttentionBlock::get_shape_hash(size_t b, size_t h, size_t s, size_t d) const {
  size_t seed = 0;
  auto hash_combine = [&](size_t v) { seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2); };
  hash_combine(b);
  hash_combine(h);
  hash_combine(s);
  hash_combine(d);
  return seed;
}

void FlashAttentionBlock::forward_impl(const Tensor &input, Tensor &output, size_t mb_id) {
  if (input->dims() != 3) {
    throw std::invalid_argument("FlashAttentionBlock: Input must be 3D (B, S, E)");
  }

  size_t embed_dim = input->dimension(2);

  if (embed_dim != embed_dim_) {
    throw std::invalid_argument("FlashAttentionBlock: Input embed_dim mismatch");
  }

#ifdef USE_CUDNN
  if (this->device_->device_type() == DeviceType::GPU) {
    cudnn_forward(input, output, mb_id);
  } else
#endif
  {
    throw std::runtime_error("CPU implementation for FlashAttentionBlock not implemented");
  }
}

#ifdef USE_CUDNN
template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> FlashAttentionBlock::flash_attention_forward_task(
    cuda::cudnn_flash_attention::feHandle_t *fe_handle, AttentionStats &stats,
    const Tensor &q_heads, const Tensor &k_heads, const Tensor &v_heads, Tensor &attn_heads,
    Tensor &stats_tensor, Tensor &workspace, const std::string &flow_id) const {
  return create_cuda_task("default", cuda::cudnn_flash_attention::run_forward, fe_handle, stats,
                          q_heads->data(), k_heads->data(), v_heads->data(), attn_heads->data(),
                          stats_tensor->data(), workspace->data());
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> FlashAttentionBlock::flash_attention_backward_task(
    cuda::cudnn_flash_attention::feHandle_t *fe_handle, AttentionStats &stats,
    const Tensor &q_heads, const Tensor &k_heads, const Tensor &v_heads, const Tensor &attn_heads,
    const Tensor &grad_attn_heads, const Tensor &stats_tensor, Tensor &grad_q_heads,
    Tensor &grad_k_heads, Tensor &grad_v_heads, Tensor &workspace,
    const std::string &flow_id) const {
  return create_cuda_task("default", cuda::cudnn_flash_attention::run_backward, fe_handle, stats,
                          q_heads->data(), k_heads->data(), v_heads->data(), attn_heads->data(),
                          grad_attn_heads->data(), stats_tensor->data(), grad_q_heads->data(),
                          grad_k_heads->data(), grad_v_heads->data(), workspace->data());
}

void FlashAttentionBlock::cudnn_forward(const Tensor &input, Tensor &output, size_t mb_id) {
  const auto &input_shape = input->shape();
  size_t batch_size = input_shape[0];
  size_t seq_len = input_shape[1];

  size_t shape_key = get_shape_hash(batch_size, num_heads_, seq_len, head_dim_);

  if (stats_cache.find(shape_key) == stats_cache.end()) {
    AttentionStats stats;
    float attn_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim_)));
    init_attention_stats(stats, batch_size, num_heads_, seq_len, head_dim_, attn_scale, is_causal_);

    auto *cuda_context = dynamic_cast<CUDAContext *>(this->device_->context());
    if (!cuda_context) {
      throw std::runtime_error("FlashAttentionBlock requires CUDAContext");
    }
    cudnnHandle_t cudnn_handle = cuda_context->getCudnnHandle();

    cudnnDataType_t io_dtype = cuda::cudnn::to_cudnn_datatype(DType_t::FP16);
    cudnnDataType_t compute_dtype = cuda::cudnn::to_cudnn_datatype(DType_t::FP32);

    fe_handle_cache[shape_key] = cuda::cudnn_flash_attention::initialize_fe_handle(
        cudnn_handle, io_dtype, compute_dtype, stats);
    stats_cache[shape_key] = stats;
  }

  auto *fe_handle = fe_handle_cache[shape_key];
  auto &stats = stats_cache[shape_key];

  Tensor &q = q_cache_[mb_id];
  Tensor &k = k_cache_[mb_id];
  Tensor &v = v_cache_[mb_id];

  if (q == nullptr) {
    q = this->get_buffer({batch_size, seq_len, embed_dim_}, io_dtype_);
    k = this->get_buffer({batch_size, seq_len, embed_dim_}, io_dtype_);
    v = this->get_buffer({batch_size, seq_len, embed_dim_}, io_dtype_);
  }

  q_proj_->forward(input, q, mb_id);
  k_proj_->forward(input, k, mb_id);
  v_proj_->forward(input, v, mb_id);

  // since cudnn SDPA only support FP16/FP16 IO, we need to convert here
  Tensor q_heads = this->get_buffer({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);
  Tensor k_heads = this->get_buffer({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);
  Tensor v_heads = this->get_buffer({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);

  DISPATCH_ON_DTYPE(io_dtype_, T, {
    create_cuda_task("default", cuda::permute_heads<T, fp16>, q->data_as<T>(),
                     q_heads->data_as<fp16>(), batch_size, seq_len, num_heads_, head_dim_);
    create_cuda_task("default", cuda::permute_heads<T, fp16>, k->data_as<T>(),
                     k_heads->data_as<fp16>(), batch_size, seq_len, num_heads_, head_dim_);
    create_cuda_task("default", cuda::permute_heads<T, fp16>, v->data_as<T>(),
                     v_heads->data_as<fp16>(), batch_size, seq_len, num_heads_, head_dim_);
  });

  size_t workspace_size = stats.fwd_workspace_size;
  size_t io_dtype_size = get_dtype_size(DType_t::FP16);
  size_t workspace_elements = (workspace_size + io_dtype_size - 1) / io_dtype_size;
  Tensor workspace = this->get_buffer({workspace_elements}, io_dtype_);

  Tensor attn_heads = this->get_buffer({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);

  // Allocate stats tensor (b, h, s, 1) in float
  Tensor &stats_tensor = stats_cache_tensor_[mb_id];
  if (stats_tensor == nullptr) {
    stats_tensor = this->get_buffer({batch_size, num_heads_, seq_len, 1}, DType_t::FP32);
  }

  DISPATCH_ON_3_DTYPES_TO_METHOD(flash_attention_forward_task, fe_handle, stats, q_heads, k_heads,
                                 v_heads, attn_heads, stats_tensor, workspace, "default");

  Tensor &attn_out = attn_out_cache_[mb_id];
  if (attn_out == nullptr) {
    attn_out = this->get_buffer({batch_size, seq_len, embed_dim_}, io_dtype_);
  }
  DISPATCH_ON_DTYPE(io_dtype_, T, {
    create_cuda_task("default", cuda::permute_heads<fp16, T>, attn_heads->data_as<fp16>(),
                     attn_out->data_as<T>(), batch_size, num_heads_, seq_len, head_dim_);
  });

  out_proj_->forward(attn_out, output, mb_id);
}

void FlashAttentionBlock::cudnn_backward(const Tensor &gradient, Tensor &grad_input, size_t mb_id) {
  const auto &grad_shape = gradient->shape();
  size_t batch_size = grad_shape[0];
  size_t seq_len = grad_shape[1];

  size_t shape_key = get_shape_hash(batch_size, num_heads_, seq_len, head_dim_);

  auto *fe_handle = fe_handle_cache[shape_key];
  auto &stats = stats_cache[shape_key];

  // Get cached forward tensors
  Tensor &q = q_cache_[mb_id];
  Tensor &k = k_cache_[mb_id];
  Tensor &v = v_cache_[mb_id];
  Tensor &attn_out = attn_out_cache_[mb_id];
  Tensor &stats_tensor = stats_cache_tensor_[mb_id];

  // Backprop through out_proj
  Tensor grad_attn_out = this->get_buffer({batch_size, seq_len, embed_dim_}, io_dtype_);
  out_proj_->backward(gradient, grad_attn_out, mb_id);

  // Convert to head layout and FP16
  Tensor grad_attn_heads =
      this->get_buffer({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);
  DISPATCH_ON_DTYPE(io_dtype_, T, {
    create_cuda_task("default", cuda::permute_heads<T, fp16>, grad_attn_out->data_as<T>(),
                     grad_attn_heads->data_as<fp16>(), batch_size, seq_len, num_heads_, head_dim_);
  });

  // Get forward pass tensors in FP16 head layout
  Tensor q_heads = this->get_buffer({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);
  Tensor k_heads = this->get_buffer({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);
  Tensor v_heads = this->get_buffer({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);
  Tensor attn_heads = this->get_buffer({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);

  DISPATCH_ON_DTYPE(io_dtype_, T, {
    create_cuda_task("default", cuda::permute_heads<T, fp16>, q->data_as<T>(),
                     q_heads->data_as<fp16>(), batch_size, seq_len, num_heads_, head_dim_);
    create_cuda_task("default", cuda::permute_heads<T, fp16>, k->data_as<T>(),
                     k_heads->data_as<fp16>(), batch_size, seq_len, num_heads_, head_dim_);
    create_cuda_task("default", cuda::permute_heads<T, fp16>, v->data_as<T>(),
                     v_heads->data_as<fp16>(), batch_size, seq_len, num_heads_, head_dim_);
    create_cuda_task("default", cuda::permute_heads<T, fp16>, attn_out->data_as<T>(),
                     attn_heads->data_as<fp16>(), batch_size, seq_len, num_heads_, head_dim_);
  });

  // Allocate gradient tensors for Q, K, V heads
  Tensor grad_q_heads =
      this->get_buffer({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);
  Tensor grad_k_heads =
      this->get_buffer({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);
  Tensor grad_v_heads =
      this->get_buffer({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);

  size_t workspace_size = stats.bwd_workspace_size;
  size_t io_dtype_size = get_dtype_size(DType_t::FP16);
  size_t workspace_elements = (workspace_size + io_dtype_size - 1) / io_dtype_size;
  Tensor workspace = this->get_buffer({workspace_elements}, io_dtype_);

  // Run backward pass
  DISPATCH_ON_3_DTYPES_TO_METHOD(flash_attention_backward_task, fe_handle, stats, q_heads, k_heads,
                                 v_heads, attn_heads, grad_attn_heads, stats_tensor, grad_q_heads,
                                 grad_k_heads, grad_v_heads, workspace, "default");

  // Convert gradients back from head layout
  Tensor grad_q = this->get_buffer({batch_size, seq_len, embed_dim_}, io_dtype_);
  Tensor grad_k = this->get_buffer({batch_size, seq_len, embed_dim_}, io_dtype_);
  Tensor grad_v = this->get_buffer({batch_size, seq_len, embed_dim_}, io_dtype_);

  DISPATCH_ON_DTYPE(io_dtype_, T, {
    create_cuda_task("default", cuda::permute_heads<fp16, T>, grad_q_heads->data_as<fp16>(),
                     grad_q->data_as<T>(), batch_size, num_heads_, seq_len, head_dim_);
    create_cuda_task("default", cuda::permute_heads<fp16, T>, grad_k_heads->data_as<fp16>(),
                     grad_k->data_as<T>(), batch_size, num_heads_, seq_len, head_dim_);
    create_cuda_task("default", cuda::permute_heads<fp16, T>, grad_v_heads->data_as<fp16>(),
                     grad_v->data_as<T>(), batch_size, num_heads_, seq_len, head_dim_);
  });

  // Backprop through projections
  Tensor grad_from_q = this->get_buffer({batch_size, seq_len, embed_dim_}, io_dtype_);
  Tensor grad_from_k = this->get_buffer({batch_size, seq_len, embed_dim_}, io_dtype_);
  Tensor grad_from_v = this->get_buffer({batch_size, seq_len, embed_dim_}, io_dtype_);

  q_proj_->backward(grad_q, grad_from_q, mb_id);
  k_proj_->backward(grad_k, grad_from_k, mb_id);
  v_proj_->backward(grad_v, grad_from_v, mb_id);

  grad_input = grad_from_q + grad_from_k + grad_from_v;
}
#endif

void FlashAttentionBlock::backward_impl(const Tensor &gradient, Tensor &grad_input, size_t mb_id) {
#ifdef USE_CUDNN
  if (this->device_->device_type() == DeviceType::GPU) {
    cudnn_backward(gradient, grad_input, mb_id);
  } else
#endif
  {
    throw std::runtime_error("CPU implementation for FlashAttentionBlock backward not implemented");
  }
}

uint64_t FlashAttentionBlock::forward_flops(const std::vector<size_t> &input_shape) const {
  return 0;
}

uint64_t FlashAttentionBlock::backward_flops(const std::vector<size_t> &input_shape) const {
  return 0;
}

LayerConfig FlashAttentionBlock::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.parameters["embed_dim"] = embed_dim_;
  config.parameters["num_heads"] = num_heads_;
  return config;
}

std::unique_ptr<Layer> FlashAttentionBlock::clone() const {
  return std::make_unique<FlashAttentionBlock>(embed_dim_, num_heads_, is_causal_, this->name_);
}

std::vector<size_t> FlashAttentionBlock::compute_output_shape(
    const std::vector<size_t> &input_shape) const {
  return input_shape;
}

void FlashAttentionBlock::collect_parameters(std::vector<Tensor> &params) {
  auto q_params = q_proj_->parameters();
  params.insert(params.end(), q_params.begin(), q_params.end());
  auto k_params = k_proj_->parameters();
  params.insert(params.end(), k_params.begin(), k_params.end());
  auto v_params = v_proj_->parameters();
  params.insert(params.end(), v_params.begin(), v_params.end());
  auto out_params = out_proj_->parameters();
  params.insert(params.end(), out_params.begin(), out_params.end());
}

void FlashAttentionBlock::collect_gradients(std::vector<Tensor> &grads) {
  auto q_grads = q_proj_->gradients();
  grads.insert(grads.end(), q_grads.begin(), q_grads.end());
  auto k_grads = k_proj_->gradients();
  grads.insert(grads.end(), k_grads.begin(), k_grads.end());
  auto v_grads = v_proj_->gradients();
  grads.insert(grads.end(), v_grads.begin(), v_grads.end());
  auto out_grads = out_proj_->gradients();
  grads.insert(grads.end(), out_grads.begin(), out_grads.end());
}

std::unique_ptr<FlashAttentionBlock> FlashAttentionBlock::create_from_config(
    const LayerConfig &config) {
  size_t embed_dim = config.get<size_t>("embed_dim");
  size_t num_heads = config.get<size_t>("num_heads");
  bool is_causal = config.get<bool>("is_causal", true);
  return std::make_unique<FlashAttentionBlock>(embed_dim, num_heads, is_causal, config.name);
}

}  // namespace tnn
