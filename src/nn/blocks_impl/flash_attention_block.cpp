/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/blocks_impl/flash_attention_block.hpp"
#include "device/device_type.hpp"
#ifdef USE_CUDA
#include "nn/blocks_impl/cuda/permute_heads.hpp"
#endif
#ifdef USE_CUDNN
#include "device/cuda/cuda_context.hpp"
#include "nn/blocks_impl/cuda/cudnn_flash_attention_ops.hpp"
#endif
#include <cmath>
#include <stdexcept>

namespace tnn {

// Constructor
FlashAttentionBlock::FlashAttentionBlock(size_t embed_dim, size_t num_heads, bool is_causal,
                                         const std::string &name)
    : ParameterizedLayer(name), embed_dim_(embed_dim), num_heads_(num_heads),
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

void FlashAttentionBlock::init_params() {
  q_proj_->init();
  k_proj_->init();
  v_proj_->init();
  out_proj_->init();
}

void FlashAttentionBlock::on_set_io_dtype(DType_t dtype) {
  if (io_dtype_ != DType_t::FP16) {
    throw std::invalid_argument("FlashAttentionBlock only supports FP16 io_dtype for cuDNN SDPA");
  }
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

void FlashAttentionBlock::forward_impl(const Tensor &input, Tensor &output, size_t mb_id) {
  const auto &input_shape = input->shape();
  if (input_shape.size() != 3) {
    throw std::invalid_argument("FlashAttentionBlock: Input must be 3D (B, S, E)");
  }

  size_t batch_size = input_shape[0];
  size_t seq_len = input_shape[1];
  size_t embed_dim = input_shape[2];

  if (embed_dim != embed_dim_) {
    throw std::invalid_argument("FlashAttentionBlock: Input embed_dim mismatch");
  }

  if (this->device_->device_type() != DeviceType::GPU) {
    throw std::runtime_error("FlashAttentionBlock requires GPU device for cuDNN SDPA");
  }

#ifndef USE_CUDNN
  throw std::runtime_error("FlashAttentionBlock requires cuDNN for SDPA");
#else
  if (this->io_dtype_ != DType_t::FP16) {
    throw std::runtime_error("FlashAttentionBlock SDPA requires FP16 io_dtype");
  }

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

  Tensor q_heads = this->get_buffer({batch_size, num_heads_, seq_len, head_dim_}, io_dtype_);
  Tensor k_heads = this->get_buffer({batch_size, num_heads_, seq_len, head_dim_}, io_dtype_);
  Tensor v_heads = this->get_buffer({batch_size, num_heads_, seq_len, head_dim_}, io_dtype_);

  DISPATCH_ON_DTYPE(io_dtype_, T, {
    create_gpu_task("default", cuda::permute_heads<T>, q->data_as<T>(), q_heads->data_as<T>(),
                    batch_size, seq_len, num_heads_, head_dim_);
    create_gpu_task("default", cuda::permute_heads<T>, k->data_as<T>(), k_heads->data_as<T>(),
                    batch_size, seq_len, num_heads_, head_dim_);
    create_gpu_task("default", cuda::permute_heads<T>, v->data_as<T>(), v_heads->data_as<T>(),
                    batch_size, seq_len, num_heads_, head_dim_);
  });

  auto cuda_context = dynamic_cast<CUDAContext *>(this->device_->context());
  if (!cuda_context) {
    throw std::runtime_error("FlashAttentionBlock requires CUDAContext for cuDNN SDPA");
  }

  cudnnHandle_t cudnn_handle = cuda_context->getCudnnHandle();
  float attn_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim_)));

  auto graph = cuda::cudnn_flash_attention::create_sdpa_forward_graph(
      static_cast<int64_t>(batch_size), static_cast<int64_t>(num_heads_),
      static_cast<int64_t>(seq_len), static_cast<int64_t>(head_dim_), attn_scale, is_causal_);

  cuda::cudnn_flash_attention::build_sdpa_forward_graph(graph, cudnn_handle);

  size_t workspace_size = cuda::cudnn_flash_attention::get_sdpa_forward_workspace_bytes(graph);
  size_t io_dtype_size = get_dtype_size(io_dtype_);
  size_t workspace_elements = (workspace_size + io_dtype_size - 1) / io_dtype_size;
  Tensor workspace = this->get_buffer({workspace_elements}, io_dtype_);

  Tensor attn_heads = this->get_buffer({batch_size, num_heads_, seq_len, head_dim_}, io_dtype_);

  create_gpu_task("default", cuda::cudnn_flash_attention::run_sdpa_forward, graph, cudnn_handle,
                  q_heads->data(), k_heads->data(), v_heads->data(), attn_heads->data(),
                  workspace->data());

  Tensor attn_out = this->get_buffer({batch_size, seq_len, embed_dim_}, io_dtype_);
  DISPATCH_ON_DTYPE(io_dtype_, T, {
    create_gpu_task("default", cuda::permute_heads<T>, attn_heads->data_as<T>(),
                    attn_out->data_as<T>(), batch_size, num_heads_, seq_len, head_dim_);
  });

  out_proj_->forward(attn_out, output, mb_id);
#endif
}

void FlashAttentionBlock::backward_impl(const Tensor &gradient, Tensor &grad_input, size_t mb_id) {}

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

std::vector<size_t>
FlashAttentionBlock::compute_output_shape(const std::vector<size_t> &input_shape) const {
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

std::unique_ptr<FlashAttentionBlock>
FlashAttentionBlock::create_from_config(const LayerConfig &config) {
  size_t embed_dim = config.get<size_t>("embed_dim");
  size_t num_heads = config.get<size_t>("num_heads");
  bool is_causal = config.get<bool>("is_causal", true);
  return std::make_unique<FlashAttentionBlock>(embed_dim, num_heads, is_causal, config.name);
}

} // namespace tnn
