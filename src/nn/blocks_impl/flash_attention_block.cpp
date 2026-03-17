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
#include "utils/misc.hpp"
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

#include "ops/ops.hpp"
#include "type/type.hpp"

namespace tnn {

// Constructor
FlashAttentionBlock::FlashAttentionBlock(size_t embed_dim, size_t num_heads, bool is_causal,
                                         const std::string &name)
    : embed_dim_(embed_dim),
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

void FlashAttentionBlock::forward_impl(const Vec<ConstTensor> &inputs, const Vec<Tensor> &outputs,
                                       size_t mb_id) {
  const ConstTensor &input = inputs[0];
  const Tensor &output = outputs[0];

  if (input->dims() != 3) {
    throw std::invalid_argument("FlashAttentionBlock: Input must be 3D (B, S, E)");
  }

  size_t embed_dim = input->dimension(2);

  if (embed_dim != embed_dim_) {
    throw std::invalid_argument("FlashAttentionBlock: Input embed_dim mismatch");
  }

#ifdef USE_CUDNN
  if (this->device().device_type() == DeviceType::GPU) {
    cudnn_forward(input, output, mb_id);
  } else
#endif
  {
    throw std::runtime_error("CPU implementation for FlashAttentionBlock not implemented");
  }
}

#ifdef USE_CUDNN
void FlashAttentionBlock::build_graph(const Vec<size_t> &input_shape) const {
  size_t batch_size = input_shape[0];
  size_t seq_len = input_shape[1];
  size_t shape_key = get_shape_hash({batch_size, num_heads_, seq_len, head_dim_});

  if (fe_handle_cache.find(shape_key) == fe_handle_cache.end()) {
    AttentionStats stats;
    float attn_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim_)));
    init_attention_stats(stats, batch_size, num_heads_, seq_len, head_dim_, attn_scale, is_causal_);

    auto cudnn_handle = CUDAContext::getCudnnHandle();

    cudnnDataType_t io_dtype = cuda::cudnn::to_cudnn_datatype(DType_t::FP16);
    cudnnDataType_t compute_dtype = cuda::cudnn::to_cudnn_datatype(DType_t::FP32);

    fe_handle_cache[shape_key] = cuda::cudnn_flash_attention::initialize_fe_handle(
        cudnn_handle, io_dtype, compute_dtype, stats);
    stats_cache[shape_key] = stats;
  }
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> FlashAttentionBlock::flash_attention_forward_task(
    cuda::cudnn_flash_attention::feHandle_t *fe_handle, AttentionStats &stats,
    const ConstTensor &q_heads, const ConstTensor &k_heads, const ConstTensor &v_heads,
    const Tensor &attn_heads, const Tensor &stats_tensor, const Tensor &workspace,
    flowHandle_t handle) const {
  return create_cuda_task(defaultFlowHandle, cuda::cudnn_flash_attention::run_forward, fe_handle,
                          stats, q_heads->data(), k_heads->data(), v_heads->data(),
                          attn_heads->data(), stats_tensor->data(), workspace->data());
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> FlashAttentionBlock::flash_attention_backward_task(
    cuda::cudnn_flash_attention::feHandle_t *fe_handle, AttentionStats &stats,
    const ConstTensor &q_heads, const ConstTensor &k_heads, const ConstTensor &v_heads,
    const ConstTensor &attn_heads, const ConstTensor &grad_attn_heads,
    const ConstTensor &stats_tensor, const Tensor &grad_q_heads, const Tensor &grad_k_heads,
    const Tensor &grad_v_heads, const Tensor &workspace, flowHandle_t handle) const {
  return create_cuda_task(defaultFlowHandle, cuda::cudnn_flash_attention::run_backward, fe_handle,
                          stats, q_heads->data(), k_heads->data(), v_heads->data(),
                          attn_heads->data(), grad_attn_heads->data(), stats_tensor->data(),
                          grad_q_heads->data(), grad_k_heads->data(), grad_v_heads->data(),
                          workspace->data());
}

void FlashAttentionBlock::cudnn_forward(const ConstTensor &input, const Tensor &output,
                                        size_t mb_id) {
  const auto &input_shape = input->shape();
  size_t batch_size = input_shape[0];
  size_t seq_len = input_shape[1];

  size_t shape_key = get_shape_hash({batch_size, num_heads_, seq_len, head_dim_});

  build_graph(input_shape);

  auto *fe_handle = fe_handle_cache[shape_key];
  auto &stats = stats_cache[shape_key];

  if (this->is_training_) {
    ConstTensor &cached_input = this->get_cached_tensor(mb_id, "input");
    cached_input = input;
  }

  Tensor q = this->get_workspace({batch_size, seq_len, embed_dim_}, io_dtype_);
  Tensor k = this->get_workspace({batch_size, seq_len, embed_dim_}, io_dtype_);
  Tensor v = this->get_workspace({batch_size, seq_len, embed_dim_}, io_dtype_);

  q_proj_->forward({input}, {q}, mb_id);
  k_proj_->forward({input}, {k}, mb_id);
  v_proj_->forward({input}, {v}, mb_id);

  // since cudnn SDPA only support FP16/FP16 IO, we need to convert here
  Tensor q_heads = this->get_workspace({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);
  Tensor k_heads = this->get_workspace({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);
  Tensor v_heads = this->get_workspace({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);

  DISPATCH_DTYPE(io_dtype_, T, {
    create_cuda_task(defaultFlowHandle, cuda::permute_heads<T, fp16>, q->data_as<T>(),
                     q_heads->data_as<fp16>(), batch_size, seq_len, num_heads_, head_dim_);
    create_cuda_task(defaultFlowHandle, cuda::permute_heads<T, fp16>, k->data_as<T>(),
                     k_heads->data_as<fp16>(), batch_size, seq_len, num_heads_, head_dim_);
    create_cuda_task(defaultFlowHandle, cuda::permute_heads<T, fp16>, v->data_as<T>(),
                     v_heads->data_as<fp16>(), batch_size, seq_len, num_heads_, head_dim_);
  });

  size_t workspace_size = stats.fwd_workspace_size;
  size_t io_dtype_size = get_dtype_size(DType_t::FP16);
  size_t workspace_elements = (workspace_size + io_dtype_size - 1) / io_dtype_size;
  Tensor workspace = this->get_workspace({workspace_elements}, io_dtype_);

  Tensor attn_heads =
      this->get_workspace({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);

  // Allocate stats tensor (b, h, s, 1) in float
  Tensor &stats_tensor = this->get_mutable_tensor(mb_id, "stats_tensor");
  if (stats_tensor == nullptr) {
    stats_tensor = this->get_workspace({batch_size, num_heads_, seq_len, 1}, DType_t::FP32);
  }

  DISPATCH_ON_3_DTYPES_TO_METHOD(flash_attention_forward_task, fe_handle, stats, q_heads, k_heads,
                                 v_heads, attn_heads, stats_tensor, workspace, defaultFlowHandle);

  Tensor &attn_out = this->get_mutable_tensor(mb_id, "attn_out");
  if (attn_out == nullptr) {
    attn_out = this->get_workspace({batch_size, seq_len, embed_dim_}, io_dtype_);
  }

  DISPATCH_DTYPE(io_dtype_, T, {
    create_cuda_task(defaultFlowHandle, cuda::permute_heads<fp16, T>, attn_heads->data_as<fp16>(),
                     attn_out->data_as<T>(), batch_size, num_heads_, seq_len, head_dim_);
  });

  out_proj_->forward({attn_out}, {output}, mb_id);
}

void FlashAttentionBlock::cudnn_backward(const ConstTensor &grad_output, const Tensor &grad_input,
                                         size_t mb_id) {
  const auto &grad_shape = grad_output->shape();
  size_t batch_size = grad_shape[0];
  size_t seq_len = grad_shape[1];

  size_t shape_key = get_shape_hash({batch_size, num_heads_, seq_len, head_dim_});

  auto *fe_handle = fe_handle_cache[shape_key];
  auto &stats = stats_cache[shape_key];

  // Get cached forward tensors
  ConstTensor &input = this->get_cached_tensor(mb_id, "input");
  const Tensor &attn_out = this->get_mutable_tensor(mb_id, "attn_out");
  const Tensor &stats_tensor = this->get_mutable_tensor(mb_id, "stats_tensor");

  // Backprop through out_proj
  Tensor grad_attn_out = this->get_workspace({batch_size, seq_len, embed_dim_}, io_dtype_);
  out_proj_->backward({grad_output}, {grad_attn_out}, mb_id);

  // Recompute Q, K, V from cached input (trading compute for memory)
  Tensor q = this->get_workspace({batch_size, seq_len, embed_dim_}, io_dtype_);
  Tensor k = this->get_workspace({batch_size, seq_len, embed_dim_}, io_dtype_);
  Tensor v = this->get_workspace({batch_size, seq_len, embed_dim_}, io_dtype_);

  q_proj_->forward({input}, {q}, mb_id);
  k_proj_->forward({input}, {k}, mb_id);
  v_proj_->forward({input}, {v}, mb_id);

  // Convert to head layout and FP16
  Tensor grad_attn_heads =
      this->get_workspace({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);
  DISPATCH_DTYPE(io_dtype_, T, {
    create_cuda_task(defaultFlowHandle, cuda::permute_heads<T, fp16>, grad_attn_out->data_as<T>(),
                     grad_attn_heads->data_as<fp16>(), batch_size, seq_len, num_heads_, head_dim_);
  });

  // Get forward pass tensors in FP16 head layout
  Tensor q_heads = this->get_workspace({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);
  Tensor k_heads = this->get_workspace({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);
  Tensor v_heads = this->get_workspace({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);
  Tensor attn_heads =
      this->get_workspace({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);

  DISPATCH_DTYPE(io_dtype_, T, {
    create_cuda_task(defaultFlowHandle, cuda::permute_heads<T, fp16>, q->data_as<T>(),
                     q_heads->data_as<fp16>(), batch_size, seq_len, num_heads_, head_dim_);
    create_cuda_task(defaultFlowHandle, cuda::permute_heads<T, fp16>, k->data_as<T>(),
                     k_heads->data_as<fp16>(), batch_size, seq_len, num_heads_, head_dim_);
    create_cuda_task(defaultFlowHandle, cuda::permute_heads<T, fp16>, v->data_as<T>(),
                     v_heads->data_as<fp16>(), batch_size, seq_len, num_heads_, head_dim_);
    create_cuda_task(defaultFlowHandle, cuda::permute_heads<T, fp16>, attn_out->data_as<T>(),
                     attn_heads->data_as<fp16>(), batch_size, seq_len, num_heads_, head_dim_);
  });

  // Allocate grad_output tensors for Q, K, V heads
  Tensor grad_q_heads =
      this->get_workspace({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);
  Tensor grad_k_heads =
      this->get_workspace({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);
  Tensor grad_v_heads =
      this->get_workspace({batch_size, num_heads_, seq_len, head_dim_}, DType_t::FP16);

  size_t workspace_size = stats.bwd_workspace_size;
  size_t io_dtype_size = get_dtype_size(DType_t::FP16);
  size_t workspace_elements = (workspace_size + io_dtype_size - 1) / io_dtype_size;
  Tensor workspace = this->get_workspace({workspace_elements}, io_dtype_);

  // Run backward pass
  DISPATCH_ON_3_DTYPES_TO_METHOD(flash_attention_backward_task, fe_handle, stats, q_heads, k_heads,
                                 v_heads, attn_heads, grad_attn_heads, stats_tensor, grad_q_heads,
                                 grad_k_heads, grad_v_heads, workspace, defaultFlowHandle);

  // Convert gradients back from head layout
  Tensor grad_q = this->get_workspace({batch_size, seq_len, embed_dim_}, io_dtype_);
  Tensor grad_k = this->get_workspace({batch_size, seq_len, embed_dim_}, io_dtype_);
  Tensor grad_v = this->get_workspace({batch_size, seq_len, embed_dim_}, io_dtype_);

  DISPATCH_DTYPE(io_dtype_, T, {
    create_cuda_task(defaultFlowHandle, cuda::permute_heads<fp16, T>, grad_q_heads->data_as<fp16>(),
                     grad_q->data_as<T>(), batch_size, num_heads_, seq_len, head_dim_);
    create_cuda_task(defaultFlowHandle, cuda::permute_heads<fp16, T>, grad_k_heads->data_as<fp16>(),
                     grad_k->data_as<T>(), batch_size, num_heads_, seq_len, head_dim_);
    create_cuda_task(defaultFlowHandle, cuda::permute_heads<fp16, T>, grad_v_heads->data_as<fp16>(),
                     grad_v->data_as<T>(), batch_size, num_heads_, seq_len, head_dim_);
  });

  // Backprop through separate Q, K, V projections
  Tensor grad_q_in = this->get_workspace({batch_size, seq_len, embed_dim_}, io_dtype_);
  Tensor grad_k_in = this->get_workspace({batch_size, seq_len, embed_dim_}, io_dtype_);
  Tensor grad_v_in = this->get_workspace({batch_size, seq_len, embed_dim_}, io_dtype_);

  q_proj_->backward({grad_q}, {grad_q_in}, mb_id);
  k_proj_->backward({grad_k}, {grad_k_in}, mb_id);
  v_proj_->backward({grad_v}, {grad_v_in}, mb_id);

  // Sum the gradients
  grad_input->ensure(grad_q_in->shape());
  size_t size = grad_q_in->size();

  Tensor temp = this->get_workspace(grad_q_in->shape(), io_dtype_);

  DISPATCH_IO_DTYPE(ops::add, grad_q_in->data_ptr(), grad_k_in->data_ptr(), temp->data_ptr(), size,
                    defaultFlowHandle);
  DISPATCH_IO_DTYPE(ops::add, temp->data_ptr(), grad_v_in->data_ptr(), grad_input->data_ptr(), size,
                    defaultFlowHandle);
}
#endif

void FlashAttentionBlock::backward_impl(const Vec<ConstTensor> &grad_outputs,
                                        const Vec<Tensor> &grad_inputs, size_t mb_id) {
  const ConstTensor &grad_output = grad_outputs[0];
  const Tensor &grad_input = grad_inputs[0];
#ifdef USE_CUDNN
  if (this->device().device_type() == DeviceType::GPU) {
    cudnn_backward(grad_output, grad_input, mb_id);
  } else
#endif
  {
    throw std::runtime_error("CPU implementation for FlashAttentionBlock backward not implemented");
  }
}

LayerConfig FlashAttentionBlock::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("embed_dim", embed_dim_);
  config.set("num_heads", num_heads_);
  return config;
}

Vec<Vec<size_t>> FlashAttentionBlock::output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
  return input_shapes;
}

size_t FlashAttentionBlock::fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  const auto &shape = input_shapes[0];
  size_t batch_size = shape[0], seq_len = shape[1];
  size_t dtype_size = get_dtype_size(io_dtype_);
  size_t fp16_size = get_dtype_size(DType_t::FP16);

  // q, k, v projected outputs: 3 * [B, L, E]
  size_t qkv_bytes = 3 * batch_size * seq_len * embed_dim_ * dtype_size;

  // q_heads, k_heads, v_heads in FP16: 3 * [B, H, L, D]
  size_t qkv_heads_bytes = 3 * batch_size * num_heads_ * seq_len * head_dim_ * fp16_size;

  // attn_heads in FP16: [B, H, L, D]
  size_t attn_heads_bytes = batch_size * num_heads_ * seq_len * head_dim_ * fp16_size;

  // cuDNN flash attention workspace
  size_t cudnn_ws_bytes = 0;
#ifdef USE_CUDNN
  if (allocator_ && allocator_->device().device_type() == DeviceType::GPU) {
    build_graph(shape);
    size_t shape_key = get_shape_hash({batch_size, num_heads_, seq_len, head_dim_});
    cudnn_ws_bytes = stats_cache.at(shape_key).fwd_workspace_size;
  }
#endif

  // Dense projection sub-layer workspace
  size_t proj_ws = 0;
  if (q_proj_) {
    proj_ws = q_proj_->fwd_workspace({{batch_size, seq_len, embed_dim_}});
  }

  size_t total = qkv_bytes + qkv_heads_bytes + attn_heads_bytes + cudnn_ws_bytes + proj_ws;
  return (total + 255) & ~static_cast<size_t>(255);
}

size_t FlashAttentionBlock::inf_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  return fwd_workspace(input_shapes);
}

size_t FlashAttentionBlock::bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  const auto &shape = input_shapes[0];
  size_t batch_size = shape[0], seq_len = shape[1];
  size_t dtype_size = get_dtype_size(io_dtype_);
  size_t fp16_size = get_dtype_size(DType_t::FP16);

  // q, k, v projected outputs + grads: 6 * [B, L, E]
  size_t qkv_bytes = 6 * batch_size * seq_len * embed_dim_ * dtype_size;

  // q_heads, k_heads, v_heads + grad heads in FP16: 6 * [B, H, L, D]
  size_t qkv_heads_bytes = 6 * batch_size * num_heads_ * seq_len * head_dim_ * fp16_size;

  // attn_heads + grad_attn_heads in FP16: 2 * [B, H, L, D]
  size_t attn_heads_bytes = 2 * batch_size * num_heads_ * seq_len * head_dim_ * fp16_size;

  // cuDNN flash attention backward workspace
  size_t cudnn_ws_bytes = 0;

#ifdef USE_CUDNN
  if (allocator_ && allocator_->device().device_type() == DeviceType::GPU) {
    size_t shape_key = get_shape_hash({batch_size, num_heads_, seq_len, head_dim_});
    build_graph(shape);
    const AttentionStats &stats = stats_cache.at(shape_key);
    cudnn_ws_bytes = stats.bwd_workspace_size;
  }
#endif

  // Dense projection sub-layer workspace
  size_t proj_ws = 0;
  if (q_proj_) {
    proj_ws = q_proj_->bwd_workspace({{batch_size, seq_len, embed_dim_}});
  }

  size_t total = qkv_bytes + qkv_heads_bytes + attn_heads_bytes + cudnn_ws_bytes + proj_ws;
  return (total + 255) & ~static_cast<size_t>(255);
}

std::unique_ptr<FlashAttentionBlock> FlashAttentionBlock::create_from_config(
    const LayerConfig &config) {
  size_t embed_dim = config.get<size_t>("embed_dim");
  size_t num_heads = config.get<size_t>("num_heads");
  bool is_causal = config.get<bool>("is_causal", true);
  return std::make_unique<FlashAttentionBlock>(embed_dim, num_heads, is_causal, config.name);
}

}  // namespace tnn
