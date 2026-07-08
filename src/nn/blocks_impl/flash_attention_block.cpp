/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/blocks_impl/flash_attention_block.hpp"

#include "device/cuda/cuda_context.hpp"
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

namespace tunx {

// Constructor
FlashAttentionBlockImpl::FlashAttentionBlockImpl(size_t embed_dim, size_t num_heads, bool is_causal,
                                                 const std::string &name)
    : Block(name),
      embed_dim_(embed_dim),
      num_heads_(num_heads),
      is_causal_(is_causal),
      q_proj_(embed_dim, embed_dim, true, name + "_q"),
      k_proj_(embed_dim, embed_dim, true, name + "_k"),
      v_proj_(embed_dim, embed_dim, true, name + "_v"),
      out_proj_(embed_dim, embed_dim, true, name + "_out") {
  if (embed_dim % num_heads != 0) {
    throw std::invalid_argument("embed_dim must be divisible by num_heads");
  }
  head_dim_ = embed_dim / num_heads;
}

FlashAttentionBlockImpl::~FlashAttentionBlockImpl() {
#ifdef USE_CUDNN
  for (auto &kv : fe_handle_cache) {
    if (kv.second) {
      cuda::cudnn_flash_attention::destroy_fe_handle(kv.second);
    }
  }
  fe_handle_cache.clear();
#endif
}

Vec<Tensor> FlashAttentionBlockImpl::forward_impl(const Vec<Tensor> &inputs, Residuals &residuals) {
  const Tensor &input = inputs[0];

  if (input.dims() != 3) {
    throw std::invalid_argument("FlashAttentionBlock: Input must be 3D (B, S, E) but got " +
                                std::to_string(input.dims()) + "D");
  }

  size_t embed_dim = input.dim(2);

  if (embed_dim != embed_dim_) {
    throw std::invalid_argument("FlashAttentionBlock: Input embed_dim mismatch");
  }

#ifdef USE_CUDNN
  if (get_engine_type() == EngineType::CUDA) {
    return {cudnn_forward(input, residuals)};
  } else
#endif
  {
    throw std::runtime_error("CPU implementation for FlashAttentionBlock not implemented");
  }
}

#ifdef USE_CUDNN
void FlashAttentionBlockImpl::build_graph(const Vec<size_t> &input_shape) const {
  size_t batch_size = input_shape[0];
  size_t seq_len = input_shape[1];
  size_t shape_key = get_shape_hash({batch_size, num_heads_, seq_len, head_dim_});

  if (fe_handle_cache.find(shape_key) == fe_handle_cache.end()) {
    AttentionStats stats;
    float attn_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim_)));
    init_attention_stats(stats, batch_size, num_heads_, seq_len, head_dim_, attn_scale, is_causal_);

    auto cudnn_handle = CUDAContext::getCudnnHandle();

    cudnnDataType_t io_dtype = cuda::cudnn::to_cudnn_datatype(DType_t::BF16);
    cudnnDataType_t compute_dtype = cuda::cudnn::to_cudnn_datatype(DType_t::FP32);

    fe_handle_cache[shape_key] = cuda::cudnn_flash_attention::initialize_fe_handle(
        cudnn_handle, io_dtype, compute_dtype, stats);
    stats_cache[shape_key] = stats;
  }
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> FlashAttentionBlockImpl::flash_attention_forward_task(
    cuda::cudnn_flash_attention::feHandle_t *fe_handle, AttentionStats &stats,
    const Tensor &q_heads, const Tensor &k_heads, const Tensor &v_heads, Tensor &attn_heads,
    Tensor &stats_tensor, Tensor &workspace, flowHandle_t handle) const {
  return create_cuda_task(defaultFlowHandle, cuda::cudnn_flash_attention::run_forward, fe_handle,
                          stats, q_heads.data_as<void>(), k_heads.data_as<void>(),
                          v_heads.data_as<void>(), attn_heads.data_as<void>(),
                          stats_tensor.data_as<void>(), workspace.data_as<void>());
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> FlashAttentionBlockImpl::flash_attention_backward_task(
    cuda::cudnn_flash_attention::feHandle_t *fe_handle, AttentionStats &stats,
    const Tensor &q_heads, const Tensor &k_heads, const Tensor &v_heads, const Tensor &attn_heads,
    Tensor &grad_attn_heads, Tensor &stats_tensor, Tensor &grad_q_heads, Tensor &grad_k_heads,
    Tensor &grad_v_heads, Tensor &workspace, flowHandle_t handle) const {
  return create_cuda_task(defaultFlowHandle, cuda::cudnn_flash_attention::run_backward, fe_handle,
                          stats, q_heads.data_as<void>(), k_heads.data_as<void>(),
                          v_heads.data_as<void>(), attn_heads.data_as<void>(),
                          grad_attn_heads.data_as<void>(), stats_tensor.data_as<void>(),
                          grad_q_heads.data_as<void>(), grad_k_heads.data_as<void>(),
                          grad_v_heads.data_as<void>(), workspace.data_as<void>());
}

Tensor FlashAttentionBlockImpl::cudnn_forward(const Tensor &input, Residuals &residuals) {
  const auto &input_shape = input.shape();
  size_t batch_size = input_shape[0];
  size_t seq_len = input_shape[1];

  size_t shape_key = get_shape_hash({batch_size, num_heads_, seq_len, head_dim_});

  build_graph(input_shape);

  auto *fe_handle = fe_handle_cache[shape_key];
  auto &stats = stats_cache[shape_key];

  if (this->is_training_) {
    residuals["input"] = input;
  }

  Tensor attn_out = this->get_tensor({batch_size, seq_len, embed_dim_}, io_dtype_);
  residuals["attn_out"] = attn_out;
  // Allocate stats tensor (b, h, s, 1) in float
  Tensor stats_tensor = this->get_tensor({batch_size, num_heads_, seq_len, 1}, DType_t::FP32);
  residuals["stats_tensor"] = stats_tensor;

  allocator_->flip();  // ensure workspace is on opposite side of output for algorithm 1

  Tensor attn_heads = this->get_tensor({batch_size, num_heads_, seq_len, head_dim_}, DType_t::BF16);

  // since cudnn SDPA only support FP16/BF16 IO, we need to convert here
  Tensor q_heads = this->get_tensor({batch_size, num_heads_, seq_len, head_dim_}, DType_t::BF16);
  Tensor k_heads = this->get_tensor({batch_size, num_heads_, seq_len, head_dim_}, DType_t::BF16);
  Tensor v_heads = this->get_tensor({batch_size, num_heads_, seq_len, head_dim_}, DType_t::BF16);

  Tensor q = q_proj_.forward({input}, residuals["q_proj"])[0];
  Tensor k = k_proj_.forward({input}, residuals["k_proj"])[0];
  Tensor v = v_proj_.forward({input}, residuals["v_proj"])[0];

  DISPATCH_DTYPE(io_dtype_, T, {
    create_cuda_task(defaultFlowHandle, cuda::permute_heads<T, bf16>, q.data_as<T>(),
                     q_heads.data_as<bf16>(), batch_size, seq_len, num_heads_, head_dim_);
    create_cuda_task(defaultFlowHandle, cuda::permute_heads<T, bf16>, k.data_as<T>(),
                     k_heads.data_as<bf16>(), batch_size, seq_len, num_heads_, head_dim_);
    create_cuda_task(defaultFlowHandle, cuda::permute_heads<T, bf16>, v.data_as<T>(),
                     v_heads.data_as<bf16>(), batch_size, seq_len, num_heads_, head_dim_);
  });

  {
    size_t workspace_size = stats.fwd_workspace_size;
    Tensor workspace = this->get_tensor({workspace_size}, DType_t::BYTE);

    DISPATCH_ON_3_DTYPES_TO_METHOD(flash_attention_forward_task, fe_handle, stats, q_heads, k_heads,
                                   v_heads, attn_heads, stats_tensor, workspace, defaultFlowHandle);
  }

  DISPATCH_DTYPE(io_dtype_, T, {
    create_cuda_task(defaultFlowHandle, cuda::permute_heads<bf16, T>, attn_heads.data_as<bf16>(),
                     attn_out.data_as<T>(), batch_size, num_heads_, seq_len, head_dim_);
  });

  allocator_->flip();  // flip back so attn_out is on opposite side as input for algorithm 1

  Tensor output = out_proj_.forward({attn_out}, residuals)[0];
  return output;
}

Tensor FlashAttentionBlockImpl::cudnn_backward(const Tensor &grad_output, Residuals &residuals) {
  const auto &grad_shape = grad_output.shape();
  size_t batch_size = grad_shape[0];
  size_t seq_len = grad_shape[1];

  size_t shape_key = get_shape_hash({batch_size, num_heads_, seq_len, head_dim_});

  auto *fe_handle = fe_handle_cache[shape_key];
  auto &stats = stats_cache[shape_key];

  // Get cached forward tensors
  const Tensor &input = residuals["input"];
  Tensor &attn_out = residuals["attn_out"];
  Tensor &stats_tensor = residuals["stats_tensor"];

  Tensor grad_input = this->get_tensor({batch_size, seq_len, embed_dim_}, io_dtype_);

  allocator_->flip();

  Tensor grad_q = this->get_tensor({batch_size, seq_len, embed_dim_}, io_dtype_);
  Tensor grad_k = this->get_tensor({batch_size, seq_len, embed_dim_}, io_dtype_);
  Tensor grad_v = this->get_tensor({batch_size, seq_len, embed_dim_}, io_dtype_);

  Tensor grad_attn_heads =
      this->get_tensor({batch_size, num_heads_, seq_len, head_dim_}, DType_t::BF16);

  {  // Backprop through out_proj
    Tensor grad_attn_out = out_proj_.backward({grad_output}, residuals["out_proj"])[0];
    // Convert to head layout and FP16
    DISPATCH_DTYPE(io_dtype_, T, {
      create_cuda_task(defaultFlowHandle, cuda::permute_heads<T, bf16>, grad_attn_out.data_as<T>(),
                       grad_attn_heads.data_as<bf16>(), batch_size, seq_len, num_heads_, head_dim_);
    });
  }

  Tensor grad_q_heads =
      this->get_tensor({batch_size, num_heads_, seq_len, head_dim_}, DType_t::BF16);
  Tensor grad_k_heads =
      this->get_tensor({batch_size, num_heads_, seq_len, head_dim_}, DType_t::BF16);
  Tensor grad_v_heads =
      this->get_tensor({batch_size, num_heads_, seq_len, head_dim_}, DType_t::BF16);

  // Get forward pass tensors in FP16 head layout
  Tensor q_heads = this->get_tensor({batch_size, num_heads_, seq_len, head_dim_}, DType_t::BF16);
  Tensor k_heads = this->get_tensor({batch_size, num_heads_, seq_len, head_dim_}, DType_t::BF16);
  Tensor v_heads = this->get_tensor({batch_size, num_heads_, seq_len, head_dim_}, DType_t::BF16);

  Tensor attn_heads = this->get_tensor({batch_size, num_heads_, seq_len, head_dim_}, DType_t::BF16);

  {
    // Recompute Q, K, V from cached input (trading compute for memory)
    Tensor q = q_proj_.forward({input}, residuals["q_proj"])[0];
    Tensor k = k_proj_.forward({input}, residuals["k_proj"])[0];
    Tensor v = v_proj_.forward({input}, residuals["v_proj"])[0];

    DISPATCH_DTYPE(io_dtype_, T, {
      create_cuda_task(defaultFlowHandle, cuda::permute_heads<T, bf16>, q.data_as<T>(),
                       q_heads.data_as<bf16>(), batch_size, seq_len, num_heads_, head_dim_);
      create_cuda_task(defaultFlowHandle, cuda::permute_heads<T, bf16>, k.data_as<T>(),
                       k_heads.data_as<bf16>(), batch_size, seq_len, num_heads_, head_dim_);
      create_cuda_task(defaultFlowHandle, cuda::permute_heads<T, bf16>, v.data_as<T>(),
                       v_heads.data_as<bf16>(), batch_size, seq_len, num_heads_, head_dim_);
      create_cuda_task(defaultFlowHandle, cuda::permute_heads<T, bf16>, attn_out.data_as<T>(),
                       attn_heads.data_as<bf16>(), batch_size, seq_len, num_heads_, head_dim_);
    });
  }

  {
    size_t workspace_size = stats.bwd_workspace_size;
    Tensor workspace = this->get_tensor({workspace_size}, DType_t::BYTE);

    // Run backward pass
    DISPATCH_ON_3_DTYPES_TO_METHOD(flash_attention_backward_task, fe_handle, stats, q_heads,
                                   k_heads, v_heads, attn_heads, grad_attn_heads, stats_tensor,
                                   grad_q_heads, grad_k_heads, grad_v_heads, workspace,
                                   defaultFlowHandle);
  }

  // Convert gradients back from head layout

  DISPATCH_DTYPE(io_dtype_, T, {
    create_cuda_task(defaultFlowHandle, cuda::permute_heads<bf16, T>, grad_q_heads.data_as<bf16>(),
                     grad_q.data_as<T>(), batch_size, num_heads_, seq_len, head_dim_);
    create_cuda_task(defaultFlowHandle, cuda::permute_heads<bf16, T>, grad_k_heads.data_as<bf16>(),
                     grad_k.data_as<T>(), batch_size, num_heads_, seq_len, head_dim_);
    create_cuda_task(defaultFlowHandle, cuda::permute_heads<bf16, T>, grad_v_heads.data_as<bf16>(),
                     grad_v.data_as<T>(), batch_size, num_heads_, seq_len, head_dim_);
  });

  allocator_->flip();

  // Backprop through separate Q, K, V projections
  Tensor grad_q_in = q_proj_.backward({grad_q}, residuals["q_proj"])[0];
  Tensor grad_k_in = k_proj_.backward({grad_k}, residuals["k_proj"])[0];
  Tensor grad_v_in = v_proj_.backward({grad_v}, residuals["v_proj"])[0];

  // Sum the gradients
  size_t size = grad_q_in.size();

  DISPATCH_IO_DTYPE(ops::add, grad_q_in.data_ptr(), grad_k_in.data_ptr(), grad_input.data_ptr(),
                    size, defaultFlowHandle);
  DISPATCH_IO_DTYPE(ops::add, grad_input.data_ptr(), grad_v_in.data_ptr(), grad_input.data_ptr(),
                    size, defaultFlowHandle);

  return grad_input;
}
#endif

Vec<Tensor> FlashAttentionBlockImpl::backward_impl(const Vec<Tensor> &grad_outputs,
                                                   Residuals &residuals) {
  const Tensor &grad_output = grad_outputs[0];

#ifdef USE_CUDNN
  if (get_engine_type() == EngineType::CUDA) {
    return {cudnn_backward(grad_output, residuals)};
  } else
#endif
  {
    throw std::runtime_error("CPU implementation for FlashAttentionBlock backward not implemented");
  }
}

LayerConfig FlashAttentionBlockImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("embed_dim", embed_dim_);
  config.set("num_heads", num_heads_);
  return config;
}

Vec<Vec<size_t>> FlashAttentionBlockImpl::output_shapes(
    const Vec<Vec<size_t>> &input_shapes) const {
  return input_shapes;
}

std::shared_ptr<FlashAttentionBlockImpl> FlashAttentionBlockImpl::create_from_config(
    const LayerConfig &config) {
  size_t embed_dim = config.get<size_t>("embed_dim");
  size_t num_heads = config.get<size_t>("num_heads");
  bool is_causal = config.get<bool>("is_causal", true);
  return std::make_shared<FlashAttentionBlockImpl>(embed_dim, num_heads, is_causal, config.name);
}

}  // namespace tunx
