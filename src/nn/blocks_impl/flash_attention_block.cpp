/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/blocks_impl/flash_attention_block.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "nn/layer.hpp"
#include "nn/stats/stats.hpp"
#include "ops/ops.hpp"
#include "type/type.hpp"
#include "utils/misc.hpp"

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

FlashAttentionBlockImpl::~FlashAttentionBlockImpl() {}

Vec<Tensor> FlashAttentionBlockImpl::forward_impl(const Vec<Tensor> &inputs, Residuals &residuals) {
  const Tensor &input = inputs[0];

  if (input.dims() != 3) {
    throw std::invalid_argument("FlashAttentionBlock: Input must be 3D (B, S, E) but got " +
                                std::to_string(input.dims()) + "D");
  }

  const auto &input_shape = input.shape();
  size_t batch_size = input_shape[0];
  size_t seq_len = input_shape[1];
  size_t embed_dim = input.dim(2);

  if (embed_dim != embed_dim_) {
    throw std::invalid_argument("FlashAttentionBlock: Input embed_dim mismatch");
  }

  AttentionStats stats{
      .batch_size = batch_size,
      .num_heads = num_heads_,
      .seq_len = seq_len,
      .head_dim = head_dim_,
      .attn_scale = 1.0f,
      .is_causal = is_causal_,
  };

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  if (this->is_training_) {
    residuals["input"] = input;
  }

  Tensor attn_out = this->get_tensor({batch_size, seq_len, embed_dim_}, io_dtype_);
  residuals["attn_out"] = attn_out;
  Tensor stats_tensor = this->get_tensor({batch_size, num_heads_, seq_len, 1}, DType_t::FP32);
  residuals["stats_tensor"] = stats_tensor;

  allocator_->flip();

  Tensor attn_heads = this->get_tensor({batch_size, num_heads_, seq_len, head_dim_},
                                       io_dtype_ == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);
  Tensor q_heads = this->get_tensor({batch_size, num_heads_, seq_len, head_dim_},
                                    io_dtype_ == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);
  Tensor k_heads = this->get_tensor({batch_size, num_heads_, seq_len, head_dim_},
                                    io_dtype_ == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);
  Tensor v_heads = this->get_tensor({batch_size, num_heads_, seq_len, head_dim_},
                                    io_dtype_ == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);

  Tensor q = q_proj_.forward({input}, residuals["q_proj"])[0];
  Tensor k = k_proj_.forward({input}, residuals["k_proj"])[0];
  Tensor v = v_proj_.forward({input}, residuals["v_proj"])[0];

  DISPATCH_DTYPE(io_dtype_, T, {
    ops::permute_heads<T, bf16>(q.data_ptr(), q_heads.data_ptr(), batch_size, seq_len, num_heads_,
                                head_dim_);
    ops::permute_heads<T, bf16>(k.data_ptr(), k_heads.data_ptr(), batch_size, seq_len, num_heads_,
                                head_dim_);
    ops::permute_heads<T, bf16>(v.data_ptr(), v_heads.data_ptr(), batch_size, seq_len, num_heads_,
                                head_dim_);
  });

  {
    WorkspaceReq ws_req = engine_->query_sdpa_graph(backend_handle_, stats, type_desc);
    size_t workspace_size = ws_req.fwd_workspace;
    Tensor workspace = this->get_tensor({workspace_size}, DType_t::BYTE);

    engine_->sdpa_fwd(backend_handle_, stats, q_heads.data_as<void>(), k_heads.data_as<void>(),
                      v_heads.data_as<void>(), attn_heads.data_as<void>(),
                      stats_tensor.data_as<void>(), workspace.data_as<void>(), type_desc);
  }

  DISPATCH_DTYPE(io_dtype_, T, {
    ops::permute_heads<bf16, T>(attn_heads.data_ptr(), attn_out.data_ptr(), batch_size, num_heads_,
                                seq_len, head_dim_);
  });

  allocator_->flip();

  Tensor output = out_proj_.forward({attn_out}, residuals)[0];
  return {output};
}

Vec<Tensor> FlashAttentionBlockImpl::backward_impl(const Vec<Tensor> &grad_outputs,
                                                   Residuals &residuals) {
  const Tensor &grad_output = grad_outputs[0];

  const auto &grad_shape = grad_output.shape();
  size_t batch_size = grad_shape[0];
  size_t seq_len = grad_shape[1];

  AttentionStats stats{
      .batch_size = batch_size,
      .num_heads = num_heads_,
      .seq_len = seq_len,
      .head_dim = head_dim_,
      .attn_scale = 1.0f,
      .is_causal = is_causal_,
  };

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  const Tensor &input = residuals["input"];
  Tensor &attn_out = residuals["attn_out"];
  Tensor &stats_tensor = residuals["stats_tensor"];

  Tensor grad_input = this->get_tensor({batch_size, seq_len, embed_dim_}, io_dtype_);

  allocator_->flip();

  Tensor grad_q = this->get_tensor({batch_size, seq_len, embed_dim_}, io_dtype_);
  Tensor grad_k = this->get_tensor({batch_size, seq_len, embed_dim_}, io_dtype_);
  Tensor grad_v = this->get_tensor({batch_size, seq_len, embed_dim_}, io_dtype_);

  Tensor grad_attn_heads =
      this->get_tensor({batch_size, num_heads_, seq_len, head_dim_},
                       io_dtype_ == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);

  {
    Tensor grad_attn_out = out_proj_.backward({grad_output}, residuals["out_proj"])[0];
    DISPATCH_DTYPE(io_dtype_, T, {
      if constexpr (std::is_same_v<T, float>) {
        ops::permute_heads<T, float>(grad_attn_out.data_ptr(), grad_attn_heads.data_ptr(),
                                     batch_size, seq_len, num_heads_, head_dim_);
      } else {
        ops::permute_heads<T, bf16>(grad_attn_out.data_ptr(), grad_attn_heads.data_ptr(),
                                    batch_size, seq_len, num_heads_, head_dim_);
      }
    });
  }

  Tensor grad_q_heads =
      this->get_tensor({batch_size, num_heads_, seq_len, head_dim_},
                       io_dtype_ == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);
  Tensor grad_k_heads =
      this->get_tensor({batch_size, num_heads_, seq_len, head_dim_},
                       io_dtype_ == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);
  Tensor grad_v_heads =
      this->get_tensor({batch_size, num_heads_, seq_len, head_dim_},
                       io_dtype_ == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);

  Tensor q_heads = this->get_tensor({batch_size, num_heads_, seq_len, head_dim_},
                                    io_dtype_ == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);
  Tensor k_heads = this->get_tensor({batch_size, num_heads_, seq_len, head_dim_},
                                    io_dtype_ == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);
  Tensor v_heads = this->get_tensor({batch_size, num_heads_, seq_len, head_dim_},
                                    io_dtype_ == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);

  Tensor attn_heads = this->get_tensor({batch_size, num_heads_, seq_len, head_dim_},
                                       io_dtype_ == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);

  {
    Tensor q = q_proj_.forward({input}, residuals["q_proj"])[0];
    Tensor k = k_proj_.forward({input}, residuals["k_proj"])[0];
    Tensor v = v_proj_.forward({input}, residuals["v_proj"])[0];

    DISPATCH_DTYPE(io_dtype_, T, {
      if constexpr (std::is_same_v<T, float>) {
        ops::permute_heads<T, float>(q.data_ptr(), q_heads.data_ptr(), batch_size, seq_len,
                                     num_heads_, head_dim_);
        ops::permute_heads<T, float>(k.data_ptr(), k_heads.data_ptr(), batch_size, seq_len,
                                     num_heads_, head_dim_);
        ops::permute_heads<T, float>(v.data_ptr(), v_heads.data_ptr(), batch_size, seq_len,
                                     num_heads_, head_dim_);
        ops::permute_heads<T, float>(attn_out.data_ptr(), attn_heads.data_ptr(), batch_size,
                                     seq_len, num_heads_, head_dim_);
      } else {
        ops::permute_heads<T, bf16>(q.data_ptr(), q_heads.data_ptr(), batch_size, seq_len,
                                    num_heads_, head_dim_);
        ops::permute_heads<T, bf16>(k.data_ptr(), k_heads.data_ptr(), batch_size, seq_len,
                                    num_heads_, head_dim_);
        ops::permute_heads<T, bf16>(v.data_ptr(), v_heads.data_ptr(), batch_size, seq_len,
                                    num_heads_, head_dim_);
        ops::permute_heads<T, bf16>(attn_out.data_ptr(), attn_heads.data_ptr(), batch_size, seq_len,
                                    num_heads_, head_dim_);
      }
    });
  }

  {
    WorkspaceReq ws_req = engine_->query_sdpa_graph(backend_handle_, stats, type_desc);
    size_t workspace_size = ws_req.bwd_workspace;
    Tensor workspace = this->get_tensor({workspace_size}, DType_t::BYTE);

    engine_->sdpa_bwd(backend_handle_, stats, q_heads.data_as<void>(), k_heads.data_as<void>(),
                      v_heads.data_as<void>(), attn_heads.data_as<void>(),
                      grad_attn_heads.data_as<void>(), stats_tensor.data_as<void>(),
                      grad_q_heads.data_as<void>(), grad_k_heads.data_as<void>(),
                      grad_v_heads.data_as<void>(), workspace.data_as<void>(), type_desc);
  }

  DISPATCH_DTYPE(io_dtype_, T, {
    ops::permute_heads<bf16, T>(grad_q_heads.data_ptr(), grad_q.data_ptr(), batch_size, num_heads_,
                                seq_len, head_dim_);
    ops::permute_heads<bf16, T>(grad_k_heads.data_ptr(), grad_k.data_ptr(), batch_size, num_heads_,
                                seq_len, head_dim_);
    ops::permute_heads<bf16, T>(grad_v_heads.data_ptr(), grad_v.data_ptr(), batch_size, num_heads_,
                                seq_len, head_dim_);
  });

  allocator_->flip();

  Tensor grad_q_in = q_proj_.backward({grad_q}, residuals["q_proj"])[0];
  Tensor grad_k_in = k_proj_.backward({grad_k}, residuals["k_proj"])[0];
  Tensor grad_v_in = v_proj_.backward({grad_v}, residuals["v_proj"])[0];

  size_t size = grad_q_in.size();

  DISPATCH_IO_DTYPE(ops::add, grad_q_in.data_ptr(), grad_k_in.data_ptr(), grad_input.data_ptr(),
                    size, defaultFlowHandle);
  DISPATCH_IO_DTYPE(ops::add, grad_input.data_ptr(), grad_v_in.data_ptr(), grad_input.data_ptr(),
                    size, defaultFlowHandle);

  return {grad_input};
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
