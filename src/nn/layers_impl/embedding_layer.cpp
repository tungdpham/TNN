/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/embedding_layer.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

#include "nn/engines/iengine.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_ops.hpp"
#include "type/type.hpp"

namespace tunx {

EmbeddingLayerImpl::EmbeddingLayerImpl(size_t vocab_size, size_t embed_dim, const std::string &name,
                                       size_t padding_idx)
    : SISOLayerImpl(name),
      vocab_size_(vocab_size),
      embed_dim_(embed_dim) {
  if (padding_idx == static_cast<size_t>(-1)) {
    padding_idx_ = vocab_size_;
  } else {
    padding_idx_ = padding_idx;
  }
}

void EmbeddingLayerImpl::init_impl() {
  float stddev = static_cast<float>(1.0 / std::sqrt(static_cast<double>(embed_dim_)));
  long long seed = this->use_seed_ ? this->srand_seed_
                                   : std::chrono::system_clock::now().time_since_epoch().count();
  fill_normal(weight_, 0, stddev, seed);

  // Set padding idx to zeros if valid
  if (padding_idx_ < vocab_size_) {
    // Zero out the padding index row
    for (size_t i = 0; i < embed_dim_; ++i) {
      DISPATCH_DTYPE(weight_.dtype(), T, weight_.at<T>({padding_idx_, i}) = 0.0f);
    }
  }

  fill(grad_weights_, 0.0f);
}

Tensor EmbeddingLayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  if (this->is_training_) {
    residuals["input"] = input;
  }

  size_t num_tokens = input.size();
  if (num_tokens == 0) return Tensor();

  Vec<size_t> out_shape = input.shape();
  out_shape.push_back(embed_dim_);
  Tensor output = get_tensor(out_shape, io_dtype_);

  EmbeddingStats stats{
      .num_indices = num_tokens,
      .vocab_size = vocab_size_,
      .embed_dim = embed_dim_,
      .padding_idx = padding_idx_,
  };

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  WorkspaceReq ws_req = engine_->query_embedding_graph(backend_handle_, stats, type_desc);
  Tensor ws = get_tensor({ws_req.fwd_workspace}, DType_t::BYTE);

  engine_->embedding_fwd(backend_handle_, stats, input.data_as<void>(), weight_.data_as<void>(),
                         output.data_as<void>(), ws.data_as<void>(), type_desc);

  return output;
}

Tensor EmbeddingLayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  const Tensor &input = residuals["input"];

  Tensor grad_input = get_tensor(input.shape(), io_dtype_);
  fill(grad_input, 0.0f);

  size_t num_tokens = input.size();

  EmbeddingStats stats{
      .num_indices = num_tokens,
      .vocab_size = vocab_size_,
      .embed_dim = embed_dim_,
      .padding_idx = padding_idx_,
  };

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  WorkspaceReq ws_req = engine_->query_embedding_graph(backend_handle_, stats, type_desc);
  Tensor ws = get_tensor({ws_req.bwd_workspace}, DType_t::BYTE);

  Tensor grad_weights_next({vocab_size_, embed_dim_}, param_dtype_);

  engine_->embedding_bwd(backend_handle_, stats, grad_output.data_as<void>(), input.data_as<void>(),
                         grad_weights_.data_as<void>(), ws.data_as<void>(), type_desc);

  grad_weights_ = grad_weights_next;

  return grad_input;
}

Vec<size_t> EmbeddingLayerImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  Vec<size_t> out = input_shape;
  out.push_back(embed_dim_);
  return out;
}

LayerConfig EmbeddingLayerImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("vocab_size", vocab_size_);
  config.set("embed_dim", embed_dim_);
  config.set("padding_idx", padding_idx_);
  return config;
}

std::shared_ptr<EmbeddingLayerImpl> EmbeddingLayerImpl::create_from_config(
    const LayerConfig &config) {
  size_t vocab_size = config.get<size_t>("vocab_size");
  size_t embed_dim = config.get<size_t>("embed_dim");
  size_t padding_idx = config.get<size_t>("padding_idx", static_cast<size_t>(-1));
  return std::make_shared<EmbeddingLayerImpl>(vocab_size, embed_dim, config.name, padding_idx);
}

}  // namespace tunx
