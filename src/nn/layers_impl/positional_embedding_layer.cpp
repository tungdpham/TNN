/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/layers_impl/positional_embedding_layer.hpp"

#include <cmath>
#include <stdexcept>

namespace synet {

PositionalEmbeddingLayerImpl::PositionalEmbeddingLayerImpl(size_t embed_dim, size_t seq_len,
                                                           const std::string &name)
    : SISOLayerImpl(name),
      embed_dim_(embed_dim),
      seq_len_(seq_len) {}

void PositionalEmbeddingLayerImpl::init_impl() {
  float bound = static_cast<float>(1.0 / std::sqrt(static_cast<double>(embed_dim_)));

  if (this->use_seed_) {
    pos_embedding_.fill_random_normal(-bound, bound, this->srand_seed_);
  } else {
    pos_embedding_.fill_random_normal(-bound, bound);
  }

  pos_embedding_gradients_.fill(0.0f);
}

Tensor PositionalEmbeddingLayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  const auto &shape = input.shape();
  if (shape.size() < 2) {
    throw std::runtime_error("PositionalEmbeddingLayerImpl: Input tensor must be at least 2D");
  }

  size_t last_dim = shape.back();
  size_t second_last_dim = shape[shape.size() - 2];

  if (last_dim != embed_dim_) {
    throw std::runtime_error("PositionalEmbeddingLayerImpl: Input last dim (" +
                             std::to_string(last_dim) + ") must match embed_dim (" +
                             std::to_string(embed_dim_) + ")");
  }
  if (second_last_dim != seq_len_) {
    throw std::runtime_error("PositionalEmbeddingLayerImpl: Input sequence length (" +
                             std::to_string(second_last_dim) + ") must match seq_len (" +
                             std::to_string(seq_len_) + ")");
  }

  Tensor output = get_tensor(shape, io_dtype_);

  DISPATCH_ON_3_DTYPES_TO_METHOD(add_positional_embedding, input, output, pos_embedding_,
                                 this->flow_handle_);

  return output;
}

Tensor PositionalEmbeddingLayerImpl::backward_impl(const Tensor &grad_output,
                                                   Residuals &residuals) {
  const auto &shape = grad_output.shape();
  if (shape.size() < 2) {
    throw std::runtime_error("PositionalEmbeddingLayerImpl: Gradient tensor must be at least 2D");
  }

  size_t last_dim = shape.back();
  size_t second_last_dim = shape[shape.size() - 2];

  if (last_dim != embed_dim_) {
    throw std::runtime_error("PositionalEmbeddingLayerImpl: Gradient last dim (" +
                             std::to_string(last_dim) + ") must match embed_dim (" +
                             std::to_string(embed_dim_) + ")");
  }
  if (second_last_dim != seq_len_) {
    throw std::runtime_error("PositionalEmbeddingLayerImpl: Gradient sequence length (" +
                             std::to_string(second_last_dim) + ") must match seq_len (" +
                             std::to_string(seq_len_) + ")");
  }

  Tensor grad_input = get_tensor(shape, io_dtype_);

  grad_output.copy_to(grad_input);

  DISPATCH_ON_3_DTYPES_TO_METHOD(accumulate_pos_gradients, grad_output, pos_embedding_gradients_,
                                 this->flow_handle_);

  return grad_input;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> PositionalEmbeddingLayerImpl::add_positional_embedding(
    const Tensor &input, Tensor &output, const Tensor &pos_embedding, flowHandle_t handle) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "PositionalEmbeddingLayerImpl mixed dtype dispatch not implemented "
        "(io/param/compute must match).");
  }
  if (input.data_type() != dtype_of<IO_T>() || output.data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error(
        "PositionalEmbeddingLayerImpl IO tensor dtype mismatch with dispatch IO_T");
  }
  if (pos_embedding.data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error(
        "PositionalEmbeddingLayerImpl pos_embedding dtype mismatch with dispatch Param_T");
  }

  size_t sample_size = seq_len_ * embed_dim_;
  size_t batch_size = 1;
  const auto &shape = input.shape();
  for (size_t i = 0; i + 2 < shape.size(); ++i) {
    batch_size *= shape[i];
  }

  if (get_engine_type() == EngineType::CPU) {
    // For CPU, we need to manually loop over batches and add
    for (size_t i = 0; i < batch_size; ++i) {
      create_cpu_task(handle, ops::cpu::add<Compute_T>,
                      input.data_as<Compute_T>() + i * sample_size,
                      pos_embedding.data_as<Compute_T>(),
                      output.data_as<Compute_T>() + i * sample_size, sample_size);
    }
    return nullptr;
  }
#ifdef USE_CUDA
  else if (get_engine_type() == EngineType::CUDA) {
    // For GPU, we need to manually loop over batches and add
    for (size_t i = 0; i < batch_size; ++i) {
      create_cuda_task(handle, ops::cuda::cuda_add<Compute_T>,
                       input.data_as<Compute_T>() + i * sample_size,
                       pos_embedding.data_as<Compute_T>(),
                       output.data_as<Compute_T>() + i * sample_size, sample_size);
    }
    return nullptr;
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for add_positional_embedding");
  }
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> PositionalEmbeddingLayerImpl::accumulate_pos_gradients(
    const Tensor &grad_output, Tensor &pos_embedding_gradients, flowHandle_t handle) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "PositionalEmbeddingLayerImpl mixed dtype dispatch not implemented "
        "(io/param/compute must match).");
  }
  if (grad_output.data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error(
        "PositionalEmbeddingLayerImpl grad_output dtype mismatch with dispatch IO_T");
  }
  if (pos_embedding_gradients.data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error(
        "PositionalEmbeddingLayerImpl pos_embedding_gradients dtype mismatch with dispatch "
        "Param_T");
  }

  size_t sample_size = seq_len_ * embed_dim_;
  size_t batch_size = 1;
  const auto &shape = grad_output.shape();
  for (size_t i = 0; i + 2 < shape.size(); ++i) {
    batch_size *= shape[i];
  }

  if (get_engine_type() == EngineType::CPU) {
    for (size_t i = 0; i < batch_size; ++i) {
      create_cpu_task(handle, ops::cpu::add<Compute_T>,
                      pos_embedding_gradients.data_as<Compute_T>(),
                      grad_output.data_as<Compute_T>() + i * sample_size,
                      pos_embedding_gradients.data_as<Compute_T>(), sample_size);
    }
    return nullptr;
  }
#ifdef USE_CUDA
  else if (get_engine_type() == EngineType::CUDA) {
    for (size_t i = 0; i < batch_size; ++i) {
      create_cuda_task(handle, ops::cuda::cuda_add<Compute_T>,
                       pos_embedding_gradients.data_as<Compute_T>(),
                       grad_output.data_as<Compute_T>() + i * sample_size,
                       pos_embedding_gradients.data_as<Compute_T>(), sample_size);
    }
    return nullptr;
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for accumulate_pos_gradients");
  }
}

LayerConfig PositionalEmbeddingLayerImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("embed_dim", embed_dim_);
  config.set("seq_len", seq_len_);
  return config;
}

Vec<size_t> PositionalEmbeddingLayerImpl::compute_output_shape(
    const Vec<size_t> &input_shape) const {
  return input_shape;
}

std::shared_ptr<PositionalEmbeddingLayerImpl> PositionalEmbeddingLayerImpl::create_from_config(
    const LayerConfig &config) {
  size_t embed_dim = config.get<size_t>("embed_dim");
  size_t seq_len = config.get<size_t>("seq_len");
  return std::make_shared<PositionalEmbeddingLayerImpl>(embed_dim, seq_len, config.name);
}

}  // namespace synet
