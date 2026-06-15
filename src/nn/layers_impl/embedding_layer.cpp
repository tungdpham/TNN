/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/embedding_layer.hpp"

#include "device/task.hpp"
#include "nn/layers_impl/cpu/embedding_ops.hpp"
#include "type/type.hpp"
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/embedding_ops.hpp"
#endif

#include <cmath>
#include <memory>
#include <stdexcept>

namespace synet {

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

  if (this->use_seed_) {
    weight_.fill_random_normal(0, stddev, this->srand_seed_);
  } else {
    weight_.fill_random_normal(0, stddev);
  }

  // Set padding idx to zeros if valid
  if (padding_idx_ < vocab_size_) {
    // Zero out the padding index row
    for (size_t i = 0; i < embed_dim_; ++i) {
      DISPATCH_DTYPE(weight_.data_type(), T, weight_.at<T>({padding_idx_, i}) = 0.0f);
    }
  }

  weight_gradients_.fill(0.0f);
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

  DISPATCH_ON_3_DTYPES_TO_METHOD(compute_forward_impl, input, weight_, output, num_tokens,
                                 vocab_size_, embed_dim_, padding_idx_, this->flow_handle_);

  return output;
}

Tensor EmbeddingLayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  const Tensor &input = residuals["input"];

  Tensor grad_input = get_tensor(input.shape(), io_dtype_);
  grad_input.fill(0);

  size_t num_tokens = input.size();

  DISPATCH_ON_3_DTYPES_TO_METHOD(compute_backward_impl, input, grad_output, weight_gradients_,
                                 num_tokens, vocab_size_, embed_dim_, padding_idx_,
                                 this->flow_handle_);

  return grad_input;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> EmbeddingLayerImpl::compute_forward_impl(
    const Tensor &input, const Tensor &weight, Tensor &output, size_t num_indices,
    size_t vocab_size, size_t embed_dim, size_t padding_idx, flowHandle_t handle) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "EmbeddingLayerImpl mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (input.data_type() != dtype_of<IO_T>() || output.data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("EmbeddingLayerImpl IO tensor dtype mismatch with dispatch IO_T");
  }
  if (weight.data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error(
        "EmbeddingLayerImpl weight tensor dtype mismatch with dispatch Param_T");
  }

  if (input.device_type() == DeviceType::CPU) {
    return create_cpu_task(handle, cpu::embedding::run_forward<Compute_T>,
                           input.data_as<Compute_T>(), weight.data_as<Compute_T>(),
                           output.data_as<Compute_T>(), num_indices, vocab_size, embed_dim,
                           padding_idx);
  }
#ifdef USE_CUDA
  else if (input.device_type() == DeviceType::GPU) {
    return create_cuda_task(handle, cuda::embedding::run_forward<Compute_T>,
                            input.data_as<Compute_T>(), weight.data_as<Compute_T>(),
                            output.data_as<Compute_T>(), num_indices, vocab_size, embed_dim,
                            padding_idx);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for embedding forward");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> EmbeddingLayerImpl::compute_backward_impl(
    const Tensor &input, const Tensor &grad_output, Tensor &weight_gradients, size_t num_indices,
    size_t vocab_size, size_t embed_dim, size_t padding_idx, flowHandle_t handle) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "EmbeddingLayerImpl mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (input.data_type() != dtype_of<IO_T>() || grad_output.data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("EmbeddingLayerImpl IO tensor dtype mismatch with dispatch IO_T");
  }
  if (weight_gradients.data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error(
        "EmbeddingLayerImpl weight grad_output dtype mismatch with dispatch Param_T");
  }

  if (input.device_type() == DeviceType::CPU) {
    return create_cpu_task(handle, cpu::embedding::run_backward<Compute_T>,
                           input.data_as<Compute_T>(), grad_output.data_as<Compute_T>(),
                           weight_gradients.data_as<Compute_T>(), num_indices, vocab_size,
                           embed_dim, padding_idx);
  }
#ifdef USE_CUDA
  else if (input.device_type() == DeviceType::GPU) {
    return create_cuda_task(handle, cuda::embedding::run_backward<Compute_T>,
                            input.data_as<Compute_T>(), grad_output.data_as<Compute_T>(),
                            weight_gradients.data_as<Compute_T>(), num_indices, vocab_size,
                            embed_dim, padding_idx);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for embedding backward");
  }
  return nullptr;
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

}  // namespace synet
