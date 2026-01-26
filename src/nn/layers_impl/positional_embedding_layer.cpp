/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/layers_impl/positional_embedding_layer.hpp"

#include <cmath>
#include <stdexcept>

namespace tnn {

PositionalEmbeddingLayer::PositionalEmbeddingLayer(size_t embed_dim, size_t seq_len,
                                                   const std::string &name)
    : ParameterizedLayer(name), embed_dim_(embed_dim), seq_len_(seq_len) {}

void PositionalEmbeddingLayer::init_params() {
  pos_embedding_ = make_param_tensor({seq_len_, embed_dim_});
  pos_embedding_gradients_ = make_param_tensor({seq_len_, embed_dim_});

  float bound = static_cast<float>(1.0 / std::sqrt(static_cast<double>(embed_dim_)));

  if (this->use_seed_) {
    pos_embedding_->fill_random_uniform(-bound, bound, this->srand_seed_);
  } else {
    pos_embedding_->fill_random_uniform(-bound, bound);
  }
  pos_embedding_gradients_->fill(0.0f);
}

void PositionalEmbeddingLayer::forward_impl(const Tensor &input, Tensor &output, size_t mb_id) {
  const auto &shape = input->shape();
  if (shape.size() < 2) {
    throw std::runtime_error("PositionalEmbeddingLayer: Input tensor must be at least 2D");
  }

  size_t last_dim = shape.back();
  size_t second_last_dim = shape[shape.size() - 2];

  if (last_dim != embed_dim_) {
    throw std::runtime_error("PositionalEmbeddingLayer: Input last dim (" +
                             std::to_string(last_dim) + ") must match embed_dim (" +
                             std::to_string(embed_dim_) + ")");
  }
  if (second_last_dim != seq_len_) {
    throw std::runtime_error("PositionalEmbeddingLayer: Input sequence length (" +
                             std::to_string(second_last_dim) + ") must match seq_len (" +
                             std::to_string(seq_len_) + ")");
  }

  output->ensure(shape);

  DISPATCH_ON_3_DTYPES_TO_METHOD(add_positional_embedding, input, output, pos_embedding_,
                                 "default");
}

void PositionalEmbeddingLayer::backward_impl(const Tensor &gradient, Tensor &grad_input,
                                             size_t mb_id) {
  const auto &shape = gradient->shape();
  if (shape.size() < 2) {
    throw std::runtime_error("PositionalEmbeddingLayer: Gradient tensor must be at least 2D");
  }

  size_t last_dim = shape.back();
  size_t second_last_dim = shape[shape.size() - 2];

  if (last_dim != embed_dim_) {
    throw std::runtime_error("PositionalEmbeddingLayer: Gradient last dim (" +
                             std::to_string(last_dim) + ") must match embed_dim (" +
                             std::to_string(embed_dim_) + ")");
  }
  if (second_last_dim != seq_len_) {
    throw std::runtime_error("PositionalEmbeddingLayer: Gradient sequence length (" +
                             std::to_string(second_last_dim) + ") must match seq_len (" +
                             std::to_string(seq_len_) + ")");
  }

  grad_input->ensure(shape);

  gradient->copy_to(grad_input);

  DISPATCH_ON_3_DTYPES_TO_METHOD(accumulate_pos_gradients, gradient, pos_embedding_gradients_,
                                 "default");
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task>
PositionalEmbeddingLayer::add_positional_embedding(const Tensor &input, Tensor &output,
                                                   const Tensor &pos_embedding,
                                                   const std::string &flow_id) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error("PositionalEmbeddingLayer mixed dtype dispatch not implemented "
                             "(io/param/compute must match).");
  }
  if (input->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error(
        "PositionalEmbeddingLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (pos_embedding->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error(
        "PositionalEmbeddingLayer pos_embedding dtype mismatch with dispatch Param_T");
  }

  size_t sample_size = seq_len_ * embed_dim_;
  size_t batch_size = 1;
  const auto &shape = input->shape();
  for (size_t i = 0; i + 2 < shape.size(); ++i) {
    batch_size *= shape[i];
  }

  if (this->device_->device_type() == DeviceType::CPU) {
    // For CPU, we need to manually loop over batches and add
    for (size_t i = 0; i < batch_size; ++i) {
      create_cpu_task(flow_id, ops::cpu::add<Compute_T>,
                      input->data_as<Compute_T>() + i * sample_size,
                      pos_embedding->data_as<Compute_T>(),
                      output->data_as<Compute_T>() + i * sample_size, sample_size);
    }
    return nullptr;
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    // For GPU, we need to manually loop over batches and add
    for (size_t i = 0; i < batch_size; ++i) {
      create_gpu_task(flow_id, ops::cuda::cuda_add<Compute_T>,
                      input->data_as<Compute_T>() + i * sample_size,
                      pos_embedding->data_as<Compute_T>(),
                      output->data_as<Compute_T>() + i * sample_size, sample_size);
    }
    return nullptr;
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for add_positional_embedding");
  }
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> PositionalEmbeddingLayer::accumulate_pos_gradients(
    const Tensor &gradient, Tensor &pos_embedding_gradients, const std::string &flow_id) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error("PositionalEmbeddingLayer mixed dtype dispatch not implemented "
                             "(io/param/compute must match).");
  }
  if (gradient->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("PositionalEmbeddingLayer gradient dtype mismatch with dispatch IO_T");
  }
  if (pos_embedding_gradients->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error(
        "PositionalEmbeddingLayer pos_embedding_gradients dtype mismatch with dispatch Param_T");
  }

  size_t sample_size = seq_len_ * embed_dim_;
  size_t batch_size = 1;
  const auto &shape = gradient->shape();
  for (size_t i = 0; i + 2 < shape.size(); ++i) {
    batch_size *= shape[i];
  }

  if (this->device_->device_type() == DeviceType::CPU) {
    for (size_t i = 0; i < batch_size; ++i) {
      create_cpu_task(flow_id, ops::cpu::add<Compute_T>,
                      pos_embedding_gradients->data_as<Compute_T>(),
                      gradient->data_as<Compute_T>() + i * sample_size,
                      pos_embedding_gradients->data_as<Compute_T>(), sample_size);
    }
    return nullptr;
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    for (size_t i = 0; i < batch_size; ++i) {
      create_gpu_task(flow_id, ops::cuda::cuda_add<Compute_T>,
                      pos_embedding_gradients->data_as<Compute_T>(),
                      gradient->data_as<Compute_T>() + i * sample_size,
                      pos_embedding_gradients->data_as<Compute_T>(), sample_size);
    }
    return nullptr;
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for accumulate_pos_gradients");
  }
}

uint64_t PositionalEmbeddingLayer::forward_flops(const std::vector<size_t> &input_shape) const {
  return 0;
}

uint64_t PositionalEmbeddingLayer::backward_flops(const std::vector<size_t> &input_shape) const {
  return 0;
}

LayerConfig PositionalEmbeddingLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.parameters["embed_dim"] = embed_dim_;
  config.parameters["seq_len"] = seq_len_;
  return config;
}

std::unique_ptr<Layer> PositionalEmbeddingLayer::clone() const {
  return std::make_unique<PositionalEmbeddingLayer>(embed_dim_, seq_len_, this->name_);
}

std::vector<size_t>
PositionalEmbeddingLayer::compute_output_shape(const std::vector<size_t> &input_shape) const {
  return input_shape;
}

void PositionalEmbeddingLayer::collect_parameters(std::vector<Tensor> &params) {
  params.push_back(pos_embedding_);
}

void PositionalEmbeddingLayer::collect_gradients(std::vector<Tensor> &grads) {
  grads.push_back(pos_embedding_gradients_);
}

std::unique_ptr<PositionalEmbeddingLayer>
PositionalEmbeddingLayer::create_from_config(const LayerConfig &config) {
  size_t embed_dim = config.get<size_t>("embed_dim");
  size_t seq_len = config.get<size_t>("seq_len");
  return std::make_unique<PositionalEmbeddingLayer>(embed_dim, seq_len, config.name);
}

} // namespace tnn
