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

namespace tnn {

EmbeddingLayer::EmbeddingLayer(size_t vocab_size, size_t embed_dim, const std::string &name,
                               size_t padding_idx)
    : ParameterizedLayer(name), vocab_size_(vocab_size), embed_dim_(embed_dim) {
  if (padding_idx == static_cast<size_t>(-1)) {
    padding_idx_ = vocab_size_;
  } else {
    padding_idx_ = padding_idx;
  }
}

void EmbeddingLayer::init_params() {
  // weight shape: [vocab_size, embed_dim]
  weight_ = make_param_tensor({vocab_size_, embed_dim_});
  grad_weight_ = make_param_tensor({vocab_size_, embed_dim_});

  if (this->use_seed_) {
    weight_->fill_random_normal(0.0, 0.02, this->srand_seed_);
  } else {
    weight_->fill_random_normal(0.0, 0.02);
  }
  grad_weight_->fill(0);
}

void EmbeddingLayer::forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id) {
  if (this->is_training_) {
    auto &cached_input = micro_batch_inputs_[mb_id];
    if (!cached_input) {
      cached_input = make_tensor(input->data_type(), input->shape(), this->device_);
    } else {
      cached_input->ensure(input->shape());
    }
    input->copy_to(cached_input);
  }

  size_t num_tokens = input->size();
  if (num_tokens == 0) return;

  std::vector<size_t> out_shape = input->shape();
  out_shape.push_back(embed_dim_);
  output->ensure(out_shape);

  DISPATCH_ON_3_DTYPES_TO_METHOD(compute_forward_impl, input, weight_, output, num_tokens,
                                 vocab_size_, embed_dim_, padding_idx_, "default");
}

void EmbeddingLayer::backward_impl(const ConstTensor &gradient, const Tensor &grad_input,
                                   size_t mb_id) {
  auto it = micro_batch_inputs_.find(mb_id);
  if (it == micro_batch_inputs_.end()) {
    throw std::runtime_error("EmbeddingLayer::backward: No cached input for mb_id " +
                             std::to_string(mb_id));
  }

  const ConstTensor &input = it->second;

  grad_input->ensure(input->shape());
  grad_input->fill(0);

  size_t num_tokens = input->size();

  DISPATCH_ON_3_DTYPES_TO_METHOD(compute_backward_impl, input, gradient, grad_weight_, num_tokens,
                                 vocab_size_, embed_dim_, padding_idx_, "default");
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> EmbeddingLayer::compute_forward_impl(
    const ConstTensor &input, const ConstTensor &weight, const Tensor &output, size_t num_indices,
    size_t vocab_size, size_t embed_dim, size_t padding_idx, const std::string &flow_id) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "EmbeddingLayer mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (input->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("EmbeddingLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (weight->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("EmbeddingLayer weight tensor dtype mismatch with dispatch Param_T");
  }

  if (input->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::embedding::compute_embedding_forward<Compute_T>,
                           input->data_as<Compute_T>(), weight->data_as<Compute_T>(),
                           output->data_as<Compute_T>(), num_indices, vocab_size, embed_dim,
                           padding_idx);
  }
#ifdef USE_CUDA
  else if (input->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::embedding::compute_embedding_forward<Compute_T>,
                            input->data_as<Compute_T>(), weight->data_as<Compute_T>(),
                            output->data_as<Compute_T>(), num_indices, vocab_size, embed_dim,
                            padding_idx);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for embedding forward");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> EmbeddingLayer::compute_backward_impl(const ConstTensor &input,
                                                            const ConstTensor &gradient,
                                                            const Tensor &grad_weight,
                                                            size_t num_indices, size_t vocab_size,
                                                            size_t embed_dim, size_t padding_idx,
                                                            const std::string &flow_id) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "EmbeddingLayer mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (input->data_type() != dtype_of<IO_T>() || gradient->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("EmbeddingLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (grad_weight->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("EmbeddingLayer weight gradient dtype mismatch with dispatch Param_T");
  }

  if (input->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::embedding::compute_embedding_backward<Compute_T>,
                           input->data_as<Compute_T>(), gradient->data_as<Compute_T>(),
                           grad_weight->data_as<Compute_T>(), num_indices, vocab_size, embed_dim,
                           padding_idx);
  }
#ifdef USE_CUDA
  else if (input->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::embedding::compute_embedding_backward<Compute_T>,
                            input->data_as<Compute_T>(), gradient->data_as<Compute_T>(),
                            grad_weight->data_as<Compute_T>(), num_indices, vocab_size, embed_dim,
                            padding_idx);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for embedding backward");
  }
  return nullptr;
}

void EmbeddingLayer::collect_parameters(std::vector<Tensor> &params) { params.push_back(weight_); }

void EmbeddingLayer::collect_gradients(std::vector<Tensor> &grads) {
  grads.push_back(grad_weight_);
}

std::vector<size_t> EmbeddingLayer::compute_output_shape(
    const std::vector<size_t> &input_shape) const {
  std::vector<size_t> out = input_shape;
  out.push_back(embed_dim_);
  return out;
}

uint64_t EmbeddingLayer::forward_flops(const std::vector<size_t> &input_shape) const { return 0; }

uint64_t EmbeddingLayer::backward_flops(const std::vector<size_t> &input_shape) const { return 0; }

LayerConfig EmbeddingLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.parameters["vocab_size"] = vocab_size_;
  config.parameters["embed_dim"] = embed_dim_;
  config.parameters["padding_idx"] = padding_idx_;
  return config;
}

std::unique_ptr<Layer> EmbeddingLayer::clone() const {
  return std::make_unique<EmbeddingLayer>(vocab_size_, embed_dim_, this->name_, padding_idx_);
}

size_t EmbeddingLayer::cached_memory_bytes() const {
  size_t total_bytes = 0;
  for (const auto &pair : micro_batch_inputs_) {
    size_t elem_size = get_dtype_size(pair.second->data_type());
    total_bytes += pair.second->size() * elem_size;
  }
  total_bytes += Layer::cached_memory_bytes();
  return total_bytes;
}

std::unique_ptr<EmbeddingLayer> EmbeddingLayer::create_from_config(const LayerConfig &config) {
  size_t vocab_size = config.get<size_t>("vocab_size");
  size_t embed_dim = config.get<size_t>("embed_dim");
  size_t padding_idx = config.get<size_t>("padding_idx", static_cast<size_t>(-1));
  return std::make_unique<EmbeddingLayer>(vocab_size, embed_dim, config.name, padding_idx);
}

}  // namespace tnn
