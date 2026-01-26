/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/layers_impl/class_token_layer.hpp"

#include "nn/layers_impl/cpu/class_token_ops.hpp"
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/class_token_ops.hpp"
#endif

#include <cmath>
#include <stdexcept>

namespace tnn {

ClassTokenLayer::ClassTokenLayer(size_t embed_dim, const std::string &name)
    : ParameterizedLayer(name), embed_dim_(embed_dim) {}

void ClassTokenLayer::init_params() {
  class_token_ = make_param_tensor({embed_dim_});
  class_token_gradients_ = make_param_tensor({embed_dim_});

  float bound = static_cast<float>(1.0 / std::sqrt(static_cast<double>(embed_dim_)));

  if (this->use_seed_) {
    class_token_->fill_random_uniform(-bound, bound, this->srand_seed_);
  } else {
    class_token_->fill_random_uniform(-bound, bound);
  }
  class_token_gradients_->fill(0.0f);
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> ClassTokenLayer::forward_task(const Tensor &input, Tensor &output,
                                                    const Tensor &class_token, size_t batch_size,
                                                    size_t seq_len, size_t embed_dim,
                                                    const std::string &flow_id) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "ClassTokenLayer mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (input->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("ClassTokenLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (class_token->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("ClassTokenLayer class_token dtype mismatch with dispatch Param_T");
  }

  if (this->device_->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::class_token_forward<Compute_T>,
                           input->data_as<Compute_T>(), class_token->data_as<Compute_T>(),
                           output->data_as<Compute_T>(), batch_size, seq_len, embed_dim);
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::class_token_forward<Compute_T>,
                            input->data_as<Compute_T>(), class_token->data_as<Compute_T>(),
                            output->data_as<Compute_T>(), batch_size, seq_len, embed_dim);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for class_token_forward");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> ClassTokenLayer::backward_task(const Tensor &gradient, Tensor &grad_input,
                                                     Tensor &class_token_gradients,
                                                     const Tensor &class_token, size_t batch_size,
                                                     size_t seq_len, size_t embed_dim,
                                                     const std::string &flow_id) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "ClassTokenLayer mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (gradient->data_type() != dtype_of<IO_T>() || grad_input->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("ClassTokenLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (class_token_gradients->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error(
        "ClassTokenLayer class_token_gradients dtype mismatch with dispatch Param_T");
  }

  if (this->device_->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::class_token_backward<Compute_T>,
                           gradient->data_as<Compute_T>(), grad_input->data_as<Compute_T>(),
                           class_token_gradients->data_as<Compute_T>(), batch_size, seq_len,
                           embed_dim);
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::class_token_backward<Compute_T>,
                            gradient->data_as<Compute_T>(), grad_input->data_as<Compute_T>(),
                            class_token_gradients->data_as<Compute_T>(), batch_size, seq_len,
                            embed_dim);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for class_token_backward");
  }
  return nullptr;
}

void ClassTokenLayer::forward_impl(const Tensor &input, Tensor &output, size_t mb_id) {
  if (input->dims() != 3) {
    throw std::runtime_error(
        "ClassTokenLayer: Input tensor must have 3 dimensions (Batch, Seq, Embed)");
  }
  size_t batch_size = input->dimension(0);
  size_t seq_len = input->dimension(1);
  size_t embed_dim = input->dimension(2);

  if (embed_dim != embed_dim_) {
    throw std::runtime_error("ClassTokenLayer: Input embed_dim must match layer embed_dim");
  }

  output->ensure({batch_size, seq_len + 1, embed_dim});

  DISPATCH_ON_3_DTYPES_TO_METHOD(forward_task, input, output, class_token_, batch_size, seq_len,
                                 embed_dim, "default");
}

void ClassTokenLayer::backward_impl(const Tensor &gradient, Tensor &grad_input, size_t mb_id) {
  if (gradient->dims() != 3) {
    throw std::runtime_error(
        "ClassTokenLayer: Gradient tensor must have 3 dimensions (Batch, Seq, Embed)");
  }
  size_t batch_size = gradient->dimension(0);
  size_t seq_len_plus_1 = gradient->dimension(1);
  size_t embed_dim = gradient->dimension(2);
  size_t seq_len = seq_len_plus_1 - 1;

  grad_input->ensure({batch_size, seq_len, embed_dim});

  DISPATCH_ON_3_DTYPES_TO_METHOD(backward_task, gradient, grad_input, class_token_gradients_,
                                 class_token_, batch_size, seq_len, embed_dim, "default");
}

uint64_t ClassTokenLayer::forward_flops(const std::vector<size_t> &input_shape) const { return 0; }

uint64_t ClassTokenLayer::backward_flops(const std::vector<size_t> &input_shape) const { return 0; }

LayerConfig ClassTokenLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.parameters["embed_dim"] = embed_dim_;
  return config;
}

std::unique_ptr<Layer> ClassTokenLayer::clone() const {
  return std::make_unique<ClassTokenLayer>(embed_dim_, this->name_);
}

std::vector<size_t>
ClassTokenLayer::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() < 3) {
    throw std::runtime_error("ClassTokenLayer: Input shape must have at least 3 dimensions");
  }
  size_t batch_size = input_shape[0];
  size_t seq_len = input_shape[1];
  size_t embed_dim = input_shape[2];
  return {batch_size, seq_len + 1, embed_dim};
}

void ClassTokenLayer::collect_parameters(std::vector<Tensor> &params) {
  params.push_back(class_token_);
}

void ClassTokenLayer::collect_gradients(std::vector<Tensor> &grads) {
  grads.push_back(class_token_gradients_);
}

std::unique_ptr<ClassTokenLayer> ClassTokenLayer::create_from_config(const LayerConfig &config) {
  size_t embed_dim = config.get<size_t>("embed_dim");
  return std::make_unique<ClassTokenLayer>(embed_dim, config.name);
}

} // namespace tnn
