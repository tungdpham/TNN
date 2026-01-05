/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/task.hpp"
#include "nn/layers_impl/cpu/class_token_ops.hpp"
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/class_token_ops.hpp"
#endif
#include "nn/layers_impl/parameterized_layer.hpp"
#include "ops/ops.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace tnn {

template <typename T = float> class ClassTokenLayer : public ParameterizedLayer<T> {
private:
  size_t embed_dim_;
  Tensor<T> class_token_;
  Tensor<T> class_token_gradients_;

public:
  ClassTokenLayer(size_t embed_dim, const std::string &name = "class_token")
      : ParameterizedLayer<T>(name), embed_dim_(embed_dim) {}

  void initialize_params() override {
    class_token_ = Tensor<T>({1, embed_dim_, 1, 1}, this->device_);
    class_token_gradients_ = Tensor<T>({1, embed_dim_, 1, 1}, this->device_);

    T fan_in = static_cast<T>(embed_dim_);
    T bound = static_cast<T>(1.0) / std::sqrt(fan_in);

    if (this->use_seed_) {
      class_token_.fill_random_uniform(-bound, bound, this->srand_seed_);
    } else {
      class_token_.fill_random_uniform(-bound, bound);
    }
    class_token_gradients_.fill(T(0));
  }

  void forward(const Tensor<T> &input, Tensor<T> &output, size_t micro_batch_id = 0) override {
    if (!this->initialized_) {
      throw std::runtime_error(
          "Layer parameters not initialized. Call initialize() before forward.");
    }

    size_t N = input.batch_size();
    size_t C = input.channels();
    size_t H = input.height();
    size_t W = input.width();
    size_t L = H * W;

    if (C != embed_dim_) {
      throw std::runtime_error("ClassTokenLayer: Input channels must match embed_dim");
    }

    // Output shape: [N, C, L + 1, 1]
    output.ensure({N, C, L + 1, 1}, this->device_);

    auto &out_ptr = output.data_ptr();
    const auto &in_ptr = input.data_ptr();
    auto &token_ptr = class_token_.data_ptr();

    if (this->device_->device_type() == DeviceType::CPU) {
      create_cpu_task("default", cpu::class_token_forward<T>, in_ptr.get(), token_ptr.get(),
                      out_ptr.get(), N, C, L);
    } else {
#ifdef USE_CUDA
      // Use create_gpu_task to ensure correct flow/stream usage, consistent with other layers
      create_gpu_task("default", cuda::class_token_forward<T>, in_ptr.get(), token_ptr.get(),
                      out_ptr.get(), N, C, L);
#else
      throw std::runtime_error("CUDA support for ClassTokenLayer not implemented");
#endif
    }
  }

  void backward(const Tensor<T> &gradient, Tensor<T> &grad_input,
                size_t micro_batch_id = 0) override {
    size_t N = gradient.batch_size();
    size_t C = gradient.channels();
    size_t L_plus_1 = gradient.height();
    size_t L = L_plus_1 - 1;

    grad_input.ensure({N, C, L, 1}, this->device_);

    const auto &grad_ptr = gradient.data_ptr();
    auto &grad_in_ptr = grad_input.data_ptr();
    auto &token_grad_ptr = class_token_gradients_.data_ptr();

    if (this->device_->device_type() == DeviceType::CPU) {
      create_cpu_task("default", cpu::class_token_backward<T>, grad_ptr.get(), grad_in_ptr.get(),
                      token_grad_ptr.get(), N, C, L);
    } else {
#ifdef USE_CUDA
      create_gpu_task("default", cuda::class_token_backward<T>, grad_ptr.get(), grad_in_ptr.get(),
                      token_grad_ptr.get(), N, C, L);
#else
      throw std::runtime_error("CUDA support for ClassTokenLayer not implemented");
#endif
    }
  }

  uint64_t forward_complexity(const std::vector<size_t> &input_shape) const override { return 0; }
  uint64_t backward_complexity(const std::vector<size_t> &input_shape) const override { return 0; }
  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override { return 0; }
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override { return 0; }

  std::string type() const override { return "class_token"; }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = this->name_;
    config.parameters["embed_dim"] = embed_dim_;
    return config;
  }

  std::unique_ptr<Layer<T>> clone() const override {
    return std::make_unique<ClassTokenLayer<T>>(embed_dim_, this->name_);
  }

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override {
    // input_shape: [N, C, H, W]
    // output_shape: [N, C, H*W + 1, 1]
    size_t N = input_shape[0];
    size_t C = input_shape[1];
    size_t H = input_shape[2];
    size_t W = input_shape[3];
    return {N, C, H * W + 1, 1};
  }

protected:
  void collect_parameters(std::vector<Tensor<T> *> &params) override {
    params.push_back(&class_token_);
  }

  void collect_gradients(std::vector<Tensor<T> *> &grads) override {
    grads.push_back(&class_token_gradients_);
  }
};

} // namespace tnn
