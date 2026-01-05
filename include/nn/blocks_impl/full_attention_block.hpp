/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/task.hpp"
#include "math/cpu/gemm.hpp"
#include "nn/mem_pool.hpp"
#include "ops/ops.hpp"
#ifdef USE_CUDA
#include "math/cuda/gemm.hpp"
#endif
#ifdef USE_CUDNN
#include "device/cuda/cuda_context.hpp"
#include "nn/blocks_impl/cuda/cudnn_attention_ops.hpp"
#endif
#include "nn/layers_impl/conv2d_layer.hpp"
#include "nn/layers_impl/parameterized_layer.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace tnn {

template <typename T = float> class FullAttentionBlock : public ParameterizedLayer<T> {
private:
  size_t embed_dim_;
  size_t num_heads_;
  size_t head_dim_;

  std::unique_ptr<Conv2DLayer<T>> q_proj_;
  std::unique_ptr<Conv2DLayer<T>> k_proj_;
  std::unique_ptr<Conv2DLayer<T>> v_proj_;
  std::unique_ptr<Conv2DLayer<T>> out_proj_;

  Tensor<T> q_, k_, v_;
  Tensor<T> scores_;

  void softmax_last_dim(Tensor<T> &input) {
    if (input.is_on_gpu()) {
#ifdef USE_CUDNN
      auto cuda_context = dynamic_cast<CUDAContext *>(this->device_->context());
      if (!cuda_context) {
        throw std::runtime_error("Failed to get CUDA context");
      }
      size_t total_rows = input.batch_size() * input.height();
      size_t cols = input.width();

      auto &input_ptr = input.data_ptr();

      create_gpu_task("default", cuda::cudnn_attn::softmax_forward<T>,
                      cuda_context->getCudnnHandle(), input_ptr.get(), input_ptr.get(), total_rows,
                      cols);
#else
      throw std::runtime_error("AttentionBlock: GPU Softmax requires cuDNN.");
#endif
    } else {
      T *data = input.data_ptr().get();
      size_t total_rows = input.batch_size() * input.height();
      size_t cols = input.width();

      for (size_t i = 0; i < total_rows; ++i) {
        T *row = data + i * cols;
        T max_val = row[0];
        for (size_t j = 1; j < cols; ++j) {
          if (row[j] > max_val)
            max_val = row[j];
        }

        T sum = 0;
        for (size_t j = 0; j < cols; ++j) {
          row[j] = std::exp(row[j] - max_val);
          sum += row[j];
        }

        T inv_sum = 1.0f / std::max(sum, static_cast<T>(1e-8));
        for (size_t j = 0; j < cols; ++j) {
          row[j] *= inv_sum;
        }
      }
    }
  }

public:
  FullAttentionBlock(size_t embed_dim, size_t num_heads, const std::string &name = "attention")
      : ParameterizedLayer<T>(name), embed_dim_(embed_dim), num_heads_(num_heads) {

    if (embed_dim % num_heads != 0) {
      throw std::invalid_argument("embed_dim must be divisible by num_heads");
    }
    head_dim_ = embed_dim / num_heads;

    q_proj_ =
        std::make_unique<Conv2DLayer<T>>(embed_dim, embed_dim, 1, 1, 1, 1, 0, 0, true, name + "_q");
    k_proj_ =
        std::make_unique<Conv2DLayer<T>>(embed_dim, embed_dim, 1, 1, 1, 1, 0, 0, true, name + "_k");
    v_proj_ =
        std::make_unique<Conv2DLayer<T>>(embed_dim, embed_dim, 1, 1, 1, 1, 0, 0, true, name + "_v");
    out_proj_ = std::make_unique<Conv2DLayer<T>>(embed_dim, embed_dim, 1, 1, 1, 1, 0, 0, true,
                                                 name + "_out");
  }

  void initialize_params() override {
    q_proj_->initialize();
    k_proj_->initialize();
    v_proj_->initialize();
    out_proj_->initialize();
  }

  void set_training(bool training) override {
    this->is_training_ = training;
    q_proj_->set_training(training);
    k_proj_->set_training(training);
    v_proj_->set_training(training);
    out_proj_->set_training(training);
  }

  void set_device(const Device *device) override {
    ParameterizedLayer<T>::set_device(device);
    q_proj_->set_device(device);
    k_proj_->set_device(device);
    v_proj_->set_device(device);
    out_proj_->set_device(device);
  }

  void forward(const Tensor<T> &input, Tensor<T> &output, size_t micro_batch_id = 0) override {
    size_t batch_size = input.batch_size();
    size_t H = input.height();
    size_t W = input.width();
    size_t L = H * W;

    q_proj_->forward(input, q_, micro_batch_id);
    k_proj_->forward(input, k_, micro_batch_id);
    v_proj_->forward(input, v_, micro_batch_id);

    size_t batch_count = batch_size * num_heads_;
    size_t M = L;
    size_t N = L;
    size_t K_dim = head_dim_;

    scores_.ensure({batch_count, 1, L, L}, this->device_);
    Tensor<T> &scores = scores_;

    T alpha = 1.0f / std::sqrt(static_cast<T>(head_dim_));
    T beta = 0.0f;

    auto q_ptr = q_.data_ptr().get();
    auto k_ptr = k_.data_ptr().get();
    auto v_ptr = v_.data_ptr().get();
    auto s_ptr = scores.data_ptr().get();

    size_t head_size = head_dim_ * L;
    size_t score_size = L * L;

    for (size_t i = 0; i < batch_count; ++i) {
      T *curr_q = q_ptr + i * head_size;
      T *curr_k = k_ptr + i * head_size;
      T *curr_s = s_ptr + i * score_size;

      if (this->device_->device_type() == DeviceType::CPU) {
        create_cpu_task("default", cpu::gemm<T>, curr_q, curr_k, curr_s, M, N, K_dim, true, false,
                        alpha, beta);
      }
#ifdef USE_CUDA
      else if (this->device_->device_type() == DeviceType::GPU) {
        create_gpu_task("default", cuda::gemm<T>, curr_q, curr_k, curr_s, M, N, K_dim, true, false,
                        alpha, beta);
      }
#endif
    }

    softmax_last_dim(scores);

    PooledTensor<T> attn_out_buffer = this->get_buffer({batch_size, embed_dim_, H, W});
    Tensor<T> &attn_out = attn_out_buffer.get();
    auto out_ptr = attn_out.data_ptr().get();

    for (size_t i = 0; i < batch_count; ++i) {
      T *curr_v = v_ptr + i * head_size;
      T *curr_s = s_ptr + i * score_size;
      T *curr_out = out_ptr + i * head_size;

      if (this->device_->device_type() == DeviceType::CPU) {
        create_cpu_task("default", cpu::gemm<T>, curr_v, curr_s, curr_out, head_dim_, L, L, false,
                        true, 1.0f, 0.0f);
      }
#ifdef USE_CUDA
      else if (this->device_->device_type() == DeviceType::GPU) {
        create_gpu_task("default", cuda::gemm<T>, curr_v, curr_s, curr_out, head_dim_, L, L, false,
                        true, 1.0f, 0.0f);
      }
#endif
    }

    out_proj_->forward(attn_out, output, micro_batch_id);
  }

  void backward(const Tensor<T> &gradient, Tensor<T> &grad_input,
                size_t micro_batch_id = 0) override {
    PooledTensor<T> grad_attn_out_buffer = this->get_buffer(q_.shape());
    Tensor<T> &grad_attn_out = grad_attn_out_buffer.get();
    out_proj_->backward(gradient, grad_attn_out, micro_batch_id);

    size_t batch_size = q_.batch_size();
    size_t L = q_.height() * q_.width();
    size_t batch_count = batch_size * num_heads_;
    size_t head_size = head_dim_ * L;
    size_t score_size = L * L;

    PooledTensor<T> grad_q_buffer = this->get_buffer(q_.shape());
    Tensor<T> &grad_q = grad_q_buffer.get();
    PooledTensor<T> grad_k_buffer = this->get_buffer(k_.shape());
    Tensor<T> &grad_k = grad_k_buffer.get();
    PooledTensor<T> grad_v_buffer = this->get_buffer(v_.shape());
    Tensor<T> &grad_v = grad_v_buffer.get();
    PooledTensor<T> grad_scores_buffer = this->get_buffer(scores_.shape());
    Tensor<T> &grad_scores = grad_scores_buffer.get();

    auto g_v_ptr = grad_v.data_ptr().get();
    auto g_out_ptr = grad_attn_out.data_ptr().get();
    auto s_ptr = scores_.data_ptr().get();
    auto g_s_ptr = grad_scores.data_ptr().get();
    auto v_ptr = v_.data_ptr().get();
    auto q_ptr = q_.data_ptr().get();
    auto k_ptr = k_.data_ptr().get();
    auto g_q_ptr = grad_q.data_ptr().get();
    auto g_k_ptr = grad_k.data_ptr().get();

    for (size_t i = 0; i < batch_count; ++i) {
      T *curr_gv = g_v_ptr + i * head_size;
      T *curr_gout = g_out_ptr + i * head_size;
      T *curr_s = s_ptr + i * score_size;
      T *curr_gs = g_s_ptr + i * score_size;
      T *curr_v = v_ptr + i * head_size;

      if (this->device_->device_type() == DeviceType::CPU) {
        cpu::gemm<T>(curr_gout, curr_s, curr_gv, head_dim_, L, L, false, false, 1.0f, 0.0f);
        cpu::gemm<T>(curr_gout, curr_v, curr_gs, L, L, head_dim_, true, false, 1.0f, 0.0f);
      }
#ifdef USE_CUDA
      else if (this->device_->device_type() == DeviceType::GPU) {
        create_gpu_task("default", cuda::gemm<T>, curr_gout, curr_s, curr_gv, head_dim_, L, L,
                        false, false, 1.0f, 0.0f);
        create_gpu_task("default", cuda::gemm<T>, curr_gout, curr_v, curr_gs, L, L, head_dim_, true,
                        false, 1.0f, 0.0f);
      }
#endif
    }

    if (this->device_->device_type() == DeviceType::GPU) {
#ifdef USE_CUDNN
      auto cuda_context = dynamic_cast<CUDAContext *>(this->device_->context());
      if (!cuda_context) {
        throw std::runtime_error("Failed to get CUDA context");
      }
      create_gpu_task("default", cuda::cudnn_attn::softmax_backward<T>,
                      cuda_context->getCudnnHandle(), s_ptr, g_s_ptr, g_s_ptr, batch_count * L, L);
#else
      throw std::runtime_error("AttentionBlock: GPU Softmax requires cuDNN.");
#endif
    } else {
      softmax_backward_cpu(scores_, grad_scores, grad_scores);
    }

    T alpha = 1.0f / std::sqrt(static_cast<T>(head_dim_));

    for (size_t i = 0; i < batch_count; ++i) {
      T *curr_gs = g_s_ptr + i * score_size;
      T *curr_q = q_ptr + i * head_size;
      T *curr_k = k_ptr + i * head_size;
      T *curr_gq = g_q_ptr + i * head_size;
      T *curr_gk = g_k_ptr + i * head_size;

      if (this->device_->device_type() == DeviceType::CPU) {
        cpu::gemm<T>(curr_k, curr_gs, curr_gq, head_dim_, L, L, false, true, alpha, 0.0f);
        cpu::gemm<T>(curr_q, curr_gs, curr_gk, head_dim_, L, L, false, false, alpha, 0.0f);
      }
#ifdef USE_CUDA
      else if (this->device_->device_type() == DeviceType::GPU) {
        create_gpu_task("default", cuda::gemm<T>, curr_k, curr_gs, curr_gq, head_dim_, L, L, false,
                        true, alpha, 0.0f);
        create_gpu_task("default", cuda::gemm<T>, curr_q, curr_gs, curr_gk, head_dim_, L, L, false,
                        false, alpha, 0.0f);
      }
#endif
    }

    PooledTensor<T> grad_input_q_buffer = this->get_buffer(grad_input.shape());
    Tensor<T> &grad_input_q = grad_input_q_buffer.get();
    PooledTensor<T> grad_input_k_buffer = this->get_buffer(grad_input.shape());
    Tensor<T> &grad_input_k = grad_input_k_buffer.get();
    PooledTensor<T> grad_input_v_buffer = this->get_buffer(grad_input.shape());
    Tensor<T> &grad_input_v = grad_input_v_buffer.get();

    q_proj_->backward(grad_q, grad_input_q, micro_batch_id);
    k_proj_->backward(grad_k, grad_input_k, micro_batch_id);
    v_proj_->backward(grad_v, grad_input_v, micro_batch_id);

    ops::add(grad_input_q.data_ptr(), grad_input_k.data_ptr(), grad_input.data_ptr(),
             grad_input.size());
    ops::add(grad_input.data_ptr(), grad_input_v.data_ptr(), grad_input.data_ptr(),
             grad_input.size());
  }

  void softmax_backward_cpu(const Tensor<T> &output, const Tensor<T> &grad_output,
                            Tensor<T> &grad_input) {
    const T *y_data = output.data_ptr().get();
    const T *dy_data = grad_output.data_ptr().get();
    T *dx_data = grad_input.data_ptr().get();

    size_t rows = output.batch_size() * output.height();
    size_t cols = output.width();

    for (size_t i = 0; i < rows; ++i) {
      const T *y_row = y_data + i * cols;
      const T *dy_row = dy_data + i * cols;
      T *dx_row = dx_data + i * cols;

      T dot = 0;
      for (size_t j = 0; j < cols; ++j) {
        dot += y_row[j] * dy_row[j];
      }

      for (size_t j = 0; j < cols; ++j) {
        dx_row[j] = y_row[j] * (dy_row[j] - dot);
      }
    }
  }

  uint64_t forward_complexity(const std::vector<size_t> &input_shape) const override { return 0; }
  uint64_t backward_complexity(const std::vector<size_t> &input_shape) const override { return 0; }

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override { return 0; }
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override { return 0; }

  std::string type() const override { return "full_attention"; }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = this->name_;
    config.parameters["embed_dim"] = embed_dim_;
    config.parameters["num_heads"] = num_heads_;
    return config;
  }

  std::unique_ptr<Layer<T>> clone() const override {
    return std::make_unique<FullAttentionBlock<T>>(embed_dim_, num_heads_, this->name_);
  }

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override {
    return input_shape;
  }

protected:
  void collect_parameters(std::vector<Tensor<T> *> &params) override {
    auto q_params = q_proj_->parameters();
    params.insert(params.end(), q_params.begin(), q_params.end());
    auto k_params = k_proj_->parameters();
    params.insert(params.end(), k_params.begin(), k_params.end());
    auto v_params = v_proj_->parameters();
    params.insert(params.end(), v_params.begin(), v_params.end());
    auto out_params = out_proj_->parameters();
    params.insert(params.end(), out_params.begin(), out_params.end());
  }

  void collect_gradients(std::vector<Tensor<T> *> &grads) override {
    auto q_grads = q_proj_->gradients();
    grads.insert(grads.end(), q_grads.begin(), q_grads.end());
    auto k_grads = k_proj_->gradients();
    grads.insert(grads.end(), k_grads.begin(), k_grads.end());
    auto v_grads = v_proj_->gradients();
    grads.insert(grads.end(), v_grads.begin(), v_grads.end());
    auto out_grads = out_proj_->gradients();
    grads.insert(grads.end(), out_grads.begin(), out_grads.end());
  }
};

} // namespace tnn
