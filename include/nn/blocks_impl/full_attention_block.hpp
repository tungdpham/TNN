/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/task.hpp"
#include "math/cpu/gemm.hpp"
#include "nn/blocks_impl/cpu/permute_heads.hpp"
#include "nn/blocks_impl/cpu/softmax.hpp"
#include "nn/mem_pool.hpp"
#include "ops/ops.hpp"
#ifdef USE_CUDA
#include "math/cuda/gemm.hpp"
#include "nn/blocks_impl/cuda/permute_heads.hpp"
#include "nn/blocks_impl/cuda/softmax.hpp"
#endif
#ifdef USE_CUDNN
#include "device/cuda/cuda_context.hpp"
#endif
#include "nn/layers_impl/dense_layer.hpp"
#include "nn/layers_impl/parameterized_layer.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tnn {

template <typename T = float> class FullAttentionBlock : public ParameterizedLayer<T> {
private:
  size_t embed_dim_;
  size_t num_heads_;
  size_t head_dim_;

  std::unique_ptr<DenseLayer<T>> q_proj_;
  std::unique_ptr<DenseLayer<T>> k_proj_;
  std::unique_ptr<DenseLayer<T>> v_proj_;
  std::unique_ptr<DenseLayer<T>> out_proj_;

  std::unordered_map<size_t, Tensor<T>> q_cache_;
  std::unordered_map<size_t, Tensor<T>> k_cache_;
  std::unordered_map<size_t, Tensor<T>> v_cache_;
  std::unordered_map<size_t, Tensor<T>> scores_cache_;

public:
  FullAttentionBlock(size_t embed_dim, size_t num_heads, const std::string &name = "attention")
      : ParameterizedLayer<T>(name), embed_dim_(embed_dim), num_heads_(num_heads) {

    if (embed_dim % num_heads != 0) {
      throw std::invalid_argument("embed_dim must be divisible by num_heads");
    }
    head_dim_ = embed_dim / num_heads;

    q_proj_ = std::make_unique<DenseLayer<T>>(embed_dim, embed_dim, true, name + "_q");
    k_proj_ = std::make_unique<DenseLayer<T>>(embed_dim, embed_dim, true, name + "_k");
    v_proj_ = std::make_unique<DenseLayer<T>>(embed_dim, embed_dim, true, name + "_v");
    out_proj_ = std::make_unique<DenseLayer<T>>(embed_dim, embed_dim, true, name + "_out");
  }

  void init_params() override {
    q_proj_->init();
    k_proj_->init();
    v_proj_->init();
    out_proj_->init();
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

  void forward_impl(const Tensor<T> &input, Tensor<T> &output, size_t micro_batch_id = 0) override {
    if (input.dims() != 3) {
      throw std::runtime_error(
          "FullAttentionBlock: Input tensor must have 3 dimensions (Batch, Seq, Embed)");
    }
    if (input.dimension(2) != embed_dim_) {
      throw std::runtime_error("FullAttentionBlock: Input embed_dim does not match layer config");
    }
    size_t batch_size = input.dimension(0);
    size_t L = input.dimension(1);

    Tensor<T> &q = q_cache_[micro_batch_id];
    Tensor<T> &k = k_cache_[micro_batch_id];
    Tensor<T> &v = v_cache_[micro_batch_id];

    q_proj_->forward(input, q, micro_batch_id);
    k_proj_->forward(input, k, micro_batch_id);
    v_proj_->forward(input, v, micro_batch_id);

    size_t batch_count = batch_size * num_heads_;
    size_t M = L;
    size_t N = L;
    size_t K_dim = head_dim_;

    Tensor<T> &scores = scores_cache_[micro_batch_id];
    scores.ensure({batch_count, 1, L, L}, this->device_);

    T alpha = 1.0f / std::sqrt(static_cast<T>(head_dim_));
    T beta = 0.0f;

    auto s_ptr = scores.data_ptr().get();

    size_t head_size = head_dim_ * L;
    size_t score_size = L * L;

    // Permute Q, K, V: (Batch, Seq, Heads, Dim) -> (Batch, Heads, Seq, Dim)
    PooledTensor<T> q_perm_buf = this->get_buffer(q.shape());
    Tensor<T> &q_perm = q_perm_buf.get();
    PooledTensor<T> k_perm_buf = this->get_buffer(k.shape());
    Tensor<T> &k_perm = k_perm_buf.get();
    PooledTensor<T> v_perm_buf = this->get_buffer(v.shape());
    Tensor<T> &v_perm = v_perm_buf.get();

    auto q_perm_ptr = q_perm.data_ptr().get();
    auto k_perm_ptr = k_perm.data_ptr().get();
    auto v_perm_ptr = v_perm.data_ptr().get();

    if (this->device_->device_type() == DeviceType::CPU) {
      create_cpu_task("default", cpu::permute_heads<T>, q.data_ptr().get(), q_perm_ptr, batch_size,
                      L, num_heads_, head_dim_);
      create_cpu_task("default", cpu::permute_heads<T>, k.data_ptr().get(), k_perm_ptr, batch_size,
                      L, num_heads_, head_dim_);
      create_cpu_task("default", cpu::permute_heads<T>, v.data_ptr().get(), v_perm_ptr, batch_size,
                      L, num_heads_, head_dim_);

      // Scores = Q * K^T
      create_cpu_task("default", cpu::gemm_strided_batched<T>, q_perm_ptr, k_perm_ptr, s_ptr, M, N,
                      K_dim, false, true, alpha, beta, batch_count, head_size, head_size,
                      score_size);
      create_cpu_task("default", cpu::softmax_forward<T>, s_ptr, s_ptr, batch_count * L, L);
    }
#ifdef USE_CUDA
    else if (this->device_->device_type() == DeviceType::GPU) {
      create_gpu_task("default", cuda::permute_heads<T>, q.data_ptr().get(), q_perm_ptr, batch_size,
                      L, num_heads_, head_dim_);
      create_gpu_task("default", cuda::permute_heads<T>, k.data_ptr().get(), k_perm_ptr, batch_size,
                      L, num_heads_, head_dim_);
      create_gpu_task("default", cuda::permute_heads<T>, v.data_ptr().get(), v_perm_ptr, batch_size,
                      L, num_heads_, head_dim_);

      create_gpu_task("default", cuda::gemm_strided_batched<T>, q_perm_ptr, k_perm_ptr, s_ptr, M, N,
                      K_dim, false, true, alpha, beta, batch_count, head_size, head_size,
                      score_size);

#ifdef USE_CUDNN
      auto cuda_context = dynamic_cast<CUDAContext *>(this->device_->context());
      if (!cuda_context) {
        throw std::runtime_error("Failed to get CUDA context");
      }
      create_gpu_task("default", cuda::softmax_forward<T>, cuda_context->getCudnnHandle(), s_ptr,
                      s_ptr, batch_count * L, L);
#else
      throw std::runtime_error("AttentionBlock: GPU Softmax requires cuDNN.");
#endif
    }
#endif

    PooledTensor<T> attn_out_perm_buf = this->get_buffer(q.shape());
    Tensor<T> &attn_out_perm = attn_out_perm_buf.get();
    auto out_perm_ptr = attn_out_perm.data_ptr().get();

    // AttnOut = Scores * V
    if (this->device_->device_type() == DeviceType::CPU) {
      create_cpu_task("default", cpu::gemm_strided_batched<T>, s_ptr, v_perm_ptr, out_perm_ptr, L,
                      head_dim_, L, false, false, 1.0f, 0.0f, batch_count, score_size, head_size,
                      head_size);
    }
#ifdef USE_CUDA
    else if (this->device_->device_type() == DeviceType::GPU) {
      create_gpu_task("default", cuda::gemm_strided_batched<T>, s_ptr, v_perm_ptr, out_perm_ptr, L,
                      head_dim_, L, false, false, 1.0f, 0.0f, batch_count, score_size, head_size,
                      head_size);
    }
#endif

    PooledTensor<T> attn_out_buffer = this->get_buffer({batch_size, L, embed_dim_});
    Tensor<T> &attn_out = attn_out_buffer.get();

    // Permute back: (Batch, Heads, Seq, Dim) -> (Batch, Seq, Heads, Dim)
    if (this->device_->device_type() == DeviceType::CPU) {
      create_cpu_task("default", cpu::permute_heads<T>, out_perm_ptr, attn_out.data_ptr().get(),
                      batch_size, num_heads_, L, head_dim_);
    }
#ifdef USE_CUDA
    else if (this->device_->device_type() == DeviceType::GPU) {
      create_gpu_task("default", cuda::permute_heads<T>, out_perm_ptr, attn_out.data_ptr().get(),
                      batch_size, num_heads_, L, head_dim_);
    }
#endif

    out_proj_->forward(attn_out, output, micro_batch_id);
  }

  void backward_impl(const Tensor<T> &gradient, Tensor<T> &grad_input,
                     size_t micro_batch_id = 0) override {
    if (q_cache_.find(micro_batch_id) == q_cache_.end()) {
      throw std::runtime_error("FullAttentionBlock: Cache not found for micro_batch_id");
    }

    Tensor<T> &q = q_cache_[micro_batch_id];
    Tensor<T> &k = k_cache_[micro_batch_id];
    Tensor<T> &v = v_cache_[micro_batch_id];
    Tensor<T> &scores = scores_cache_[micro_batch_id];

    grad_input.ensure(q.shape(), this->device_);

    PooledTensor<T> grad_attn_out_buffer = this->get_buffer(q.shape());
    Tensor<T> &grad_attn_out = grad_attn_out_buffer.get();
    out_proj_->backward(gradient, grad_attn_out, micro_batch_id);

    const auto &q_shape = q.shape();
    size_t batch_size = q_shape[0];
    size_t L = q_shape[1];

    // Re-permute Q, K, V for backward
    PooledTensor<T> q_perm_buf = this->get_buffer(q.shape());
    Tensor<T> &q_perm = q_perm_buf.get();
    PooledTensor<T> k_perm_buf = this->get_buffer(k.shape());
    Tensor<T> &k_perm = k_perm_buf.get();
    PooledTensor<T> v_perm_buf = this->get_buffer(v.shape());
    Tensor<T> &v_perm = v_perm_buf.get();

    // Permute grad_attn_out
    PooledTensor<T> grad_attn_out_perm_buf = this->get_buffer(q.shape());
    Tensor<T> &grad_attn_out_perm = grad_attn_out_perm_buf.get();

    if (this->device_->device_type() == DeviceType::CPU) {
      create_cpu_task("default", cpu::permute_heads<T>, q.data_ptr().get(), q_perm.data_ptr().get(),
                      batch_size, L, num_heads_, head_dim_);
      create_cpu_task("default", cpu::permute_heads<T>, k.data_ptr().get(), k_perm.data_ptr().get(),
                      batch_size, L, num_heads_, head_dim_);
      create_cpu_task("default", cpu::permute_heads<T>, v.data_ptr().get(), v_perm.data_ptr().get(),
                      batch_size, L, num_heads_, head_dim_);
      create_cpu_task("default", cpu::permute_heads<T>, grad_attn_out.data_ptr().get(),
                      grad_attn_out_perm.data_ptr().get(), batch_size, L, num_heads_, head_dim_);
    }
#ifdef USE_CUDA
    else if (this->device_->device_type() == DeviceType::GPU) {
      create_gpu_task("default", cuda::permute_heads<T>, q.data_ptr().get(),
                      q_perm.data_ptr().get(), batch_size, L, num_heads_, head_dim_);
      create_gpu_task("default", cuda::permute_heads<T>, k.data_ptr().get(),
                      k_perm.data_ptr().get(), batch_size, L, num_heads_, head_dim_);
      create_gpu_task("default", cuda::permute_heads<T>, v.data_ptr().get(),
                      v_perm.data_ptr().get(), batch_size, L, num_heads_, head_dim_);
      create_gpu_task("default", cuda::permute_heads<T>, grad_attn_out.data_ptr().get(),
                      grad_attn_out_perm.data_ptr().get(), batch_size, L, num_heads_, head_dim_);
    }
#endif

    size_t batch_count = batch_size * num_heads_;
    size_t head_size = head_dim_ * L;
    size_t score_size = L * L;

    // Buffers for permuted gradients
    PooledTensor<T> grad_q_perm_buf = this->get_buffer(q.shape());
    Tensor<T> &grad_q_perm = grad_q_perm_buf.get();
    PooledTensor<T> grad_k_perm_buf = this->get_buffer(k.shape());
    Tensor<T> &grad_k_perm = grad_k_perm_buf.get();
    PooledTensor<T> grad_v_perm_buf = this->get_buffer(v.shape());
    Tensor<T> &grad_v_perm = grad_v_perm_buf.get();
    PooledTensor<T> grad_scores_buffer = this->get_buffer(scores.shape());
    Tensor<T> &grad_scores = grad_scores_buffer.get();

    auto g_v_ptr = grad_v_perm.data_ptr().get();
    auto g_out_ptr = grad_attn_out_perm.data_ptr().get();
    auto s_ptr = scores.data_ptr().get();
    auto g_s_ptr = grad_scores.data_ptr().get();
    auto v_ptr = v_perm.data_ptr().get();
    auto q_ptr = q_perm.data_ptr().get();
    auto k_ptr = k_perm.data_ptr().get();
    auto g_q_ptr = grad_q_perm.data_ptr().get();
    auto g_k_ptr = grad_k_perm.data_ptr().get();

    // dV = S^T * dOut
    // dScores = dOut * V^T
    if (this->device_->device_type() == DeviceType::CPU) {
      create_cpu_task("default", cpu::gemm_strided_batched<T>, s_ptr, g_out_ptr, g_v_ptr, L,
                      head_dim_, L, true, false, 1.0f, 0.0f, batch_count, score_size, head_size,
                      head_size);
      create_cpu_task("default", cpu::gemm_strided_batched<T>, g_out_ptr, v_ptr, g_s_ptr, L, L,
                      head_dim_, false, true, 1.0f, 0.0f, batch_count, head_size, head_size,
                      score_size);
    }
#ifdef USE_CUDA
    else if (this->device_->device_type() == DeviceType::GPU) {
      create_gpu_task("default", cuda::gemm_strided_batched<T>, s_ptr, g_out_ptr, g_v_ptr, L,
                      head_dim_, L, true, false, 1.0f, 0.0f, batch_count, score_size, head_size,
                      head_size);
      create_gpu_task("default", cuda::gemm_strided_batched<T>, g_out_ptr, v_ptr, g_s_ptr, L, L,
                      head_dim_, false, true, 1.0f, 0.0f, batch_count, head_size, head_size,
                      score_size);
    }
#endif

    if (this->device_->device_type() == DeviceType::CPU) {
      create_cpu_task("default", cpu::softmax_backward<T>, s_ptr, g_s_ptr, g_s_ptr, batch_count * L,
                      L);
    }
#ifdef USE_CUDA
    else if (this->device_->device_type() == DeviceType::GPU) {
#ifdef USE_CUDNN
      auto cuda_context = dynamic_cast<CUDAContext *>(this->device_->context());
      if (!cuda_context) {
        throw std::runtime_error("Failed to get CUDA context");
      }
      create_gpu_task("default", cuda::softmax_backward<T>, cuda_context->getCudnnHandle(), s_ptr,
                      g_s_ptr, g_s_ptr, batch_count * L, L);
#else
      throw std::runtime_error("AttentionBlock: GPU Softmax requires cuDNN.");
#endif
    }
#endif

    T alpha = 1.0f / std::sqrt(static_cast<T>(head_dim_));

    // dK = dScores^T * Q
    // dQ = dScores * K
    if (this->device_->device_type() == DeviceType::CPU) {
      create_cpu_task("default", cpu::gemm_strided_batched<T>, g_s_ptr, q_ptr, g_k_ptr, L,
                      head_dim_, L, true, false, alpha, 0.0f, batch_count, score_size, head_size,
                      head_size);
      create_cpu_task("default", cpu::gemm_strided_batched<T>, g_s_ptr, k_ptr, g_q_ptr, L,
                      head_dim_, L, false, false, alpha, 0.0f, batch_count, score_size, head_size,
                      head_size);
    }
#ifdef USE_CUDA
    else if (this->device_->device_type() == DeviceType::GPU) {
      create_gpu_task("default", cuda::gemm_strided_batched<T>, g_s_ptr, q_ptr, g_k_ptr, L,
                      head_dim_, L, true, false, alpha, 0.0f, batch_count, score_size, head_size,
                      head_size);
      create_gpu_task("default", cuda::gemm_strided_batched<T>, g_s_ptr, k_ptr, g_q_ptr, L,
                      head_dim_, L, false, false, alpha, 0.0f, batch_count, score_size, head_size,
                      head_size);
    }
#endif

    PooledTensor<T> grad_q_buffer = this->get_buffer(q.shape());
    Tensor<T> &grad_q = grad_q_buffer.get();
    PooledTensor<T> grad_k_buffer = this->get_buffer(k.shape());
    Tensor<T> &grad_k = grad_k_buffer.get();
    PooledTensor<T> grad_v_buffer = this->get_buffer(v.shape());
    Tensor<T> &grad_v = grad_v_buffer.get();

    // Permute grads back: (B, H, L, D) -> (B, L, H, D)
    if (this->device_->device_type() == DeviceType::CPU) {
      create_cpu_task("default", cpu::permute_heads<T>, g_q_ptr, grad_q.data_ptr().get(),
                      batch_size, num_heads_, L, head_dim_);
      create_cpu_task("default", cpu::permute_heads<T>, g_k_ptr, grad_k.data_ptr().get(),
                      batch_size, num_heads_, L, head_dim_);
      create_cpu_task("default", cpu::permute_heads<T>, g_v_ptr, grad_v.data_ptr().get(),
                      batch_size, num_heads_, L, head_dim_);
    }
#ifdef USE_CUDA
    else if (this->device_->device_type() == DeviceType::GPU) {
      create_gpu_task("default", cuda::permute_heads<T>, g_q_ptr, grad_q.data_ptr().get(),
                      batch_size, num_heads_, L, head_dim_);
      create_gpu_task("default", cuda::permute_heads<T>, g_k_ptr, grad_k.data_ptr().get(),
                      batch_size, num_heads_, L, head_dim_);
      create_gpu_task("default", cuda::permute_heads<T>, g_v_ptr, grad_v.data_ptr().get(),
                      batch_size, num_heads_, L, head_dim_);
    }
#endif

    PooledTensor<T> grad_input_q_buffer = this->get_buffer(q.shape());
    Tensor<T> &grad_input_q = grad_input_q_buffer.get();
    PooledTensor<T> grad_input_k_buffer = this->get_buffer(q.shape());
    Tensor<T> &grad_input_k = grad_input_k_buffer.get();
    PooledTensor<T> grad_input_v_buffer = this->get_buffer(q.shape());
    Tensor<T> &grad_input_v = grad_input_v_buffer.get();

    q_proj_->backward(grad_q, grad_input_q, micro_batch_id);
    k_proj_->backward(grad_k, grad_input_k, micro_batch_id);
    v_proj_->backward(grad_v, grad_input_v, micro_batch_id);

    ops::add(grad_input_q.data_ptr(), grad_input_k.data_ptr(), grad_input.data_ptr(), q.size());
    ops::add(grad_input.data_ptr(), grad_input_v.data_ptr(), grad_input.data_ptr(), q.size());
  }

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override {
    size_t batch_size = input_shape[0];
    size_t L = input_shape[1];
    uint64_t flops = q_proj_->forward_flops(input_shape) + k_proj_->forward_flops(input_shape) +
                     v_proj_->forward_flops(input_shape);
    size_t score_ops = batch_size * num_heads_ * L * L * head_dim_ * 2;
    size_t attn_ops = batch_size * num_heads_ * L * L * head_dim_ * 2;
    flops += score_ops + attn_ops;
    flops += out_proj_->forward_flops(input_shape);
    return flops;
  }

  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override {
    return forward_flops(input_shape) * 2;
  }

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
