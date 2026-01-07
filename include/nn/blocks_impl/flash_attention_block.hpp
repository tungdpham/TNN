/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/task.hpp"
#include "math/cpu/gemm.hpp"
#include "nn/blocks_impl/cpu/flash_attention.hpp"
#include "nn/mem_pool.hpp"
#ifdef USE_CUDA
#include "math/cuda/gemm.hpp"
#endif
#ifdef USE_CUDNN
#include "device/cuda/cuda_context.hpp"
#include "nn/blocks_impl/cuda/cudnn_attention_ops.hpp"
#endif
#include "nn/layers_impl/conv2d_layer.hpp"
#include "nn/layers_impl/parameterized_layer.hpp"
#include "ops/ops.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tnn {

template <typename T = float> class FlashAttentionBlock : public ParameterizedLayer<T> {
private:
  size_t embed_dim_;
  size_t num_heads_;
  size_t head_dim_;

  std::unique_ptr<Conv2DLayer<T>> q_proj_;
  std::unique_ptr<Conv2DLayer<T>> k_proj_;
  std::unique_ptr<Conv2DLayer<T>> v_proj_;
  std::unique_ptr<Conv2DLayer<T>> out_proj_;

  std::unordered_map<size_t, Tensor<T>> q_cache_;
  std::unordered_map<size_t, Tensor<T>> k_cache_;
  std::unordered_map<size_t, Tensor<T>> v_cache_;

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
      // CPU implementation
      T *data = input.data_ptr().get();
      size_t total_rows =
          input.batch_size() * input.height(); // N * H (here N=batch*heads, H=seq_len)
      size_t cols = input.width();             // W (here seq_len)

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

  static void cpu_attention_backward(T *q, T *k, T *v, T *d_out, T *dq, T *dk, T *dv,
                                     size_t batch_count, size_t head_dim, size_t L) {
    size_t D = head_dim;
    std::vector<T> S(L * L);
    std::vector<T> dS(L * L);
    std::vector<T> dA(L * L);

    for (size_t b = 0; b < batch_count; ++b) {
      T *Q_b = q + b * D * L;
      T *K_b = k + b * D * L;
      T *V_b = v + b * D * L;
      T *dO_b = d_out + b * D * L;
      T *dQ_b = dq + b * D * L;
      T *dK_b = dk + b * D * L;
      T *dV_b = dv + b * D * L;

      // 1. Compute S = Softmax(Q^T * K / sqrt(D))
      T scale = 1.0f / std::sqrt(static_cast<T>(D));
      cpu::gemm(Q_b, K_b, S.data(), L, L, D, true, false, scale, 0.0f);

      for (size_t i = 0; i < L; ++i) {
        T *row = S.data() + i * L;
        T max_val = -INFINITY;
        for (size_t j = 0; j < L; ++j)
          max_val = std::max(max_val, row[j]);
        T sum = 0;
        for (size_t j = 0; j < L; ++j) {
          row[j] = std::exp(row[j] - max_val);
          sum += row[j];
        }
        T inv_sum = 1.0f / sum;
        for (size_t j = 0; j < L; ++j)
          row[j] *= inv_sum;
      }

      // 2. dS = V^T * dO
      cpu::gemm(V_b, dO_b, dS.data(), L, L, D, true, false, 1.0f, 0.0f);

      // 3. dV = dO * S
      cpu::gemm(dO_b, S.data(), dV_b, D, L, L, false, false, 1.0f, 0.0f);

      // 4. dA = S * (dS - rowsum(dS * S)) * scale
      for (size_t i = 0; i < L; ++i) {
        T dot = 0;
        for (size_t j = 0; j < L; ++j) {
          dot += dS[i * L + j] * S[i * L + j];
        }
        for (size_t j = 0; j < L; ++j) {
          dA[i * L + j] = S[i * L + j] * (dS[i * L + j] - dot) * scale;
        }
      }

      // 5. dQ = K * dA^T
      cpu::gemm(K_b, dA.data(), dQ_b, D, L, L, false, true, 1.0f, 0.0f);

      // 6. dK = Q * dA
      cpu::gemm(Q_b, dA.data(), dK_b, D, L, L, false, false, 1.0f, 0.0f);
    }
  }

#ifdef USE_CUDNN
  static void run_cudnn_softmax_backward(cudnnHandle_t handle, float alpha_val, const T *y,
                                         const T *dy, T *dx, size_t rows, size_t cols,
                                         cudaStream_t stream) {
    cudnnTensorDescriptor_t yDesc, dyDesc, dxDesc;
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnCreateTensorDescriptor(&dyDesc);
    cudnnCreateTensorDescriptor(&dxDesc);

    int n = rows;
    int c = cols;
    int h = 1;
    int w = 1;
    cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

    float alpha = alpha_val;
    float beta = 0.0f;

    cudnnSetStream(handle, stream);

    cudnnSoftmaxBackward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, yDesc,
                         y, dyDesc, dy, &beta, dxDesc, dx);

    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(dyDesc);
    cudnnDestroyTensorDescriptor(dxDesc);
  }
#endif

public:
  FlashAttentionBlock(size_t embed_dim, size_t num_heads,
                      const std::string &name = "flash_attention")
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

  void forward(const Tensor<T> &input, Tensor<T> &output, size_t micro_batch_id = 0) override {
    size_t batch_size = input.batch_size();
    size_t H = input.height();
    size_t W = input.width();
    size_t L = H * W;

    PooledTensor<T> q_buffer = this->get_buffer(q_proj_->compute_output_shape(input.shape()));
    Tensor<T> &q = q_buffer.get();
    PooledTensor<T> k_buffer = this->get_buffer(k_proj_->compute_output_shape(input.shape()));
    Tensor<T> &k = k_buffer.get();
    PooledTensor<T> v_buffer = this->get_buffer(v_proj_->compute_output_shape(input.shape()));
    Tensor<T> &v = v_buffer.get();

    q_proj_->forward(input, q, micro_batch_id);
    k_proj_->forward(input, k, micro_batch_id);
    v_proj_->forward(input, v, micro_batch_id);

    size_t batch_count = batch_size * num_heads_;

    // We need a temporary buffer for attention output before final projection
    PooledTensor<T> attn_out_buffer = this->get_buffer({batch_size, embed_dim_, H, W});
    Tensor<T> &attn_out = attn_out_buffer.get();

    auto &q_ptr = q.data_ptr();
    auto &k_ptr = k.data_ptr();
    auto &v_ptr = v.data_ptr();
    auto &attn_out_ptr = attn_out.data_ptr();

    if (this->device_->device_type() == DeviceType::CPU) {
      create_cpu_task("default", cpu::flash_attention_forward<T>, q_ptr.get(), k_ptr.get(),
                      v_ptr.get(), attn_out_ptr.get(), batch_count, head_dim_, L);
    }
#ifdef USE_CUDA
    else if (this->device_->device_type() == DeviceType::GPU) {
      // Fallback to standard attention using cuDNN softmax
      size_t M = L;
      size_t N = L;
      size_t K_dim = head_dim_;

      PooledTensor<T> scores_buffer = this->get_buffer({batch_count, 1, L, L});
      Tensor<T> &scores = scores_buffer.get();

      T alpha = 1.0f / std::sqrt(static_cast<T>(head_dim_));
      T beta = 0.0f;

      auto q_ptr = q.data_ptr().get();
      auto k_ptr = k.data_ptr().get();
      auto v_ptr = v.data_ptr().get();
      auto s_ptr = scores.data_ptr().get();
      auto out_ptr = attn_out.data_ptr().get();

      size_t head_size = head_dim_ * L;
      size_t score_size = L * L;

      for (size_t i = 0; i < batch_count; ++i) {
        T *curr_q = q_ptr + i * head_size;
        T *curr_k = k_ptr + i * head_size;
        T *curr_s = s_ptr + i * score_size;

        create_gpu_task("default", cuda::gemm<T>, curr_q, curr_k, curr_s, M, N, K_dim, true, false,
                        alpha, beta);
      }

      softmax_last_dim(scores);

      for (size_t i = 0; i < batch_count; ++i) {
        T *curr_v = v_ptr + i * head_size;
        T *curr_s = s_ptr + i * score_size;
        T *curr_out = out_ptr + i * head_size;

        create_gpu_task("default", cuda::gemm<T>, curr_v, curr_s, curr_out, head_dim_, L, L, false,
                        true, 1.0f, 0.0f);
      }
    }
#endif
    auto it_q_cache = q_cache_.find(micro_batch_id);
    auto it_k_cache = k_cache_.find(micro_batch_id);
    auto it_v_cache = v_cache_.find(micro_batch_id);
    if (q_cache_.find(micro_batch_id) == q_cache_.end()) {
      q_cache_[micro_batch_id] = q.clone();
    } else {
      it_q_cache->second.ensure(q.shape(), this->device_);
      ops::copy(q.data_ptr(), it_q_cache->second.data_ptr(), q.size());
    }

    if (k_cache_.find(micro_batch_id) == k_cache_.end()) {
      k_cache_[micro_batch_id] = k.clone();
    } else {
      it_k_cache->second.ensure(k.shape(), this->device_);
      ops::copy(k.data_ptr(), it_k_cache->second.data_ptr(), k.size());
    }

    if (v_cache_.find(micro_batch_id) == v_cache_.end()) {
      v_cache_[micro_batch_id] = v.clone();
    } else {
      it_v_cache->second.ensure(v.shape(), this->device_);
      ops::copy(v.data_ptr(), it_v_cache->second.data_ptr(), v.size());
    }

    out_proj_->forward(attn_out, output, micro_batch_id);
  }

  void backward(const Tensor<T> &gradient, Tensor<T> &grad_input,
                size_t micro_batch_id = 0) override {
    if (q_cache_.find(micro_batch_id) == q_cache_.end()) {
      throw std::runtime_error("FlashAttentionBlock: Cache not found for micro_batch_id");
    }
    Tensor<T> &q = q_cache_[micro_batch_id];
    Tensor<T> &k = k_cache_[micro_batch_id];
    Tensor<T> &v = v_cache_[micro_batch_id];

    Tensor<T> d_attn_out;
    out_proj_->backward(gradient, d_attn_out, micro_batch_id);

    size_t batch_size = q.batch_size();
    size_t H = q.height();
    size_t W = q.width();
    size_t L = H * W;
    size_t batch_count = batch_size * num_heads_;

    PooledTensor<T> dq_buffer = this->get_buffer(q.shape());
    Tensor<T> &dq = dq_buffer.get();
    PooledTensor<T> dk_buffer = this->get_buffer(k.shape());
    Tensor<T> &dk = dk_buffer.get();
    PooledTensor<T> dv_buffer = this->get_buffer(v.shape());
    Tensor<T> &dv = dv_buffer.get();

    auto &q_ptr = q.data_ptr();
    auto &k_ptr = k.data_ptr();
    auto &v_ptr = v.data_ptr();
    auto &dout_ptr = d_attn_out.data_ptr();
    auto &dq_ptr = dq.data_ptr();
    auto &dk_ptr = dk.data_ptr();
    auto &dv_ptr = dv.data_ptr();

    if (this->device_->device_type() == DeviceType::CPU) {
      create_cpu_task("default", cpu_attention_backward, q_ptr.get(), k_ptr.get(), v_ptr.get(),
                      dout_ptr.get(), dq_ptr.get(), dk_ptr.get(), dv_ptr.get(), batch_count,
                      head_dim_, L);
    }
#ifdef USE_CUDA
    else if (this->device_->device_type() == DeviceType::GPU) {
      size_t M = L;
      size_t N = L;
      size_t K_dim = head_dim_;
      size_t score_size = L * L;
      size_t head_size = head_dim_ * L;

      PooledTensor<T> s_buffer = this->get_buffer({batch_count, 1, L, L});
      Tensor<T> &s = s_buffer.get();
      PooledTensor<T> ds_buffer = this->get_buffer({batch_count, 1, L, L});
      Tensor<T> &ds = ds_buffer.get();
      PooledTensor<T> da_buffer = this->get_buffer({batch_count, 1, L, L});
      Tensor<T> &da = da_buffer.get();

      T alpha = 1.0f / std::sqrt(static_cast<T>(head_dim_));
      T beta = 0.0f;

      auto s_ptr = s.data_ptr().get();
      auto ds_ptr_raw = ds.data_ptr().get();

      for (size_t i = 0; i < batch_count; ++i) {
        T *curr_q = q_ptr.get() + i * head_size;
        T *curr_k = k_ptr.get() + i * head_size;
        T *curr_s = s_ptr + i * score_size;
        create_gpu_task("default", cuda::gemm<T>, curr_q, curr_k, curr_s, M, N, K_dim, true, false,
                        alpha, beta);
      }

#ifdef USE_CUDNN
      auto cuda_context = dynamic_cast<CUDAContext *>(this->device_->context());
      create_gpu_task("default", cuda::cudnn_attn::softmax_forward<T>,
                      cuda_context->getCudnnHandle(), s_ptr, s_ptr, batch_count * L, L);
#endif

      for (size_t i = 0; i < batch_count; ++i) {
        T *curr_v = v_ptr.get() + i * head_size;
        T *curr_do = dout_ptr.get() + i * head_size;
        T *curr_ds = ds_ptr_raw + i * score_size;
        create_gpu_task("default", cuda::gemm<T>, curr_v, curr_do, curr_ds, L, L, head_dim_, true,
                        false, 1.0f, 0.0f);
      }

      for (size_t i = 0; i < batch_count; ++i) {
        T *curr_do = dout_ptr.get() + i * head_size;
        T *curr_s = s_ptr + i * score_size;
        T *curr_dv = dv_ptr.get() + i * head_size;
        create_gpu_task("default", cuda::gemm<T>, curr_do, curr_s, curr_dv, head_dim_, L, L, false,
                        false, 1.0f, 0.0f);
      }

#ifdef USE_CUDNN
      create_gpu_task("default", run_cudnn_softmax_backward, cuda_context->getCudnnHandle(), alpha,
                      s_ptr, ds_ptr_raw, da.data_ptr().get(), batch_count * L, L);
#endif

      for (size_t i = 0; i < batch_count; ++i) {
        T *curr_k = k_ptr.get() + i * head_size;
        T *curr_da = da.data_ptr().get() + i * score_size;
        T *curr_dq = dq_ptr.get() + i * head_size;
        create_gpu_task("default", cuda::gemm<T>, curr_k, curr_da, curr_dq, head_dim_, L, L, false,
                        true, 1.0f, 0.0f);
      }

      for (size_t i = 0; i < batch_count; ++i) {
        T *curr_q = q_ptr.get() + i * head_size;
        T *curr_da = da.data_ptr().get() + i * score_size;
        T *curr_dk = dk_ptr.get() + i * head_size;
        create_gpu_task("default", cuda::gemm<T>, curr_q, curr_da, curr_dk, head_dim_, L, L, false,
                        false, 1.0f, 0.0f);
      }
    }
#endif

    PooledTensor<T> dq_input_buffer = this->get_buffer(q.shape());
    Tensor<T> &dq_input = dq_input_buffer.get();
    PooledTensor<T> dk_input_buffer = this->get_buffer(k.shape());
    Tensor<T> &dk_input = dk_input_buffer.get();
    PooledTensor<T> dv_input_buffer = this->get_buffer(v.shape());
    Tensor<T> &dv_input = dv_input_buffer.get();

    q_proj_->backward(dq, dq_input, micro_batch_id);
    k_proj_->backward(dk, dk_input, micro_batch_id);
    v_proj_->backward(dv, dv_input, micro_batch_id);

    // grad_input = dq_input + dk_input + dv_input
    grad_input.ensure(dq_input.shape());

    size_t size = dq_input.size();
    PooledTensor<T> temp_buffer = this->get_buffer(dq_input.shape());
    Tensor<T> &temp = temp_buffer.get();

    auto &dq_in_ptr = dq_input.data_ptr();
    auto &dk_in_ptr = dk_input.data_ptr();
    auto &dv_in_ptr = dv_input.data_ptr();
    auto &temp_ptr = temp.data_ptr();
    auto &grad_in_ptr = grad_input.data_ptr();

    ops::add(dq_in_ptr, dk_in_ptr, temp_ptr, size, "default");
    ops::add(temp_ptr, dv_in_ptr, grad_in_ptr, size, "default");
  }

  uint64_t forward_complexity(const std::vector<size_t> &input_shape) const override { return 0; }
  uint64_t backward_complexity(const std::vector<size_t> &input_shape) const override { return 0; }

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override { return 0; }
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override { return 0; }

  std::string type() const override { return "flash_attention"; }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = this->name_;
    config.parameters["embed_dim"] = embed_dim_;
    config.parameters["num_heads"] = num_heads_;
    return config;
  }

  std::unique_ptr<Layer<T>> clone() const override {
    return std::make_unique<FlashAttentionBlock<T>>(embed_dim_, num_heads_, this->name_);
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
