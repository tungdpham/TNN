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
#include "nn/blocks_impl/cpu/permute_heads.hpp"
#include "nn/blocks_impl/cpu/softmax.hpp"
#include "nn/mem_pool.hpp"
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

  std::unique_ptr<DenseLayer<T>> q_proj_;
  std::unique_ptr<DenseLayer<T>> k_proj_;
  std::unique_ptr<DenseLayer<T>> v_proj_;
  std::unique_ptr<DenseLayer<T>> out_proj_;

  std::unordered_map<size_t, Tensor<T>> q_cache_;
  std::unordered_map<size_t, Tensor<T>> k_cache_;
  std::unordered_map<size_t, Tensor<T>> v_cache_;

public:
  FlashAttentionBlock(size_t embed_dim, size_t num_heads,
                      const std::string &name = "flash_attention")
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

  void forward_impl(const Tensor<T> &input, Tensor<T> &output, size_t micro_batch_id = 0) override {
    const auto &shape = input.shape();
    // Assuming (Batch, Seq, Embed) or similar
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    Tensor<T> &q = q_cache_[micro_batch_id];
    Tensor<T> &k = k_cache_[micro_batch_id];
    Tensor<T> &v = v_cache_[micro_batch_id];

    q_proj_->forward(input, q, micro_batch_id);
    k_proj_->forward(input, k, micro_batch_id);
    v_proj_->forward(input, v, micro_batch_id);

    // Permute Q, K, V from (B, L, H*D) -> (B, H, L, D)
    PooledTensor<T> q_permuted = this->get_buffer({batch_size, num_heads_, seq_len, head_dim_});
    PooledTensor<T> k_permuted = this->get_buffer({batch_size, num_heads_, seq_len, head_dim_});
    PooledTensor<T> v_permuted = this->get_buffer({batch_size, num_heads_, seq_len, head_dim_});

    auto q_raw = q.data_ptr().get();
    auto qp_raw = q_permuted.get().data_ptr().get();
    auto k_raw = k.data_ptr().get();
    auto kp_raw = k_permuted.get().data_ptr().get();
    auto v_raw = v.data_ptr().get();
    auto vp_raw = v_permuted.get().data_ptr().get();

    if (this->device_->device_type() == DeviceType::CPU) {
      create_cpu_task("default", cpu::permute_heads<T>, q_raw, qp_raw, batch_size, seq_len,
                      num_heads_, head_dim_);
      create_cpu_task("default", cpu::permute_heads<T>, k_raw, kp_raw, batch_size, seq_len,
                      num_heads_, head_dim_);
      create_cpu_task("default", cpu::permute_heads<T>, v_raw, vp_raw, batch_size, seq_len,
                      num_heads_, head_dim_);
    }
#ifdef USE_CUDA
    else {
      create_gpu_task("default", cuda::permute_heads<T>, q_raw, qp_raw, batch_size, seq_len,
                      num_heads_, head_dim_);
      create_gpu_task("default", cuda::permute_heads<T>, k_raw, kp_raw, batch_size, seq_len,
                      num_heads_, head_dim_);
      create_gpu_task("default", cuda::permute_heads<T>, v_raw, vp_raw, batch_size, seq_len,
                      num_heads_, head_dim_);
    }
#endif

    PooledTensor<T> attn_out_permuted =
        this->get_buffer({batch_size, num_heads_, seq_len, head_dim_});
    size_t batch_count = batch_size * num_heads_;
    size_t L = seq_len;

    auto out_perm_ptr = attn_out_permuted.get().data_ptr().get();

    if (this->device_->device_type() == DeviceType::CPU) {
      create_cpu_task("default", cpu::flash_attention_forward<T>, qp_raw, kp_raw, vp_raw,
                      out_perm_ptr, batch_count, head_dim_, L);
    }
#ifdef USE_CUDA
    else if (this->device_->device_type() == DeviceType::GPU) {
      // GPU Fallback (Standard Attn)
      size_t M = L;
      size_t N = L;
      size_t K_dim = head_dim_;

      PooledTensor<T> scores_buffer = this->get_buffer({batch_count, 1, L, L});
      Tensor<T> &scores = scores_buffer.get();

      T alpha = 1.0f / std::sqrt(static_cast<T>(head_dim_));
      T beta = 0.0f;
      auto s_ptr = scores.data_ptr().get();

      size_t head_size = head_dim_ * L;
      size_t score_size = L * L;

      // Q(L, D) * K(L, D)^T -> S(L, L)
      create_gpu_task("default", cuda::gemm_strided_batched<T>, qp_raw, kp_raw, s_ptr, M, N, K_dim,
                      false, true, alpha, beta, batch_count, head_size, head_size, score_size);

#ifdef USE_CUDNN
      auto cuda_context = dynamic_cast<CUDAContext *>(this->device_->context());
      create_gpu_task("default", cuda::softmax_forward<T>, cuda_context->getCudnnHandle(), s_ptr,
                      s_ptr, batch_count * L, L);
#endif

      // S(L, L) * V(L, D) -> Out(L, D)
      // gemm(A, B, C, m, n, k) -> C = A*B
      // m=L, n=D, k=L
      create_gpu_task("default", cuda::gemm_strided_batched<T>, s_ptr, vp_raw, out_perm_ptr, L,
                      head_dim_, L, false, false, 1.0f, 0.0f, batch_count, score_size, head_size,
                      head_size);
    }
#endif

    // Permute Output back
    PooledTensor<T> attn_out_buffer = this->get_buffer({batch_size, 1, seq_len, embed_dim_});
    Tensor<T> &attn_out = attn_out_buffer.get();
    auto att_out_ptr = attn_out.data_ptr().get();

    if (this->device_->device_type() == DeviceType::CPU) {
      create_cpu_task("default", cpu::permute_heads<T>, out_perm_ptr, att_out_ptr, batch_size,
                      num_heads_, seq_len, head_dim_);
    }
#ifdef USE_CUDA
    else {
      create_gpu_task("default", cuda::permute_heads<T>, out_perm_ptr, att_out_ptr, batch_size,
                      num_heads_, seq_len, head_dim_);
    }
#endif

    out_proj_->forward(attn_out, output, micro_batch_id);
  }

  void backward_impl(const Tensor<T> &gradient, Tensor<T> &grad_input,
                     size_t micro_batch_id = 0) override {
    if (q_cache_.find(micro_batch_id) == q_cache_.end()) {
      throw std::runtime_error("FlashAttentionBlock: Cache not found for micro_batch_id");
    }
    Tensor<T> &q = q_cache_[micro_batch_id];
    Tensor<T> &k = k_cache_[micro_batch_id];
    Tensor<T> &v = v_cache_[micro_batch_id];

    Tensor<T> d_attn_out;
    out_proj_->backward(gradient, d_attn_out, micro_batch_id);

    const auto &q_shape = q.shape();
    size_t batch_size = q_shape[0];
    size_t seq_len = q_shape[2];
    size_t batch_count = batch_size * num_heads_;

    PooledTensor<T> dq_buffer = this->get_buffer(q.shape());
    Tensor<T> &dq = dq_buffer.get();
    PooledTensor<T> dk_buffer = this->get_buffer(k.shape());
    Tensor<T> &dk = dk_buffer.get();
    PooledTensor<T> dv_buffer = this->get_buffer(v.shape());
    Tensor<T> &dv = dv_buffer.get();

    // Permute Q, K, V
    PooledTensor<T> q_permuted = this->get_buffer({batch_size, num_heads_, seq_len, head_dim_});
    PooledTensor<T> k_permuted = this->get_buffer({batch_size, num_heads_, seq_len, head_dim_});
    PooledTensor<T> v_permuted = this->get_buffer({batch_size, num_heads_, seq_len, head_dim_});
    PooledTensor<T> d_attn_out_permuted =
        this->get_buffer({batch_size, num_heads_, seq_len, head_dim_});

    T *q_raw = q.data_ptr().get();
    T *k_raw = k.data_ptr().get();
    T *v_raw = v.data_ptr().get();
    T *qp_raw = q_permuted.get().data_ptr().get();
    T *kp_raw = k_permuted.get().data_ptr().get();
    T *vp_raw = v_permuted.get().data_ptr().get();
    T *d_out_raw = d_attn_out.data_ptr().get();
    T *d_out_p_raw = d_attn_out_permuted.get().data_ptr().get();

    if (this->device_->device_type() == DeviceType::CPU) {
      create_cpu_task("default", cpu::permute_heads<T>, q_raw, qp_raw, batch_size, seq_len,
                      num_heads_, head_dim_);
      create_cpu_task("default", cpu::permute_heads<T>, k_raw, kp_raw, batch_size, seq_len,
                      num_heads_, head_dim_);
      create_cpu_task("default", cpu::permute_heads<T>, v_raw, vp_raw, batch_size, seq_len,
                      num_heads_, head_dim_);
      create_cpu_task("default", cpu::permute_heads<T>, d_out_raw, d_out_p_raw, batch_size, seq_len,
                      num_heads_, head_dim_);
    }
#ifdef USE_CUDA
    else {
      create_gpu_task("default", cuda::permute_heads<T>, q_raw, qp_raw, batch_size, seq_len,
                      num_heads_, head_dim_);
      create_gpu_task("default", cuda::permute_heads<T>, k_raw, kp_raw, batch_size, seq_len,
                      num_heads_, head_dim_);
      create_gpu_task("default", cuda::permute_heads<T>, v_raw, vp_raw, batch_size, seq_len,
                      num_heads_, head_dim_);
      create_gpu_task("default", cuda::permute_heads<T>, d_out_raw, d_out_p_raw, batch_size,
                      seq_len, num_heads_, head_dim_);
    }
#endif

    PooledTensor<T> dq_permuted = this->get_buffer({batch_size, num_heads_, seq_len, head_dim_});
    PooledTensor<T> dk_permuted = this->get_buffer({batch_size, num_heads_, seq_len, head_dim_});
    PooledTensor<T> dv_permuted = this->get_buffer({batch_size, num_heads_, seq_len, head_dim_});
    T *dqp_raw = dq_permuted.get().data_ptr().get();
    T *dkp_raw = dk_permuted.get().data_ptr().get();
    T *dvp_raw = dv_permuted.get().data_ptr().get();

    if (this->device_->device_type() == DeviceType::CPU) {
      size_t L = seq_len;
      size_t head_size = head_dim_ * L;
      size_t score_size = L * L;

      // Need temporary buffers
      PooledTensor<T> s_buf = this->get_buffer({batch_count, 1, L, L});
      PooledTensor<T> ds_buf = this->get_buffer({batch_count, 1, L, L});
      PooledTensor<T> da_buf = this->get_buffer({batch_count, 1, L, L});

      T *s_ptr = s_buf.get().data_ptr().get();
      T *ds_ptr = ds_buf.get().data_ptr().get();
      T *da_ptr = da_buf.get().data_ptr().get();

      T alpha = 1.0f / std::sqrt(static_cast<T>(head_dim_));

      // 1. Recompute S = Q * K^T
      create_cpu_task("default", cpu::gemm_strided_batched<T>, qp_raw, kp_raw, s_ptr, L, L,
                      head_dim_, true, false, alpha, 0.0f, batch_count, head_size, head_size,
                      score_size);

      // 2. Softmax(S)
      create_cpu_task("default", cpu::softmax_forward<T>, s_ptr, s_ptr, batch_count * L, L);

      // 3. dV = S^T * dOut -> Using the parameter order from FullAttentionBlock
      // dO(L,D), S(L,L). dV(L,D).
      // FullAttn uses m=head_dim, n=L, k=L. Computes (D,L).
      // cpu_attention_backward used D, L, L.
      create_cpu_task("default", cpu::gemm_strided_batched<T>, d_out_p_raw, s_ptr, dvp_raw,
                      head_dim_, L, L, false, false, 1.0f, 0.0f, batch_count, head_size, score_size,
                      head_size);

      // 4. dS_out = dOut * V^T
      create_cpu_task("default", cpu::gemm_strided_batched<T>, d_out_p_raw, vp_raw, ds_ptr, L, L,
                      head_dim_, true, false, 1.0f, 0.0f, batch_count, head_size, head_size,
                      score_size);

      // 5. dS_in = SoftmaxBackward(dS_out, S)
      create_cpu_task("default", cpu::softmax_backward<T>, s_ptr, ds_ptr, da_ptr, batch_count * L,
                      L);

      // 6. dQ = dS_in * K
      create_cpu_task("default", cpu::gemm_strided_batched<T>, kp_raw, da_ptr, dqp_raw, head_dim_,
                      L, L, false, true, 1.0f, 0.0f, batch_count, head_size, score_size, head_size);

      // 7. dK = dS_in^T * Q
      create_cpu_task("default", cpu::gemm_strided_batched<T>, qp_raw, da_ptr, dkp_raw, head_dim_,
                      L, L, false, false, 1.0f, 0.0f, batch_count, head_size, score_size,
                      head_size);
    }
#ifdef USE_CUDA
    else if (this->device_->device_type() == DeviceType::GPU) {
      size_t L = seq_len;
      PooledTensor<T> scores_buffer = this->get_buffer({batch_count, 1, L, L});
      T *s_ptr = scores_buffer.get().data_ptr().get();

      size_t head_size = head_dim_ * L;
      size_t score_size = L * L;
      T alpha = 1.0f / std::sqrt(static_cast<T>(head_dim_));

      // 1. Recompute S
      create_gpu_task("default", cuda::gemm_strided_batched<T>, qp_raw, kp_raw, s_ptr, L, L,
                      head_dim_, false, true, alpha, 0.0f, batch_count, head_size, head_size,
                      score_size);

#ifdef USE_CUDNN
      auto cuda_context = dynamic_cast<CUDAContext *>(this->device_->context());
      create_gpu_task("default", cuda::softmax_forward<T>, cuda_context->getCudnnHandle(), s_ptr,
                      s_ptr, batch_count * L, L);
#endif

      // 2. dV = S^T * dOut
      create_gpu_task("default", cuda::gemm_strided_batched<T>, s_ptr, d_out_p_raw, dvp_raw, L,
                      head_dim_, L, true, false, 1.0f, 0.0f, batch_count, score_size, head_size,
                      head_size);

      // 3. dS = dOut * V^T
      PooledTensor<T> ds_buffer = this->get_buffer({batch_count, 1, L, L});
      T *ds_ptr = ds_buffer.get().data_ptr().get();
      create_gpu_task("default", cuda::gemm_strided_batched<T>, d_out_p_raw, vp_raw, ds_ptr, L, L,
                      head_dim_, false, true, 1.0f, 0.0f, batch_count, head_size, head_size,
                      score_size);

      // 4. Backward Softmax
      PooledTensor<T> da_buffer = this->get_buffer({batch_count, 1, L, L});
      T *da_ptr = da_buffer.get().data_ptr().get();

#ifdef USE_CUDNN
      create_gpu_task("default", cuda::softmax_backward<T>, cuda_context->getCudnnHandle(), s_ptr,
                      ds_ptr, da_ptr, batch_count * L, L);
#endif
      // 5. dQ = dA * K
      create_gpu_task("default", cuda::gemm_strided_batched<T>, da_ptr, kp_raw, dqp_raw, L,
                      head_dim_, L, false, false, alpha, 0.0f, batch_count, score_size, head_size,
                      head_size);

      // 6. dK = dA^T * Q
      create_gpu_task("default", cuda::gemm_strided_batched<T>, da_ptr, qp_raw, dkp_raw, L,
                      head_dim_, L, true, false, alpha, 0.0f, batch_count, score_size, head_size,
                      head_size);
    }
#endif

    T *dq_ptr = dq.data_ptr().get();
    T *dk_ptr = dk.data_ptr().get();
    T *dv_ptr = dv.data_ptr().get();

    if (this->device_->device_type() == DeviceType::CPU) {
      create_cpu_task("default", cpu::permute_heads<T>, dqp_raw, dq_ptr, batch_size, seq_len,
                      num_heads_, head_dim_);
      create_cpu_task("default", cpu::permute_heads<T>, dkp_raw, dk_ptr, batch_size, seq_len,
                      num_heads_, head_dim_);
      create_cpu_task("default", cpu::permute_heads<T>, dvp_raw, dv_ptr, batch_size, seq_len,
                      num_heads_, head_dim_);
    }
#ifdef USE_CUDA
    else {
      create_gpu_task("default", cuda::permute_heads<T>, dqp_raw, dq_ptr, batch_size, seq_len,
                      num_heads_, head_dim_);
      create_gpu_task("default", cuda::permute_heads<T>, dkp_raw, dk_ptr, batch_size, seq_len,
                      num_heads_, head_dim_);
      create_gpu_task("default", cuda::permute_heads<T>, dvp_raw, dv_ptr, batch_size, seq_len,
                      num_heads_, head_dim_);
    }
#endif

    Tensor<T> dq_in, dk_in, dv_in;
    q_proj_->backward(dq, dq_in, micro_batch_id);
    k_proj_->backward(dk, dk_in, micro_batch_id);
    v_proj_->backward(dv, dv_in, micro_batch_id);

    // Accumulate gradients
    grad_input.ensure(dq_in.shape());
    size_t size = dq_in.size();

    PooledTensor<T> temp_buffer = this->get_buffer(dq_in.shape());

    auto &dq_in_ptr = dq_in.data_ptr();
    auto &dk_in_ptr = dk_in.data_ptr();
    auto &dv_in_ptr = dv_in.data_ptr();
    auto &temp_ptr = temp_buffer.get().data_ptr();
    auto &grad_in_ptr = grad_input.data_ptr();

    ops::add(dq_in_ptr, dk_in_ptr, temp_ptr, size, "default");
    ops::add(temp_ptr, dv_in_ptr, grad_in_ptr, size, "default");
  }

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
