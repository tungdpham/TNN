/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/task.hpp"
#include "math/cpu/gemm.hpp"
#include "nn/blocks_impl/cpu/causal_mask.hpp"
#include "nn/blocks_impl/cpu/softmax.hpp"
#include "nn/mem_pool.hpp"
#ifdef USE_CUDA
#include "math/cuda/gemm.hpp"
#include "nn/blocks_impl/cuda/causal_mask.hpp"
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

template <typename T = float> class AttentionBlock : public ParameterizedLayer<T> {
private:
  size_t embed_dim_;
  size_t num_heads_;
  size_t head_dim_;
  bool is_causal_;

  std::unique_ptr<DenseLayer<T>> q_proj_;
  std::unique_ptr<DenseLayer<T>> k_proj_;
  std::unique_ptr<DenseLayer<T>> v_proj_;
  std::unique_ptr<DenseLayer<T>> out_proj_;

  std::unordered_map<size_t, Tensor<T>> q_cache_;
  std::unordered_map<size_t, Tensor<T>> k_cache_;
  std::unordered_map<size_t, Tensor<T>> v_cache_;

public:
  AttentionBlock(size_t embed_dim, size_t num_heads, bool is_causal = true,
                 const std::string &name = "attention_block")
      : ParameterizedLayer<T>(name), embed_dim_(embed_dim), num_heads_(num_heads),
        is_causal_(is_causal) {

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

  void set_device(const Device *device) override {
    ParameterizedLayer<T>::set_device(device);
    q_proj_->set_device(device);
    k_proj_->set_device(device);
    v_proj_->set_device(device);
    out_proj_->set_device(device);
  }

  void forward_impl(const Tensor<T> &input, Tensor<T> &output, size_t micro_batch_id = 0) override {
    const auto &input_shape = input.shape();
    // assuming (Batch, Seq, Embed) or similar
    size_t batch_size = input_shape[0];
    size_t seq_len = input_shape[1];

    Tensor<T> &q = q_cache_[micro_batch_id];
    Tensor<T> &k = k_cache_[micro_batch_id];
    Tensor<T> &v = v_cache_[micro_batch_id];

    q_proj_->forward(input, q, micro_batch_id);
    k_proj_->forward(input, k, micro_batch_id);
    v_proj_->forward(input, v, micro_batch_id);

    PooledTensor<T> attn_out_buffer = this->get_buffer(input_shape);
    Tensor<T> &attn_out = attn_out_buffer.get();
    auto att_out_ptr = attn_out.data_ptr().get();

    auto q_raw = q.data_ptr().get();
    auto k_raw = k.data_ptr().get();
    auto v_raw = v.data_ptr().get();

    if (this->device_->device_type() == DeviceType::CPU) {
      size_t L = seq_len;
      size_t batch_count = batch_size * num_heads_;
      T alpha = 1.0f / std::sqrt(static_cast<T>(head_dim_));

      // temp buffer for scores S: (B*H, L, L)
      PooledTensor<T> scores_buffer = this->get_buffer({batch_count, L, L});
      T *s_ptr = scores_buffer.get().data_ptr().get();

      // Q * K^T -> S
      create_cpu_task("default", cpu::gemm_strided_batched_ex<T>, q_raw, k_raw, s_ptr, L, L,
                      head_dim_, false, true, alpha, static_cast<T>(0.0), batch_count, head_dim_,
                      head_dim_, L * L, embed_dim_, embed_dim_, L);

      if (is_causal_) {
        create_cpu_task("default", cpu::apply_causal_mask<T>, s_ptr, batch_count, L,
                        static_cast<T>(-INFINITY));
      }

      // Softmax(S)
      create_cpu_task("default", cpu::softmax_forward<T>, s_ptr, s_ptr, batch_count * L, L);

      // S * V -> Out
      create_cpu_task("default", cpu::gemm_strided_batched_ex<T>, s_ptr, v_raw, att_out_ptr, L,
                      head_dim_, L, false, false, static_cast<T>(1.0), static_cast<T>(0.0),
                      batch_count, L * L, head_dim_, head_dim_, L, embed_dim_, embed_dim_);
    }
#ifdef USE_CUDA
    else if (this->device_->device_type() == DeviceType::GPU) {
      size_t L = seq_len;
      size_t batch_count = batch_size * num_heads_;

      PooledTensor<T> scores_buffer = this->get_buffer({batch_count, L, L});
      Tensor<T> &scores = scores_buffer.get();
      auto s_ptr = scores.data_ptr().get();

      T alpha = 1.0f / std::sqrt(static_cast<T>(head_dim_));
      T beta = 0.0f;

      // Q * K^T -> S
      // input Layout: (B, L, H, D) -> Strided interpretation as (B, H, L, D)
      size_t lda_q = num_heads_ * head_dim_;
      size_t ldb_k = num_heads_ * head_dim_;
      size_t ldc_s = L;

      size_t stride_q = head_dim_;
      size_t stride_k = head_dim_;
      size_t stride_s = L * L;

      size_t total_gemm_batches = batch_size * num_heads_;

      create_gpu_task("default", cuda::gemm_strided_batched_ex<T>, q_raw, k_raw, s_ptr, L, L,
                      head_dim_, false, true, alpha, beta, total_gemm_batches, stride_q, stride_k,
                      stride_s, lda_q, ldb_k, ldc_s);

      if (is_causal_) {
        create_gpu_task("default", cuda::apply_causal_mask<T>, s_ptr, batch_count, L,
                        static_cast<T>(-INFINITY));
      }

#ifdef USE_CUDNN
      auto cuda_context = dynamic_cast<CUDAContext *>(this->device_->context());
      create_gpu_task("default", cuda::softmax_forward<T>, cuda_context->getCudnnHandle(), s_ptr,
                      s_ptr, batch_count * L, L);
#else
      throw std::runtime_error("AttentionBlock requires CUDNN for Softmax on GPU");
#endif

      // S * V -> Out
      // Out Layout: (B, L, H, D) -> Strided write
      size_t lda_s2 = L;
      size_t ldb_v2 = num_heads_ * head_dim_;
      size_t ldc_o2 = num_heads_ * head_dim_;

      size_t stride_s2 = L * L;
      size_t stride_v2 = head_dim_;
      size_t stride_o2 = head_dim_;

      create_gpu_task("default", cuda::gemm_strided_batched_ex<T>, s_ptr, v_raw, att_out_ptr, L,
                      head_dim_, L, false, false, 1.0f, 0.0f, total_gemm_batches, stride_s2,
                      stride_v2, stride_o2, lda_s2, ldb_v2, ldc_o2);
    }
#endif

    out_proj_->forward(attn_out, output, micro_batch_id);
  }

  void backward_impl(const Tensor<T> &gradient, Tensor<T> &grad_input,
                     size_t micro_batch_id = 0) override {
    if (q_cache_.find(micro_batch_id) == q_cache_.end()) {
      throw std::runtime_error("AttentionBlock: Cache not found for micro_batch_id");
    }
    Tensor<T> &q = q_cache_[micro_batch_id];
    Tensor<T> &k = k_cache_[micro_batch_id];
    Tensor<T> &v = v_cache_[micro_batch_id];

    PooledTensor<T> d_attn_out_buffer = this->get_buffer(gradient.shape());
    Tensor<T> &d_attn_out = d_attn_out_buffer.get();
    out_proj_->backward(gradient, d_attn_out, micro_batch_id);

    const auto &q_shape = q.shape();
    size_t batch_size = q_shape[0];
    size_t seq_len = q_shape[1];

    PooledTensor<T> dq_buffer = this->get_buffer(q.shape());
    Tensor<T> &dq = dq_buffer.get();
    PooledTensor<T> dk_buffer = this->get_buffer(k.shape());
    Tensor<T> &dk = dk_buffer.get();
    PooledTensor<T> dv_buffer = this->get_buffer(v.shape());
    Tensor<T> &dv = dv_buffer.get();

    T *q_raw = q.data_ptr().get();
    T *k_raw = k.data_ptr().get();
    T *v_raw = v.data_ptr().get();
    T *d_out_raw = d_attn_out.data_ptr().get();

    T *dq_ptr = dq.data_ptr().get();
    T *dk_ptr = dk.data_ptr().get();
    T *dv_ptr = dv.data_ptr().get();

    if (this->device_->device_type() == DeviceType::CPU) {
      size_t L = seq_len;
      size_t batch_count = batch_size * num_heads_;
      T alpha = 1.0f / std::sqrt(static_cast<T>(head_dim_));

      // temp buffers for scores (B*H, L, L) - contiguous
      PooledTensor<T> s_buf = this->get_buffer({batch_count, L, L});
      PooledTensor<T> ds_buf = this->get_buffer({batch_count, L, L});
      PooledTensor<T> da_buf = this->get_buffer({batch_count, L, L});

      T *s_ptr = s_buf.get().data_ptr().get();
      T *ds_ptr = ds_buf.get().data_ptr().get();
      T *da_ptr = da_buf.get().data_ptr().get(); // dS_in

      // Recompute S = Q * K^T
      create_cpu_task("default", cpu::gemm_strided_batched_ex<T>, q_raw, k_raw, s_ptr, L, L,
                      head_dim_, false, true, alpha, static_cast<T>(0.0), batch_count, head_dim_,
                      head_dim_, L * L, embed_dim_, embed_dim_, L);

      if (is_causal_) {
        create_cpu_task("default", cpu::apply_causal_mask<T>, s_ptr, batch_count, L,
                        static_cast<T>(-INFINITY));
      }

      // softmax(S)
      create_cpu_task("default", cpu::softmax_forward<T>, s_ptr, s_ptr, batch_count * L, L);

      // dV = S^T * dOut
      create_cpu_task("default", cpu::gemm_strided_batched_ex<T>, s_ptr, d_out_raw, dv_ptr, L,
                      head_dim_, L, true, false, static_cast<T>(1.0), static_cast<T>(0.0),
                      batch_count, L * L, head_dim_, head_dim_, L, embed_dim_, embed_dim_);

      // dS_out = dOut * V^T
      create_cpu_task("default", cpu::gemm_strided_batched_ex<T>, d_out_raw, v_raw, ds_ptr, L, L,
                      head_dim_, false, true, static_cast<T>(1.0), static_cast<T>(0.0), batch_count,
                      head_dim_, head_dim_, L * L, embed_dim_, embed_dim_, L);

      // dS_in = SoftmaxBackward(dS_out, S)
      create_cpu_task("default", cpu::softmax_backward<T>, s_ptr, ds_ptr, da_ptr, batch_count * L,
                      L);

      if (is_causal_) {
        create_cpu_task("default", cpu::apply_causal_mask<T>, da_ptr, batch_count, L,
                        static_cast<T>(0.0));
      }

      // dQ = dS_in * K
      create_cpu_task("default", cpu::gemm_strided_batched_ex<T>, da_ptr, k_raw, dq_ptr, L,
                      head_dim_, L, false, false, alpha, static_cast<T>(0.0), batch_count, L * L,
                      head_dim_, head_dim_, L, embed_dim_, embed_dim_);

      // dK = dS_in^T * Q
      create_cpu_task("default", cpu::gemm_strided_batched_ex<T>, da_ptr, q_raw, dk_ptr, L,
                      head_dim_, L, true, false, alpha, static_cast<T>(0.0), batch_count, L * L,
                      head_dim_, head_dim_, L, embed_dim_, embed_dim_);
    }
#ifdef USE_CUDA
    else if (this->device_->device_type() == DeviceType::GPU) {
      size_t L = seq_len;
      size_t batch_count = batch_size * num_heads_;
      T alpha = 1.0f / std::sqrt(static_cast<T>(head_dim_));

#ifdef USE_CUDNN
      auto cuda_context = dynamic_cast<CUDAContext *>(this->device_->context());
#endif

      // recompute S (Q * K^T)
      PooledTensor<T> scores_buffer = this->get_buffer({batch_count, L, L});
      T *s_ptr = scores_buffer.get().data_ptr().get();

      create_gpu_task("default", cuda::gemm_strided_batched_ex<T>, q_raw, k_raw, s_ptr, L, L,
                      head_dim_, false, true, alpha, 0.0f, batch_count, head_dim_, head_dim_, L * L,
                      embed_dim_, embed_dim_, L);

      if (is_causal_) {
        create_gpu_task("default", cuda::apply_causal_mask<T>, s_ptr, batch_count, L,
                        static_cast<T>(-INFINITY));
      }

#ifdef USE_CUDNN
      create_gpu_task("default", cuda::softmax_forward<T>, cuda_context->getCudnnHandle(), s_ptr,
                      s_ptr, batch_count * L, L);
#else
      throw std::runtime_error("AttentionBlock requires CUDNN for Softmax on GPU");
#endif

      // dV = S^T * dOut
      create_gpu_task("default", cuda::gemm_strided_batched_ex<T>, s_ptr, d_out_raw, dv_ptr, L,
                      head_dim_, L, true, false, 1.0f, 0.0f, batch_count, L * L, head_dim_,
                      head_dim_, L, embed_dim_, embed_dim_);

      // dS = dOut * V^T
      PooledTensor<T> ds_buffer = this->get_buffer({batch_count, L, L});
      T *ds_ptr = ds_buffer.get().data_ptr().get();
      create_gpu_task("default", cuda::gemm_strided_batched_ex<T>, d_out_raw, v_raw, ds_ptr, L, L,
                      head_dim_, false, true, 1.0f, 0.0f, batch_count, head_dim_, head_dim_, L * L,
                      embed_dim_, embed_dim_, L);

      // Backward Softmax
      PooledTensor<T> da_buffer = this->get_buffer({batch_count, L, L});
      T *da_ptr = da_buffer.get().data_ptr().get();
#ifdef USE_CUDNN
      create_gpu_task("default", cuda::softmax_backward<T>, cuda_context->getCudnnHandle(), s_ptr,
                      ds_ptr, da_ptr, batch_count * L, L);
#else
      throw std::runtime_error("AttentionBlock requires CUDNN for SoftmaxBackward on GPU");
#endif

      if (is_causal_) {
        create_gpu_task("default", cuda::apply_causal_mask<T>, da_ptr, batch_count, L,
                        static_cast<T>(0.0));
      }

      // dQ = dA * K
      create_gpu_task("default", cuda::gemm_strided_batched_ex<T>, da_ptr, k_raw, dq_ptr, L,
                      head_dim_, L, false, false, alpha, 0.0f, batch_count, L * L, head_dim_,
                      head_dim_, L, embed_dim_, embed_dim_);

      // dK = dA^T * Q
      create_gpu_task("default", cuda::gemm_strided_batched_ex<T>, da_ptr, q_raw, dk_ptr, L,
                      head_dim_, L, true, false, alpha, 0.0f, batch_count, L * L, head_dim_,
                      head_dim_, L, embed_dim_, embed_dim_);
    }
#endif

    PooledTensor<T> dq_in_buffer = this->get_buffer(q.shape());
    Tensor<T> &dq_in = dq_in_buffer.get();
    PooledTensor<T> dk_in_buffer = this->get_buffer(k.shape());
    Tensor<T> &dk_in = dk_in_buffer.get();
    PooledTensor<T> dv_in_buffer = this->get_buffer(v.shape());
    Tensor<T> &dv_in = dv_in_buffer.get();
    q_proj_->backward(dq, dq_in, micro_batch_id);
    k_proj_->backward(dk, dk_in, micro_batch_id);
    v_proj_->backward(dv, dv_in, micro_batch_id);

    grad_input.ensure(dq_in.shape(), this->device_);
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

  std::string type() const override { return "attention_block"; }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = this->name_;
    config.parameters["embed_dim"] = embed_dim_;
    config.parameters["num_heads"] = num_heads_;
    return config;
  }

  std::unique_ptr<Layer<T>> clone() const override {
    return std::make_unique<AttentionBlock<T>>(embed_dim_, num_heads_, is_causal_, this->name_);
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
