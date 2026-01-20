/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/blocks_impl/attention_block.hpp"
#include "math/cpu/gemm.hpp"
#include "nn/blocks_impl/cpu/causal_mask.hpp"
#include "nn/blocks_impl/cpu/softmax.hpp"
#ifdef USE_CUDA
#include "math/cuda/gemm.hpp"
#include "nn/blocks_impl/cuda/causal_mask.hpp"
#include "nn/blocks_impl/cuda/softmax.hpp"
#endif
#ifdef USE_CUDNN
#include "device/cuda/cuda_context.hpp"
#endif
#include "tensor/ops.hpp"
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace tnn {

// Constructor
AttentionBlock::AttentionBlock(size_t embed_dim, size_t num_heads, bool is_causal,
                               const std::string &name)
    : ParameterizedLayer(name), embed_dim_(embed_dim), num_heads_(num_heads),
      is_causal_(is_causal) {

  if (embed_dim % num_heads != 0) {
    throw std::invalid_argument("embed_dim must be divisible by num_heads");
  }
  head_dim_ = embed_dim / num_heads;

  q_proj_ = std::make_unique<DenseLayer>(embed_dim, embed_dim, true, name + "_q");
  k_proj_ = std::make_unique<DenseLayer>(embed_dim, embed_dim, true, name + "_k");
  v_proj_ = std::make_unique<DenseLayer>(embed_dim, embed_dim, true, name + "_v");
  out_proj_ = std::make_unique<DenseLayer>(embed_dim, embed_dim, true, name + "_out");
}

void AttentionBlock::init_params() {
  q_proj_->init();
  k_proj_->init();
  v_proj_->init();
  out_proj_->init();
}

void AttentionBlock::on_set_device(const Device &device) {
  q_proj_->set_device(device);
  k_proj_->set_device(device);
  v_proj_->set_device(device);
  out_proj_->set_device(device);
}

void AttentionBlock::forward_impl(const Tensor &input, Tensor &output, size_t micro_batch_id) {
  const auto &input_shape = input->shape();

  size_t batch_size = input_shape[0];
  size_t seq_len = input_shape[1];

  Tensor &q = q_cache_[micro_batch_id];
  Tensor &k = k_cache_[micro_batch_id];
  Tensor &v = v_cache_[micro_batch_id];

  q_proj_->forward(input, q, micro_batch_id);
  k_proj_->forward(input, k, micro_batch_id);
  v_proj_->forward(input, v, micro_batch_id);

  Tensor attn_out = this->get_buffer(input_shape);
  attn_out->ensure(input_shape, this->device_);

  DISPATCH_ON_3_DTYPES_TO_METHOD(compute_attention_forward, q, k, v, attn_out, batch_size, seq_len,
                                 "default");

  out_proj_->forward(attn_out, output, micro_batch_id);
}

void AttentionBlock::backward_impl(const Tensor &gradient, Tensor &grad_input,
                                   size_t micro_batch_id) {
  if (q_cache_.find(micro_batch_id) == q_cache_.end()) {
    throw std::runtime_error("AttentionBlock: Cache not found for micro_batch_id");
  }
  Tensor &q = q_cache_[micro_batch_id];
  Tensor &k = k_cache_[micro_batch_id];
  Tensor &v = v_cache_[micro_batch_id];

  Tensor d_attn_out = this->get_buffer(gradient->shape());
  out_proj_->backward(gradient, d_attn_out, micro_batch_id);

  const auto &q_shape = q->shape();
  size_t batch_size = q_shape[0];
  size_t seq_len = q_shape[1];

  Tensor dq = this->get_buffer(q->shape());
  Tensor dk = this->get_buffer(k->shape());
  Tensor dv = this->get_buffer(v->shape());

  DISPATCH_ON_3_DTYPES_TO_METHOD(compute_attention_backward, q, k, v, d_attn_out, dq, dk, dv,
                                 batch_size, seq_len, "default");

  Tensor dq_in = this->get_buffer(q->shape());
  Tensor dk_in = this->get_buffer(k->shape());
  Tensor dv_in = this->get_buffer(v->shape());
  q_proj_->backward(dq, dq_in, micro_batch_id);
  k_proj_->backward(dk, dk_in, micro_batch_id);
  v_proj_->backward(dv, dv_in, micro_batch_id);

  grad_input->ensure(dq_in->shape(), this->device_);
  size_t size = dq_in->size();

  Tensor temp = this->get_buffer(dq_in->shape());

  DISPATCH_ON_DTYPE_TO_METHOD(TensorOps::add, dq_in, dk_in, temp, size, "default");
  DISPATCH_ON_DTYPE_TO_METHOD(TensorOps::add, temp, dv_in, grad_input, size, "default");
}

uint64_t AttentionBlock::forward_flops(const std::vector<size_t> &input_shape) const { return 0; }

uint64_t AttentionBlock::backward_flops(const std::vector<size_t> &input_shape) const { return 0; }

std::string AttentionBlock::type() const { return "attention_block"; }

LayerConfig AttentionBlock::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["embed_dim"] = embed_dim_;
  config.parameters["num_heads"] = num_heads_;
  return config;
}

std::unique_ptr<Layer> AttentionBlock::clone() const {
  return std::make_unique<AttentionBlock>(embed_dim_, num_heads_, is_causal_, this->name_);
}

std::vector<size_t>
AttentionBlock::compute_output_shape(const std::vector<size_t> &input_shape) const {
  return input_shape;
}

void AttentionBlock::collect_parameters(std::vector<Tensor> &params) {
  auto q_params = q_proj_->parameters();
  params.insert(params.end(), q_params.begin(), q_params.end());
  auto k_params = k_proj_->parameters();
  params.insert(params.end(), k_params.begin(), k_params.end());
  auto v_params = v_proj_->parameters();
  params.insert(params.end(), v_params.begin(), v_params.end());
  auto out_params = out_proj_->parameters();
  params.insert(params.end(), out_params.begin(), out_params.end());
}

void AttentionBlock::collect_gradients(std::vector<Tensor> &grads) {
  auto q_grads = q_proj_->gradients();
  grads.insert(grads.end(), q_grads.begin(), q_grads.end());
  auto k_grads = k_proj_->gradients();
  grads.insert(grads.end(), k_grads.begin(), k_grads.end());
  auto v_grads = v_proj_->gradients();
  grads.insert(grads.end(), v_grads.begin(), v_grads.end());
  auto out_grads = out_proj_->gradients();
  grads.insert(grads.end(), out_grads.begin(), out_grads.end());
}

// Template method implementations

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> AttentionBlock::compute_attention_forward(const Tensor &q, const Tensor &k,
                                                                const Tensor &v, Tensor &output,
                                                                size_t batch_size, size_t seq_len,
                                                                const std::string &flow_id) {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "AttentionBlock mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (q->data_type() != dtype_of<IO_T>() || k->data_type() != dtype_of<IO_T>() ||
      v->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("AttentionBlock IO tensor dtype mismatch with dispatch IO_T");
  }

  if (this->device_->device_type() == DeviceType::CPU) {
    auto q_raw = q->data_as<Compute_T>();
    auto k_raw = k->data_as<Compute_T>();
    auto v_raw = v->data_as<Compute_T>();
    auto out_ptr = output->data_as<Compute_T>();

    size_t L = seq_len;
    size_t batch_count = batch_size * num_heads_;
    Compute_T alpha = static_cast<Compute_T>(1.0 / std::sqrt(static_cast<double>(head_dim_)));

    Tensor scores = this->get_buffer({batch_count, L, L});
    Compute_T *s_ptr = scores->data_as<Compute_T>();

    for (size_t b = 0; b < batch_size; ++b) {
      auto q_b = q_raw + b * (L * embed_dim_);
      auto k_b = k_raw + b * (L * embed_dim_);
      auto s_b = s_ptr + b * (num_heads_ * L * L);

      create_cpu_task(flow_id, cpu::gemm_strided_batched_ex<Compute_T>, q_b, k_b, s_b, L, L,
                      head_dim_, false, true, alpha, static_cast<Compute_T>(0.0), num_heads_,
                      head_dim_, head_dim_, L * L, embed_dim_, embed_dim_, L);
    }

    if (is_causal_) {
      create_cpu_task(flow_id, cpu::apply_causal_mask<Compute_T>, s_ptr, batch_count, L,
                      static_cast<Compute_T>(-INFINITY));
    }

    create_cpu_task(flow_id, cpu::softmax_forward<Compute_T>, s_ptr, s_ptr, batch_count * L, L);

    for (size_t b = 0; b < batch_size; ++b) {
      auto s_b = s_ptr + b * (num_heads_ * L * L);
      auto v_b = v_raw + b * (L * embed_dim_);
      auto out_b = out_ptr + b * (L * embed_dim_);

      create_cpu_task(flow_id, cpu::gemm_strided_batched_ex<Compute_T>, s_b, v_b, out_b, L,
                      head_dim_, L, false, false, static_cast<Compute_T>(1.0),
                      static_cast<Compute_T>(0.0), num_heads_, L * L, head_dim_, head_dim_, L,
                      embed_dim_, embed_dim_);
    }
    return nullptr;
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    auto q_raw = q->data_as<Compute_T>();
    auto k_raw = k->data_as<Compute_T>();
    auto v_raw = v->data_as<Compute_T>();
    auto out_ptr = output->data_as<Compute_T>();

    size_t L = seq_len;
    size_t batch_count = batch_size * num_heads_;

    Tensor scores = this->get_buffer({batch_count, L, L});
    auto s_ptr = scores->data_as<Compute_T>();

    Compute_T alpha = static_cast<Compute_T>(1.0 / std::sqrt(static_cast<double>(head_dim_)));
    Compute_T beta = static_cast<Compute_T>(0.0);

    size_t lda_q = num_heads_ * head_dim_;
    size_t ldb_k = num_heads_ * head_dim_;
    size_t ldc_s = L;

    size_t stride_q = head_dim_;
    size_t stride_k = head_dim_;
    size_t stride_s = L * L;

    for (size_t b = 0; b < batch_size; ++b) {
      auto q_ptr_b = q_raw + b * (seq_len * num_heads_ * head_dim_);
      auto k_ptr_b = k_raw + b * (seq_len * num_heads_ * head_dim_);
      auto s_ptr_b = s_ptr + b * (num_heads_ * L * L);

      create_gpu_task(flow_id, cuda::gemm_strided_batched_ex<Compute_T>, q_ptr_b, k_ptr_b, s_ptr_b,
                      L, L, head_dim_, false, true, alpha, beta, num_heads_, stride_q, stride_k,
                      stride_s, lda_q, ldb_k, ldc_s);
    }

    if (is_causal_) {
      create_gpu_task(flow_id, cuda::apply_causal_mask<Compute_T>, s_ptr, batch_count, L,
                      static_cast<Compute_T>(-INFINITY));
    }

#ifdef USE_CUDNN
    auto cuda_context = dynamic_cast<CUDAContext *>(this->device_->context());
    create_gpu_task(flow_id, cuda::softmax_forward<Compute_T>, cuda_context->getCudnnHandle(),
                    s_ptr, s_ptr, batch_count * L, L);
#else
    throw std::runtime_error("AttentionBlock requires CUDNN for Softmax on GPU");
#endif

    size_t lda_s2 = L;
    size_t ldb_v2 = num_heads_ * head_dim_;
    size_t ldc_o2 = num_heads_ * head_dim_;

    size_t stride_s2 = L * L;
    size_t stride_v2 = head_dim_;
    size_t stride_o2 = head_dim_;

    for (size_t b = 0; b < batch_size; ++b) {
      auto s_ptr_b = s_ptr + b * (num_heads_ * L * L);
      auto v_ptr_b = v_raw + b * (seq_len * num_heads_ * head_dim_);
      auto out_ptr_b = out_ptr + b * (seq_len * num_heads_ * head_dim_);

      create_gpu_task(flow_id, cuda::gemm_strided_batched_ex<Compute_T>, s_ptr_b, v_ptr_b,
                      out_ptr_b, L, head_dim_, L, false, false, static_cast<Compute_T>(1.0),
                      static_cast<Compute_T>(0.0), num_heads_, stride_s2, stride_v2, stride_o2,
                      lda_s2, ldb_v2, ldc_o2);
    }
    return nullptr;
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_attention_forward.");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> AttentionBlock::compute_attention_backward(
    const Tensor &q, const Tensor &k, const Tensor &v, const Tensor &d_attn_out, Tensor &dq,
    Tensor &dk, Tensor &dv, size_t batch_size, size_t seq_len, const std::string &flow_id) {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "AttentionBlock mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (q->data_type() != dtype_of<IO_T>() || k->data_type() != dtype_of<IO_T>() ||
      v->data_type() != dtype_of<IO_T>() || d_attn_out->data_type() != dtype_of<IO_T>() ||
      dq->data_type() != dtype_of<IO_T>() || dk->data_type() != dtype_of<IO_T>() ||
      dv->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("AttentionBlock IO tensor dtype mismatch with dispatch IO_T");
  }

  if (this->device_->device_type() == DeviceType::CPU) {
    auto q_raw = q->data_as<Compute_T>();
    auto k_raw = k->data_as<Compute_T>();
    auto v_raw = v->data_as<Compute_T>();
    auto d_out_raw = d_attn_out->data_as<Compute_T>();

    auto dq_ptr = dq->data_as<Compute_T>();
    auto dk_ptr = dk->data_as<Compute_T>();
    auto dv_ptr = dv->data_as<Compute_T>();

    size_t L = seq_len;
    size_t batch_count = batch_size * num_heads_;
    Compute_T alpha = static_cast<Compute_T>(1.0 / std::sqrt(static_cast<double>(head_dim_)));

    Tensor scores = this->get_buffer({batch_count, L, L});
    Tensor dscores = this->get_buffer({batch_count, L, L});
    Tensor dattn = this->get_buffer({batch_count, L, L});

    Compute_T *s_ptr = scores->data_as<Compute_T>();
    Compute_T *ds_ptr = dscores->data_as<Compute_T>();
    Compute_T *da_ptr = dattn->data_as<Compute_T>();

    for (size_t b = 0; b < batch_size; ++b) {
      auto q_b = q_raw + b * (L * embed_dim_);
      auto k_b = k_raw + b * (L * embed_dim_);
      auto s_b = s_ptr + b * (num_heads_ * L * L);

      create_cpu_task(flow_id, cpu::gemm_strided_batched_ex<Compute_T>, q_b, k_b, s_b, L, L,
                      head_dim_, false, true, alpha, static_cast<Compute_T>(0.0), num_heads_,
                      head_dim_, head_dim_, L * L, embed_dim_, embed_dim_, L);
    }

    if (is_causal_) {
      create_cpu_task(flow_id, cpu::apply_causal_mask<Compute_T>, s_ptr, batch_count, L,
                      static_cast<Compute_T>(-INFINITY));
    }

    create_cpu_task(flow_id, cpu::softmax_forward<Compute_T>, s_ptr, s_ptr, batch_count * L, L);

    for (size_t b = 0; b < batch_size; ++b) {
      auto s_b = s_ptr + b * (num_heads_ * L * L);
      auto dout_b = d_out_raw + b * (L * embed_dim_);
      auto v_b = v_raw + b * (L * embed_dim_);

      auto dv_b = dv_ptr + b * (L * embed_dim_);
      auto ds_b = ds_ptr + b * (num_heads_ * L * L);

      create_cpu_task(flow_id, cpu::gemm_strided_batched_ex<Compute_T>, s_b, dout_b, dv_b, L,
                      head_dim_, L, true, false, static_cast<Compute_T>(1.0),
                      static_cast<Compute_T>(0.0), num_heads_, L * L, head_dim_, head_dim_, L,
                      embed_dim_, embed_dim_);

      create_cpu_task(flow_id, cpu::gemm_strided_batched_ex<Compute_T>, dout_b, v_b, ds_b, L, L,
                      head_dim_, false, true, static_cast<Compute_T>(1.0),
                      static_cast<Compute_T>(0.0), num_heads_, head_dim_, head_dim_, L * L,
                      embed_dim_, embed_dim_, L);
    }

    create_cpu_task(flow_id, cpu::softmax_backward<Compute_T>, s_ptr, ds_ptr, da_ptr,
                    batch_count * L, L);

    if (is_causal_) {
      create_cpu_task(flow_id, cpu::apply_causal_mask<Compute_T>, da_ptr, batch_count, L,
                      static_cast<Compute_T>(0.0));
    }

    for (size_t b = 0; b < batch_size; ++b) {
      auto da_b = da_ptr + b * (num_heads_ * L * L);
      auto k_b = k_raw + b * (L * embed_dim_);
      auto q_b = q_raw + b * (L * embed_dim_);
      auto dq_b = dq_ptr + b * (L * embed_dim_);
      auto dk_b = dk_ptr + b * (L * embed_dim_);

      create_cpu_task(flow_id, cpu::gemm_strided_batched_ex<Compute_T>, da_b, k_b, dq_b, L,
                      head_dim_, L, false, false, alpha, static_cast<Compute_T>(0.0), num_heads_,
                      L * L, head_dim_, head_dim_, L, embed_dim_, embed_dim_);

      create_cpu_task(flow_id, cpu::gemm_strided_batched_ex<Compute_T>, da_b, q_b, dk_b, L,
                      head_dim_, L, true, false, alpha, static_cast<Compute_T>(0.0), num_heads_,
                      L * L, head_dim_, head_dim_, L, embed_dim_, embed_dim_);
    }
    return nullptr;
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    auto q_raw = q->data_as<Compute_T>();
    auto k_raw = k->data_as<Compute_T>();
    auto v_raw = v->data_as<Compute_T>();
    auto d_out_raw = d_attn_out->data_as<Compute_T>();

    auto dq_ptr = dq->data_as<Compute_T>();
    auto dk_ptr = dk->data_as<Compute_T>();
    auto dv_ptr = dv->data_as<Compute_T>();

    size_t L = seq_len;
    size_t batch_count = batch_size * num_heads_;
    Compute_T alpha = static_cast<Compute_T>(1.0 / std::sqrt(static_cast<double>(head_dim_)));

#ifdef USE_CUDNN
    auto cuda_context = dynamic_cast<CUDAContext *>(this->device_->context());
#endif

    Tensor scores = this->get_buffer({batch_count, L, L});
    Compute_T *s_ptr = scores->data_as<Compute_T>();

    for (size_t b = 0; b < batch_size; ++b) {
      auto q_b = q_raw + b * (L * embed_dim_);
      auto k_b = k_raw + b * (L * embed_dim_);
      auto s_b = s_ptr + b * (num_heads_ * L * L);

      create_gpu_task(flow_id, cuda::gemm_strided_batched_ex<Compute_T>, q_b, k_b, s_b, L, L,
                      head_dim_, false, true, alpha, static_cast<Compute_T>(0.0), num_heads_,
                      head_dim_, head_dim_, L * L, embed_dim_, embed_dim_, L);
    }

    if (is_causal_) {
      create_gpu_task(flow_id, cuda::apply_causal_mask<Compute_T>, s_ptr, batch_count, L,
                      static_cast<Compute_T>(-INFINITY));
    }

#ifdef USE_CUDNN
    create_gpu_task(flow_id, cuda::softmax_forward<Compute_T>, cuda_context->getCudnnHandle(),
                    s_ptr, s_ptr, batch_count * L, L);
#else
    throw std::runtime_error("AttentionBlock requires CUDNN for Softmax on GPU");
#endif

    for (size_t b = 0; b < batch_size; ++b) {
      auto s_b = s_ptr + b * (num_heads_ * L * L);
      auto dout_b = d_out_raw + b * (L * embed_dim_);
      auto dv_b = dv_ptr + b * (L * embed_dim_);

      create_gpu_task(flow_id, cuda::gemm_strided_batched_ex<Compute_T>, s_b, dout_b, dv_b, L,
                      head_dim_, L, true, false, static_cast<Compute_T>(1.0),
                      static_cast<Compute_T>(0.0), num_heads_, L * L, head_dim_, head_dim_, L,
                      embed_dim_, embed_dim_);
    }

    Tensor dscores = this->get_buffer({batch_count, L, L});
    Compute_T *ds_ptr = dscores->data_as<Compute_T>();

    for (size_t b = 0; b < batch_size; ++b) {
      auto dout_b = d_out_raw + b * (L * embed_dim_);
      auto v_b = v_raw + b * (L * embed_dim_);
      auto ds_b = ds_ptr + b * (num_heads_ * L * L);

      create_gpu_task(flow_id, cuda::gemm_strided_batched_ex<Compute_T>, dout_b, v_b, ds_b, L, L,
                      head_dim_, false, true, static_cast<Compute_T>(1.0),
                      static_cast<Compute_T>(0.0), num_heads_, head_dim_, head_dim_, L * L,
                      embed_dim_, embed_dim_, L);
    }

    Tensor dattn = this->get_buffer({batch_count, L, L});
    Compute_T *da_ptr = dattn->data_as<Compute_T>();

#ifdef USE_CUDNN
    create_gpu_task(flow_id, cuda::softmax_backward<Compute_T>, cuda_context->getCudnnHandle(),
                    s_ptr, ds_ptr, da_ptr, batch_count * L, L);
#else
    throw std::runtime_error("AttentionBlock requires CUDNN for SoftmaxBackward on GPU");
#endif

    if (is_causal_) {
      create_gpu_task(flow_id, cuda::apply_causal_mask<Compute_T>, da_ptr, batch_count, L,
                      static_cast<Compute_T>(0.0));
    }

    for (size_t b = 0; b < batch_size; ++b) {
      auto da_b = da_ptr + b * (num_heads_ * L * L);
      auto k_b = k_raw + b * (L * embed_dim_);
      auto q_b = q_raw + b * (L * embed_dim_);
      auto dq_b = dq_ptr + b * (L * embed_dim_);
      auto dk_b = dk_ptr + b * (L * embed_dim_);

      create_gpu_task(flow_id, cuda::gemm_strided_batched_ex<Compute_T>, da_b, k_b, dq_b, L,
                      head_dim_, L, false, false, alpha, static_cast<Compute_T>(0.0), num_heads_,
                      L * L, head_dim_, head_dim_, L, embed_dim_, embed_dim_);

      create_gpu_task(flow_id, cuda::gemm_strided_batched_ex<Compute_T>, da_b, q_b, dk_b, L,
                      head_dim_, L, true, false, alpha, static_cast<Compute_T>(0.0), num_heads_,
                      L * L, head_dim_, head_dim_, L, embed_dim_, embed_dim_);
    }
    return nullptr;
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_attention_backward.");
  }
  return nullptr;
}

// // Explicit template instantiations for all type combinations
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<float, float, float>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<float, float, double>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<float, float, __half>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<float, double, float>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<float, double, double>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<float, double, __half>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<float, __half, float>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<float, __half, double>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<float, __half, __half>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<double, float, float>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<double, float, double>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<double, float, __half>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<double, double, float>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<double, double, double>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<double, double, __half>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<double, __half, float>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<double, __half, double>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<double, __half, __half>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<__half, float, float>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<__half, float, double>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<__half, float, __half>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<__half, double, float>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<__half, double, double>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<__half, double, __half>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<__half, __half, float>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<__half, __half, double>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_forward<__half, __half, __half>(
//     const Tensor &, const Tensor &, const Tensor &, Tensor &, size_t, size_t, const std::string
//     &);

// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<float, float, float>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<float, float, double>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<float, float, __half>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<float, double, float>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<float, double, double>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<float, double, __half>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<float, __half, float>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<float, __half, double>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<float, __half, __half>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<double, float, float>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<double, float, double>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<double, float, __half>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<double, double, float>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<double, double,
// double>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<double, double,
// __half>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<double, __half, float>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<double, __half,
// double>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<double, __half,
// __half>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<__half, float, float>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<__half, float, double>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<__half, float, __half>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<__half, double, float>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<__half, double,
// double>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<__half, double,
// __half>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<__half, __half, float>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<__half, __half,
// double>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);
// template std::unique_ptr<Task> AttentionBlock::compute_attention_backward<__half, __half,
// __half>(
//     const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor &, Tensor &, Tensor &,
//     size_t, size_t, const std::string &);

} // namespace tnn
