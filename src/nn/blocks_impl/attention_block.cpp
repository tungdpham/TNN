/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/blocks_impl/attention_block.hpp"

#include "tensor/tensor.hpp"
#include "type/type.hpp"
#ifdef USE_CUDA
#include "device/cuda/cuda_context.hpp"
#include "math/cuda/gemm.hpp"
#include "nn/blocks_impl/cuda/causal_mask.hpp"
#include "nn/blocks_impl/cuda/permute_heads.hpp"
#include "nn/blocks_impl/cuda/softmax.hpp"
#endif
#include <cmath>
#include <stdexcept>

#include "ops/ops.hpp"

namespace tnn {

AttentionBlock::AttentionBlock(size_t embed_dim, size_t num_heads, bool is_causal,
                               const std::string &name)
    : Block(name),
      embed_dim_(embed_dim),
      num_heads_(num_heads),
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

void AttentionBlock::init_impl() {
  q_proj_->init();
  k_proj_->init();
  v_proj_->init();
  out_proj_->init();
}

void AttentionBlock::on_set_io_dtype(DType_t dtype) {
  q_proj_->set_io_dtype(dtype);
  k_proj_->set_io_dtype(dtype);
  v_proj_->set_io_dtype(dtype);
  out_proj_->set_io_dtype(dtype);
}

void AttentionBlock::on_set_param_dtype(DType_t dtype) {
  q_proj_->set_param_dtype(dtype);
  k_proj_->set_param_dtype(dtype);
  v_proj_->set_param_dtype(dtype);
  out_proj_->set_param_dtype(dtype);
}

void AttentionBlock::on_set_compute_dtype(DType_t dtype) {
  q_proj_->set_compute_dtype(dtype);
  k_proj_->set_compute_dtype(dtype);
  v_proj_->set_compute_dtype(dtype);
  out_proj_->set_compute_dtype(dtype);
}

void AttentionBlock::on_set_training(bool training) {
  this->is_training_ = training;
  q_proj_->set_training(training);
  k_proj_->set_training(training);
  v_proj_->set_training(training);
  out_proj_->set_training(training);
}

void AttentionBlock::on_set_device(const Device &device) {
  q_proj_->set_device(device);
  k_proj_->set_device(device);
  v_proj_->set_device(device);
  out_proj_->set_device(device);
}

void AttentionBlock::on_set_flow_handle(flowHandle_t handle) {
  q_proj_->set_flow_handle(handle);
  k_proj_->set_flow_handle(handle);
  v_proj_->set_flow_handle(handle);
  out_proj_->set_flow_handle(handle);
}

void AttentionBlock::on_set_seed(unsigned long long seed) {
  q_proj_->set_seed(seed);
  k_proj_->set_seed(seed);
  v_proj_->set_seed(seed);
  out_proj_->set_seed(seed);
}

void AttentionBlock::forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id) {
  const auto &input_shape = input->shape();

  size_t batch_size = input_shape[0];
  size_t seq_len = input_shape[1];

  if (this->is_training_) {
    ConstTensor &cached_input = this->get_cached_tensor(mb_id, "input");
    cached_input = input;
  }

  Tensor q = this->get_buffer({batch_size, seq_len, embed_dim_}, io_dtype_);
  Tensor k = this->get_buffer({batch_size, seq_len, embed_dim_}, io_dtype_);
  Tensor v = this->get_buffer({batch_size, seq_len, embed_dim_}, io_dtype_);

  q_proj_->forward(input, q, mb_id);
  k_proj_->forward(input, k, mb_id);
  v_proj_->forward(input, v, mb_id);

  Tensor attn_out = this->get_buffer(input_shape, io_dtype_);
  attn_out->ensure(input_shape);

  DISPATCH_ON_3_DTYPES_TO_METHOD(compute_attention_forward, q, k, v, attn_out, batch_size, seq_len,
                                 defaultFlowHandle);

  out_proj_->forward(attn_out, output, mb_id);
}

void AttentionBlock::backward_impl(const ConstTensor &gradient, const Tensor &grad_input,
                                   size_t mb_id) {
  ConstTensor &input = this->get_cached_tensor(mb_id, "input");
  if (!input) {
    throw std::runtime_error("No cached input found for micro-batch ID: " + std::to_string(mb_id));
  }

  size_t batch_size = input->dimension(0);
  size_t seq_len = input->dimension(1);

  Tensor q = this->get_buffer({batch_size, seq_len, embed_dim_}, io_dtype_);
  Tensor k = this->get_buffer({batch_size, seq_len, embed_dim_}, io_dtype_);
  Tensor v = this->get_buffer({batch_size, seq_len, embed_dim_}, io_dtype_);

  q_proj_->forward(input, q, mb_id);
  k_proj_->forward(input, k, mb_id);
  v_proj_->forward(input, v, mb_id);

  Tensor d_attn_out = this->get_buffer(gradient->shape(), io_dtype_);
  out_proj_->backward(gradient, d_attn_out, mb_id);

  Tensor dq = this->get_buffer(q->shape(), io_dtype_);
  Tensor dk = this->get_buffer(k->shape(), io_dtype_);
  Tensor dv = this->get_buffer(v->shape(), io_dtype_);

  DISPATCH_ON_3_DTYPES_TO_METHOD(compute_attention_backward, q, k, v, d_attn_out, dq, dk, dv,
                                 batch_size, seq_len, defaultFlowHandle);

  Tensor dq_in = this->get_buffer(q->shape(), io_dtype_);
  Tensor dk_in = this->get_buffer(k->shape(), io_dtype_);
  Tensor dv_in = this->get_buffer(v->shape(), io_dtype_);

  q_proj_->backward(dq, dq_in, mb_id);
  k_proj_->backward(dk, dk_in, mb_id);
  v_proj_->backward(dv, dv_in, mb_id);

  grad_input->ensure(dq_in->shape());
  size_t size = dq_in->size();

  Tensor temp = this->get_buffer(dq_in->shape(), io_dtype_);

  DISPATCH_ON_DTYPE_TO_METHOD(ops::add, dq_in->data_ptr(), dk_in->data_ptr(), temp->data_ptr(),
                              size, defaultFlowHandle);
  DISPATCH_ON_DTYPE_TO_METHOD(ops::add, temp->data_ptr(), dv_in->data_ptr(), grad_input->data_ptr(),
                              size, defaultFlowHandle);
}

uint64_t AttentionBlock::forward_flops(const std::vector<size_t> &input_shape) const { return 0; }

uint64_t AttentionBlock::backward_flops(const std::vector<size_t> &input_shape) const { return 0; }

LayerConfig AttentionBlock::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("embed_dim", embed_dim_);
  config.set("num_heads", num_heads_);
  return config;
}

std::unique_ptr<Layer> AttentionBlock::clone() const {
  return std::make_unique<AttentionBlock>(embed_dim_, num_heads_, is_causal_, this->name_);
}

std::vector<size_t> AttentionBlock::compute_output_shape(
    const std::vector<size_t> &input_shape) const {
  return input_shape;
}

std::vector<Tensor> AttentionBlock::parameters() {
  std::vector<Tensor> params;
  auto q_params = q_proj_->parameters();
  params.insert(params.end(), q_params.begin(), q_params.end());
  auto k_params = k_proj_->parameters();
  params.insert(params.end(), k_params.begin(), k_params.end());
  auto v_params = v_proj_->parameters();
  params.insert(params.end(), v_params.begin(), v_params.end());
  auto out_params = out_proj_->parameters();
  params.insert(params.end(), out_params.begin(), out_params.end());
  return params;
}

std::vector<Tensor> AttentionBlock::gradients() {
  std::vector<Tensor> grads;
  auto q_grads = q_proj_->gradients();
  grads.insert(grads.end(), q_grads.begin(), q_grads.end());
  auto k_grads = k_proj_->gradients();
  grads.insert(grads.end(), k_grads.begin(), k_grads.end());
  auto v_grads = v_proj_->gradients();
  grads.insert(grads.end(), v_grads.begin(), v_grads.end());
  auto out_grads = out_proj_->gradients();
  grads.insert(grads.end(), out_grads.begin(), out_grads.end());
  return grads;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> AttentionBlock::compute_attention_forward(
    const ConstTensor &q, const ConstTensor &k, const ConstTensor &v, const Tensor &output,
    size_t batch_size, size_t seq_len, flowHandle_t handle) {
  if (q->data_type() != dtype_of<IO_T>() || k->data_type() != dtype_of<IO_T>() ||
      v->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("AttentionBlock IO tensor dtype mismatch with dispatch IO_T");
  }

  if (this->device_->device_type() == DeviceType::CPU) {
    throw std::runtime_error("AttentionBlock CPU implementation not yet available.");
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {

    size_t L = seq_len;
    size_t batch_count = batch_size * num_heads_;

    Tensor q_heads = this->get_buffer({batch_size, num_heads_, L, head_dim_}, io_dtype_);
    Tensor k_heads = this->get_buffer({batch_size, num_heads_, L, head_dim_}, io_dtype_);
    Tensor v_heads = this->get_buffer({batch_size, num_heads_, L, head_dim_}, io_dtype_);

    create_cuda_task(handle, cuda::permute_heads<IO_T, IO_T>, q->data_as<IO_T>(),
                     q_heads->data_as<IO_T>(), batch_size, L, num_heads_, head_dim_);
    create_cuda_task(handle, cuda::permute_heads<IO_T, IO_T>, k->data_as<IO_T>(),
                     k_heads->data_as<IO_T>(), batch_size, L, num_heads_, head_dim_);
    create_cuda_task(handle, cuda::permute_heads<IO_T, IO_T>, v->data_as<IO_T>(),
                     v_heads->data_as<IO_T>(), batch_size, L, num_heads_, head_dim_);

    Tensor scores = this->get_buffer({batch_count, L, L}, io_dtype_);

    Compute_T alpha = static_cast<Compute_T>(1.0 / std::sqrt(static_cast<double>(head_dim_)));

    size_t strideA = L * head_dim_;
    size_t strideB = L * head_dim_;
    size_t strideC = L * L;

    create_cuda_task(handle, cuda::gemm_strided_batched_ex<IO_T, IO_T, IO_T, Compute_T>,
                     q_heads->data_as<IO_T>(), k_heads->data_as<IO_T>(), scores->data_as<IO_T>(), L,
                     L, head_dim_, false, true, alpha, static_cast<Compute_T>(0.0), head_dim_,
                     head_dim_, L, strideA, strideB, strideC, batch_count);

    if (is_causal_) {
      create_cuda_task(handle, cuda::apply_causal_mask<IO_T>, scores->data_as<IO_T>(), batch_count,
                       L, static_cast<IO_T>(-INFINITY));
    }

    auto context = dynamic_cast<CUDAContext *>(this->device_->context());
    if (!context) {
      throw std::runtime_error("AttentionBlock requires CUDAContext for CUDA operations.");
    }
    auto cudnn_handle = context->getCudnnHandle();

    create_cuda_task(handle, cuda::softmax_forward<IO_T>, cudnn_handle, scores->data_as<IO_T>(),
                     scores->data_as<IO_T>(), batch_count * L, L);

    Tensor attn_heads = this->get_buffer({batch_size, num_heads_, L, head_dim_}, io_dtype_);

    strideA = L * L;
    strideB = L * head_dim_;
    strideC = L * head_dim_;

    create_cuda_task(handle, cuda::gemm_strided_batched_ex<IO_T, IO_T, IO_T, Compute_T>,
                     scores->data_as<IO_T>(), v_heads->data_as<IO_T>(), attn_heads->data_as<IO_T>(),
                     L, head_dim_, L, false, false, static_cast<Compute_T>(1.0),
                     static_cast<Compute_T>(0.0), L, head_dim_, head_dim_, strideA, strideB,
                     strideC, batch_count);

    create_cuda_task(handle, cuda::permute_heads<IO_T, IO_T>, attn_heads->data_as<IO_T>(),
                     output->data_as<IO_T>(), batch_size, num_heads_, L, head_dim_);

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
    const ConstTensor &q, const ConstTensor &k, const ConstTensor &v, const ConstTensor &d_attn_out,
    const Tensor &dq, const Tensor &dk, const Tensor &dv, size_t batch_size, size_t seq_len,
    flowHandle_t handle) {
  if (q->data_type() != dtype_of<IO_T>() || k->data_type() != dtype_of<IO_T>() ||
      v->data_type() != dtype_of<IO_T>() || d_attn_out->data_type() != dtype_of<IO_T>() ||
      dq->data_type() != dtype_of<IO_T>() || dk->data_type() != dtype_of<IO_T>() ||
      dv->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("AttentionBlock IO tensor dtype mismatch with dispatch IO_T");
  }

  if (this->device_->device_type() == DeviceType::CPU) {
    throw std::runtime_error("AttentionBlock CPU implementation not yet available.");
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    auto q_raw = q->data_as<IO_T>();
    auto k_raw = k->data_as<IO_T>();
    auto v_raw = v->data_as<IO_T>();
    auto d_out_raw = d_attn_out->data_as<IO_T>();

    auto dq_ptr = dq->data_as<IO_T>();
    auto dk_ptr = dk->data_as<IO_T>();
    auto dv_ptr = dv->data_as<IO_T>();

    size_t L = seq_len;
    size_t batch_count = batch_size * num_heads_;
    Compute_T alpha = static_cast<Compute_T>(1.0 / std::sqrt(static_cast<double>(head_dim_)));

    Tensor q_heads = this->get_buffer({batch_size, num_heads_, L, head_dim_}, io_dtype_);
    Tensor k_heads = this->get_buffer({batch_size, num_heads_, L, head_dim_}, io_dtype_);
    Tensor v_heads = this->get_buffer({batch_size, num_heads_, L, head_dim_}, io_dtype_);

    create_cuda_task(handle, cuda::permute_heads<IO_T, IO_T>, q_raw, q_heads->data_as<IO_T>(),
                     batch_size, L, num_heads_, head_dim_);
    create_cuda_task(handle, cuda::permute_heads<IO_T, IO_T>, k_raw, k_heads->data_as<IO_T>(),
                     batch_size, L, num_heads_, head_dim_);
    create_cuda_task(handle, cuda::permute_heads<IO_T, IO_T>, v_raw, v_heads->data_as<IO_T>(),
                     batch_size, L, num_heads_, head_dim_);

    Tensor scores = this->get_buffer({batch_count, L, L}, io_dtype_);

    create_cuda_task(handle, cuda::gemm_strided_batched_ex<IO_T, IO_T, IO_T, Compute_T>,
                     q_heads->data_as<IO_T>(), k_heads->data_as<IO_T>(), scores->data_as<IO_T>(), L,
                     L, head_dim_, false, true, alpha, static_cast<Compute_T>(0.0), head_dim_,
                     head_dim_, L, L * head_dim_, L * head_dim_, L * L, batch_count);

    if (is_causal_) {
      create_cuda_task(handle, cuda::apply_causal_mask<IO_T>, scores->data_as<IO_T>(), batch_count,
                       L, static_cast<IO_T>(-INFINITY));
    }

    CUDAContext *context = dynamic_cast<CUDAContext *>(this->device_->context());
    if (!context) {
      throw std::runtime_error("AttentionBlock requires CUDAContext for CUDA operations.");
    }
    auto cudnn_handle = context->getCudnnHandle();

    create_cuda_task(handle, cuda::softmax_forward<IO_T>, cudnn_handle, scores->data_as<IO_T>(),
                     scores->data_as<IO_T>(), batch_count * L, L);

    Tensor d_attn_heads = this->get_buffer({batch_size, num_heads_, L, head_dim_}, io_dtype_);
    create_cuda_task(handle, cuda::permute_heads<IO_T, IO_T>, d_out_raw,
                     d_attn_heads->data_as<IO_T>(), batch_size, L, num_heads_, head_dim_);

    Tensor dv_heads = this->get_buffer({batch_size, num_heads_, L, head_dim_}, io_dtype_);
    create_cuda_task(handle, cuda::gemm_strided_batched_ex<IO_T, IO_T, IO_T, Compute_T>,
                     scores->data_as<IO_T>(), d_attn_heads->data_as<IO_T>(),
                     dv_heads->data_as<IO_T>(), L, head_dim_, L, true, false,
                     static_cast<Compute_T>(1.0), static_cast<Compute_T>(0.0), L, head_dim_,
                     head_dim_, L * L, L * head_dim_, L * head_dim_, batch_count);

    Tensor dscores = this->get_buffer({batch_count, L, L}, io_dtype_);

    create_cuda_task(handle, cuda::gemm_strided_batched_ex<IO_T, IO_T, IO_T, Compute_T>,
                     d_attn_heads->data_as<IO_T>(), v_heads->data_as<IO_T>(),
                     dscores->data_as<IO_T>(), L, L, head_dim_, false, true,
                     static_cast<Compute_T>(1.0), static_cast<Compute_T>(0.0), head_dim_, head_dim_,
                     L, L * head_dim_, L * head_dim_, L * L, batch_count);

    Tensor dattn = this->get_buffer({batch_count, L, L}, io_dtype_);

    create_cuda_task(handle, cuda::softmax_backward<IO_T>, cudnn_handle, scores->data_as<IO_T>(),
                     dscores->data_as<IO_T>(), dattn->data_as<IO_T>(), batch_count * L, L);

    if (is_causal_) {
      create_cuda_task(handle, cuda::apply_causal_mask<IO_T>, dattn->data_as<IO_T>(), batch_count,
                       L, static_cast<IO_T>(0.0));
    }

    Tensor dq_heads = this->get_buffer({batch_size, num_heads_, L, head_dim_}, io_dtype_);
    create_cuda_task(handle, cuda::gemm_strided_batched_ex<IO_T, IO_T, IO_T, Compute_T>,
                     dattn->data_as<IO_T>(), k_heads->data_as<IO_T>(), dq_heads->data_as<IO_T>(), L,
                     head_dim_, L, false, false, alpha, static_cast<Compute_T>(0.0), L, head_dim_,
                     head_dim_, L * L, L * head_dim_, L * head_dim_, batch_count);

    Tensor dk_heads = this->get_buffer({batch_size, num_heads_, L, head_dim_}, io_dtype_);
    create_cuda_task(handle, cuda::gemm_strided_batched_ex<IO_T, IO_T, IO_T, Compute_T>,
                     dattn->data_as<IO_T>(), q_heads->data_as<IO_T>(), dk_heads->data_as<IO_T>(), L,
                     head_dim_, L, true, false, alpha, static_cast<Compute_T>(0.0), L, head_dim_,
                     head_dim_, L * L, L * head_dim_, L * head_dim_, batch_count);

    create_cuda_task(handle, cuda::permute_heads<IO_T, IO_T>, dq_heads->data_as<IO_T>(), dq_ptr,
                     batch_size, num_heads_, L, head_dim_);
    create_cuda_task(handle, cuda::permute_heads<IO_T, IO_T>, dk_heads->data_as<IO_T>(), dk_ptr,
                     batch_size, num_heads_, L, head_dim_);
    create_cuda_task(handle, cuda::permute_heads<IO_T, IO_T>, dv_heads->data_as<IO_T>(), dv_ptr,
                     batch_size, num_heads_, L, head_dim_);

    return nullptr;
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_attention_backward.");
  }
  return nullptr;
}

std::unique_ptr<AttentionBlock> AttentionBlock::create_from_config(const LayerConfig &config) {
  size_t embed_dim = config.get<size_t>("embed_dim");
  size_t num_heads = config.get<size_t>("num_heads");
  bool is_causal = config.get<bool>("is_causal", true);
  return std::make_unique<AttentionBlock>(embed_dim, num_heads, is_causal, config.name);
}

}  // namespace tnn
