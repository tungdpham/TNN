/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/dense_layer.hpp"

#include "device/task.hpp"
#include "nn/layers_impl/cpu/dense_ops.hpp"
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/dense_ops.hpp"
#endif
#ifdef USE_CUDNN
#include "cuda/cudnn/common.hpp"
#include "device/cuda/cuda_context.hpp"
#include "math/cuda/cudnn_gemm.hpp"
#endif
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "nn/layers_impl/parameterized_layer.hpp"
#include "type/type.hpp"

namespace tnn {

DenseLayer::DenseLayer(size_t input_features, size_t output_features, bool use_bias,
                       const std::string &name)
    : ParameterizedLayer(name),
      input_features_(input_features),
      output_features_(output_features),
      use_bias_(use_bias) {}

DenseLayer::~DenseLayer() {
#ifdef USE_CUDNN
  for (auto &pair : handle_cache) {
    cuda::cudnn_gemm::destroy_fe_handle(pair.second);
  }
  handle_cache.clear();
#endif
}

size_t DenseLayer::get_shape_hash(size_t batch_size) const {
  size_t seed = 0;
  auto hash_combine = [&](size_t v) { seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2); };
  hash_combine(batch_size);
  hash_combine(input_features_);
  hash_combine(output_features_);
  return seed;
}

void DenseLayer::init_params() {
  weights_ = make_param_tensor({output_features_, input_features_});
  weight_gradients_ = make_param_tensor({output_features_, input_features_});
  weight_gradients_->fill(0);
  if (use_bias_) {
    bias_ = make_param_tensor({output_features_});
    bias_gradients_ = make_param_tensor({output_features_});
    bias_gradients_->fill(0);
  }
  // PyTorch default Kaiming Uniform: Uniform(-bound, bound) where bound = 1 / sqrt(fan_in)
  double bound = 1.0 / std::sqrt(static_cast<double>(input_features_));

  if (this->use_seed_) {
    weights_->fill_random_uniform(-bound, bound, this->srand_seed_);
  } else {
    weights_->fill_random_uniform(-bound, bound);
  }

  if (use_bias_) {
    if (this->use_seed_) {
      bias_->fill_random_uniform(-bound, bound, this->srand_seed_);
    } else {
      bias_->fill_random_uniform(-bound, bound);
    }
  }
}

void DenseLayer::forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id) {
  const std::vector<size_t> &in_shape = input->shape();
  size_t last_dim = in_shape.back();

  if (last_dim != input_features_) {
    std::cerr << "Input last dimension: " << last_dim << " features, expected: " << input_features_
              << " features" << std::endl;
    throw std::invalid_argument("Input feature size mismatch in DenseLayer");
  }

  if (this->is_training_) {
    ConstTensor &cached_input = this->get_cached_tensor(mb_id, "input");
    cached_input = input;
  }

  std::vector<size_t> out_shape = in_shape;
  out_shape.back() = output_features_;
  output->ensure(out_shape);

#ifdef USE_CUDNN
  if (this->device_->device_type() == DeviceType::GPU) {
    cudnn_forward(input, output, mb_id);
    return;
  }
#endif
}

void DenseLayer::backward_impl(const ConstTensor &gradient, const Tensor &grad_input,
                               size_t mb_id) {
  if (gradient->shape().back() != output_features_) {
    throw std::invalid_argument("Gradient feature size mismatch in DenseLayer");
  }
#ifdef USE_CUDNN
  if (this->device_->device_type() == DeviceType::GPU) {
    cudnn_backward(gradient, grad_input, mb_id);
    return;
  }
#endif
}

#ifdef USE_CUDNN
void DenseLayer::cudnn_forward(const ConstTensor &input, const Tensor &output, size_t mb_id) {
  const std::vector<size_t> &in_shape = input->shape();
  size_t batch_size = 1;
  for (size_t i = 0; i < in_shape.size() - 1; ++i) {
    batch_size *= in_shape[i];
  }

  size_t shape_key = get_shape_hash(batch_size);

  cuda::cudnn_gemm::feHandle_t *handle = nullptr;

  if (handle_cache.find(shape_key) == handle_cache.end()) {
    cudnnDataType_t io_dtype = cuda::cudnn::to_cudnn_datatype(io_dtype_);
    cudnnDataType_t param_dtype = cuda::cudnn::to_cudnn_datatype(param_dtype_);
    cudnnDataType_t compute_dtype = cuda::cudnn::to_cudnn_datatype(compute_dtype_);

    CUDAContext *context = dynamic_cast<CUDAContext *>(this->device_->context());
    cudnnHandle_t cudnn_handle = context->getCudnnHandle();

    GemmStats stats;

    init_gemm_stats(stats, batch_size, output_features_, input_features_);

    handle = cuda::cudnn_gemm::initialize_fe_handle(cudnn_handle, io_dtype, param_dtype,
                                                    compute_dtype, stats);
    handle_cache[shape_key] = handle;
    stats_cache[shape_key] = stats;
  }

  handle = handle_cache[shape_key];
  GemmStats &stats = stats_cache[shape_key];

  size_t io_dtype_size = get_dtype_size(io_dtype_);
  size_t max_workspace_size =
      std::max({stats.fwd_workspace_size, stats.dgrad_workspace_size, stats.wgrad_workspace_size});
  size_t workspace_elements = (max_workspace_size + io_dtype_size - 1) / io_dtype_size;
  Tensor cudnn_workspace = this->get_buffer({workspace_elements});

  create_cuda_task("default", cuda::cudnn_gemm::run_forward, handle, stats, input->data(),
                   weights_->data(), output->data(), cudnn_workspace->data());

  if (use_bias_) {
    DISPATCH_ON_3_DTYPES_TO_METHOD(add_bias_vector, output, bias_, batch_size, output_features_,
                                   "default");
  }
}

void DenseLayer::cudnn_backward(const ConstTensor &gradient, const Tensor &grad_input,
                                size_t mb_id) {
  ConstTensor &input = this->get_cached_tensor(mb_id, "input");
  if (!input) {
    throw std::runtime_error("No cached input found for micro-batch ID: " + std::to_string(mb_id));
  }

  grad_input->ensure(input->shape());

  const std::vector<size_t> &in_shape = input->shape();
  size_t batch_size = 1;
  for (size_t i = 0; i < in_shape.size() - 1; ++i) {
    batch_size *= in_shape[i];
  }

  size_t shape_key = get_shape_hash(batch_size);
  cuda::cudnn_gemm::feHandle_t *handle = handle_cache.at(shape_key);

  GemmStats &stats = stats_cache.at(shape_key);

  size_t io_dtype_size = get_dtype_size(io_dtype_);
  size_t max_workspace_size =
      std::max({stats.fwd_workspace_size, stats.dgrad_workspace_size, stats.wgrad_workspace_size});
  size_t workspace_elements = (max_workspace_size + io_dtype_size - 1) / io_dtype_size;
  Tensor cudnn_workspace = this->get_buffer({workspace_elements});

  // Compute weight gradients
  create_cuda_task("default", cuda::cudnn_gemm::run_wgrad, handle, stats, input->data(),
                   gradient->data(), weight_gradients_->data(), cudnn_workspace->data());

  if (use_bias_) {
    DISPATCH_ON_3_DTYPES_TO_METHOD(compute_bias_gradients, gradient, bias_gradients_, batch_size,
                                   output_features_, "default");
  }

  // Compute input gradients
  create_cuda_task("default", cuda::cudnn_gemm::run_dgrad, handle, stats, gradient->data(),
                   weights_->data(), grad_input->data(), cudnn_workspace->data());
}
#endif

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> DenseLayer::compute_bias_gradients(const ConstTensor &gradient,
                                                         const Tensor &bias_gradient,
                                                         size_t batch_size, size_t output_features,
                                                         const std::string &flow_id) const {
  if (gradient->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("DenseLayer gradient dtype mismatch with dispatch IO_T");
  }
  if (bias_gradient->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("DenseLayer bias gradient dtype mismatch with dispatch Param_T");
  }
  if (this->device_->device_type() == DeviceType::CPU) {
    if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
      throw std::runtime_error(
          "DenseLayer mixed dtype dispatch not implemented for CPU (io/param/compute must match).");
    }
    return create_cpu_task(flow_id, cpu::legacy_dense::compute_bias_gradients<IO_T>,
                           gradient->data_as<IO_T>(), bias_gradient->data_as<IO_T>(), batch_size,
                           output_features);
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    return create_cuda_task(
        flow_id, cuda::legacy_dense::compute_bias_gradients_ex<IO_T, Param_T, Compute_T>,
        gradient->data_as<IO_T>(), bias_gradient->data_as<Param_T>(), batch_size, output_features);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_bias_gradients");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> DenseLayer::add_bias_vector(const Tensor &output, const ConstTensor &bias,
                                                  size_t batch_size, size_t output_features,
                                                  const std::string &flow_id) const {
  if (output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("DenseLayer output dtype mismatch with dispatch IO_T");
  }
  if (bias->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("DenseLayer bias dtype mismatch with dispatch Param_T");
  }
  if (this->device_->device_type() == DeviceType::CPU) {
    if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
      throw std::runtime_error(
          "DenseLayer mixed dtype dispatch not implemented for CPU (io/param/compute must match).");
    }
    return create_cpu_task(flow_id, cpu::legacy_dense::add_bias_vector<IO_T>,
                           output->data_as<IO_T>(), bias->data_as<IO_T>(), batch_size,
                           output_features);
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    return create_cuda_task(
        flow_id, cuda::legacy_dense::add_bias_vector_ex<IO_T, Param_T, Compute_T>,
        output->data_as<IO_T>(), bias->data_as<Param_T>(), batch_size, output_features);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for add_bias_vector");
  }
  return nullptr;
}

LayerConfig DenseLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("input_features", input_features_);
  config.set("output_features", output_features_);
  config.set("use_bias", use_bias_);
  return config;
}

std::unique_ptr<Layer> DenseLayer::clone() const {
  return std::make_unique<DenseLayer>(input_features_, output_features_, use_bias_, this->name_);
}

std::vector<size_t> DenseLayer::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (input_shape.empty()) {
    throw std::runtime_error("DenseLayer::compute_output_shape: Input shape is empty.");
  }
  std::vector<size_t> out_shape = input_shape;
  out_shape.back() = output_features_;
  return out_shape;
}

void DenseLayer::collect_parameters(std::vector<Tensor> &params) {
  params.push_back(weights_);
  if (use_bias_) {
    params.push_back(bias_);
  }
}

void DenseLayer::collect_gradients(std::vector<Tensor> &grads) {
  grads.push_back(weight_gradients_);
  if (use_bias_) {
    grads.push_back(bias_gradients_);
  }
}

std::unique_ptr<DenseLayer> DenseLayer::create_from_config(const LayerConfig &config) {
  size_t input_features = config.get<size_t>("input_features");
  size_t output_features = config.get<size_t>("output_features");
  bool use_bias = config.get<bool>("use_bias");

  return std::make_unique<DenseLayer>(input_features, output_features, use_bias, config.name);
}

uint64_t DenseLayer::forward_flops(const std::vector<size_t> &input_shape) const {
  size_t batch_size = input_shape[0];
  uint64_t gemm_flops = 2ULL * batch_size * input_features_ * output_features_;
  uint64_t bias_flops = use_bias_ ? (batch_size * output_features_) : 0;
  return gemm_flops + bias_flops;
}

uint64_t DenseLayer::backward_flops(const std::vector<size_t> &input_shape) const {
  size_t batch_size = input_shape[0];
  uint64_t weight_grad_flops = 2ULL * input_features_ * batch_size * output_features_;
  uint64_t bias_grad_flops = use_bias_ ? (batch_size * output_features_) : 0;
  uint64_t input_grad_flops = 2ULL * batch_size * output_features_ * input_features_;
  return weight_grad_flops + bias_grad_flops + input_grad_flops;
}

}  // namespace tnn
