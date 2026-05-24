/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/maxpool2d_layer.hpp"

#include "device/task.hpp"
#include "nn/layers_impl/cpu/maxpool_ops.hpp"
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/maxpool_ops.hpp"
#endif
#ifdef USE_DNNL
#include "nn/layers_impl/common/maxpool.hpp"
#include "nn/layers_impl/cpu/dnnl_maxpool_ops.hpp"
#include "utils/misc.hpp"
#endif
#include <cstddef>
#include <stdexcept>

namespace tnn {

MaxPool2DLayer::MaxPool2DLayer(size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,
                               size_t pad_h, size_t pad_w, const std::string &name)
    : StatelessLayer(name),
      pool_h_(pool_h),
      pool_w_(pool_w),
      stride_h_(stride_h == 0 ? pool_h : stride_h),
      stride_w_(stride_w == 0 ? pool_w : stride_w),
      pad_h_(pad_h),
      pad_w_(pad_w) {
  if (pool_h_ == 0 || pool_w_ == 0) {
    throw std::invalid_argument("Pool dimensions must be positive");
  }
  if (stride_h_ == 0 || stride_w_ == 0) {
    throw std::invalid_argument("Stride dimensions must be positive");
  }
}

MaxPool2DLayer::~MaxPool2DLayer() {
#ifdef USE_DNNL
  for (auto &pair : dnnl_handle_cache) {
    if (pair.second) {
      cpu::dnnl_maxpool::destroy_dnnl_handle(pair.second);
    }
  }
  dnnl_handle_cache.clear();
  dnnl_stats_cache.clear();
#endif
}

Tensor MaxPool2DLayer::forward_impl(const ConstTensor &input, size_t mb_id) {
  const auto &shape = input->shape();
  if (shape.size() != 4) {
    throw std::runtime_error("MaxPool2DLayer: input must be 4D (NHWC format)");
  }
  const size_t batch_size = shape[0];
  const size_t input_h = shape[1];
  const size_t input_w = shape[2];
  const size_t channels = shape[3];

  micro_batch_input_shapes_[mb_id] = {batch_size, input_h, input_w, channels};

  const size_t output_h = (input_h + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  const size_t output_w = (input_w + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

#ifdef USE_DNNL
  if (get_engine_type() == EngineType::CPU) {
    return dnnl_forward(input, mb_id);
  }
#endif

  if (is_training_) {
    Tensor mask_indices =
        this->get_cache_tensor({batch_size, output_h, output_w, channels}, DType_t::INT32_T);
    set_mutable_cache(mb_id, "mask_indices", mask_indices);

    Tensor output = get_output_tensor({batch_size, output_h, output_w, channels});

    run_forward(input, output, batch_size, input_h, input_w, channels, output_h, output_w,
                mask_indices, this->flow_handle_);

    return output;
  } else {
    Tensor output = get_output_tensor({batch_size, output_h, output_w, channels});

    Tensor mask_indices =
        this->get_cache_tensor({batch_size, output_h, output_w, channels}, DType_t::INT32_T);

    run_forward(input, output, batch_size, input_h, input_w, channels, output_h, output_w,
                mask_indices, this->flow_handle_);

    return output;
  }
}

Tensor MaxPool2DLayer::backward_impl(const ConstTensor &grad_output, size_t mb_id) {
#ifdef USE_DNNL
  if (get_engine_type() == EngineType::CPU) {
    return dnnl_backward(grad_output, mb_id);
  }
#endif

  const ConstTensor &mask_indices = this->get_mutable_cache(mb_id, "mask_indices");
  const Vec<size_t> &input_shape = micro_batch_input_shapes_[mb_id];

  const size_t batch_size = input_shape[0];
  const size_t input_h = input_shape[1];
  const size_t input_w = input_shape[2];
  const size_t channels = input_shape[3];
  const auto &grad_shape = grad_output->shape();
  if (grad_shape.size() != 4) {
    throw std::runtime_error("MaxPool2DLayer: grad_output must be 4D (NHWC format)");
  }
  const size_t output_h = grad_shape[1];
  const size_t output_w = grad_shape[2];

  Tensor grad_input = get_output_tensor({batch_size, input_h, input_w, channels});

  grad_input->fill(0);

  run_backward(grad_output, grad_input, batch_size, channels, output_h, output_w, mask_indices,
               this->flow_handle_);

  return grad_input;
}

template <typename IO_T>
std::unique_ptr<Task> MaxPool2DLayer::run_forward(const ConstTensor &input_data,
                                                  const Tensor &output_data, size_t batch_size,
                                                  size_t height, size_t width, size_t channels,
                                                  size_t output_h, size_t output_w,
                                                  const Tensor &mask_indices,
                                                  flowHandle_t handle) const {
  if (input_data->data_type() != dtype_of<IO_T>() || output_data->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("MaxPool2DLayer: data type mismatch in forward pass");
  }

  if (input_data->device_type() == DeviceType::CPU) {
    cpu::maxpool::run_forward<IO_T>(input_data->data_as<IO_T>(), output_data->data_as<IO_T>(),
                                    mask_indices->data_as<int>(), batch_size, height, width,
                                    channels, pool_h_, pool_w_, stride_h_, stride_w_, pad_h_,
                                    pad_w_, output_h, output_w);
  }
#ifdef USE_CUDA
  else if (input_data->device_type() == DeviceType::GPU) {
    cuda::maxpool::run_forward<IO_T>(input_data->data_as<IO_T>(), output_data->data_as<IO_T>(),
                                     mask_indices->data_as<int>(), batch_size, height, width,
                                     channels, pool_h_, pool_w_, stride_h_, stride_w_, pad_h_,
                                     pad_w_, output_h, output_w);
  }
#endif
  else {
    throw std::runtime_error("MaxPool2DLayer: unsupported device type");
  }
  return nullptr;
}

std::unique_ptr<Task> MaxPool2DLayer::run_forward(const ConstTensor &input_data,
                                                  const Tensor &output_data, size_t batch_size,
                                                  size_t height, size_t width, size_t channels,
                                                  size_t output_h, size_t output_w,
                                                  const Tensor &mask_indices,
                                                  flowHandle_t handle) const {
  DISPATCH_IO_DTYPE(run_forward, input_data, output_data, batch_size, height, width, channels,
                    output_h, output_w, mask_indices, handle);
  return nullptr;
}

template <typename IO_T>
std::unique_ptr<Task> MaxPool2DLayer::run_backward(const ConstTensor &gradient_data,
                                                   const Tensor &grad_input_data, size_t batch_size,
                                                   size_t channels, size_t output_h,
                                                   size_t output_w, const ConstTensor &mask_indices,
                                                   flowHandle_t handle) const {
  if (gradient_data->data_type() != dtype_of<IO_T>() ||
      grad_input_data->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("MaxPool2DLayer: data type mismatch in backward pass");
  }

  if (gradient_data->device_type() == DeviceType::CPU) {
    cpu::maxpool::run_backward<IO_T>(gradient_data->data_as<IO_T>(),
                                     grad_input_data->data_as<IO_T>(), mask_indices->data_as<int>(),
                                     batch_size, channels, output_h, output_w);
  }
#ifdef USE_CUDA
  else if (gradient_data->device_type() == DeviceType::GPU) {
    cuda::maxpool::run_backward<IO_T>(
        gradient_data->data_as<IO_T>(), grad_input_data->data_as<IO_T>(),
        mask_indices->data_as<int>(), batch_size, channels, output_h, output_w);
  }
#endif
  else {
    throw std::runtime_error("MaxPool2DLayer: unsupported device type");
  }
  return nullptr;
}

std::unique_ptr<Task> MaxPool2DLayer::run_backward(const ConstTensor &gradient_data,
                                                   const Tensor &grad_input_data, size_t batch_size,
                                                   size_t channels, size_t output_h,
                                                   size_t output_w, const ConstTensor &mask_indices,
                                                   flowHandle_t handle) const {
  DISPATCH_IO_DTYPE(run_backward, gradient_data, grad_input_data, batch_size, channels, output_h,
                    output_w, mask_indices, handle);
  return nullptr;
}

LayerConfig MaxPool2DLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("pool_h", pool_h_);
  config.set("pool_w", pool_w_);
  config.set("stride_h", stride_h_);
  config.set("stride_w", stride_w_);
  config.set("pad_h", pad_h_);
  config.set("pad_w", pad_w_);
  return config;
}

Vec<size_t> MaxPool2DLayer::compute_output_shape(const Vec<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("MaxPool2DLayer: input shape must be 4D (NHWC format)");
  }

  size_t batch_size = input_shape[0];
  size_t output_h = (input_shape[1] + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  size_t output_w = (input_shape[2] + 2 * pad_w_ - pool_w_) / stride_w_ + 1;
  size_t channels = input_shape[3];

  return {batch_size, output_h, output_w, channels};
}

std::unique_ptr<MaxPool2DLayer> MaxPool2DLayer::create_from_config(const LayerConfig &config) {
  size_t pool_h = config.get<size_t>("pool_h");
  size_t pool_w = config.get<size_t>("pool_w");
  size_t stride_h = config.get<size_t>("stride_h");
  size_t stride_w = config.get<size_t>("stride_w");
  size_t pad_h = config.get<size_t>("pad_h");
  size_t pad_w = config.get<size_t>("pad_w");

  return std::make_unique<MaxPool2DLayer>(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                          config.name);
}

#ifdef USE_DNNL
void MaxPool2DLayer::build_dnnl_handle(const Vec<size_t> &input_shape) const {
  size_t shape_key = get_shape_hash(input_shape);
  if (dnnl_handle_cache.find(shape_key) == dnnl_handle_cache.end()) {
    MaxPoolStats new_stats;
    init_maxpool_stats(new_stats, input_shape[0], input_shape[1], input_shape[2], input_shape[3],
                       pool_h_, pool_w_, stride_h_, stride_w_, pad_h_, pad_w_);
    dnnl_handle_cache[shape_key] = cpu::dnnl_maxpool::initialize_dnnl_handle(new_stats, io_dtype_);
    dnnl_stats_cache[shape_key] = new_stats;
  }
}

Tensor MaxPool2DLayer::dnnl_forward(const ConstTensor &input, size_t mb_id) {
  build_dnnl_handle(input->shape());
  const size_t shape_key = get_shape_hash(input->shape());
  cpu::dnnl_maxpool::dnnlMaxPoolHandle_t *dnnl_handle = dnnl_handle_cache.at(shape_key);
  const MaxPoolStats &current_stats = dnnl_stats_cache.at(shape_key);

  Tensor output = get_output_tensor({current_stats.batch_size, current_stats.output_h,
                                     current_stats.output_w, current_stats.channels});

  if (this->is_training_) {
    Tensor pool_ws = get_cache_tensor({current_stats.pool_workspace_size}, DType_t::BYTE);
    set_mutable_cache(mb_id, "dnnl_pool_ws", pool_ws);

    create_cpu_task(this->flow_handle_, cpu::dnnl_maxpool::run_forward, dnnl_handle, current_stats,
                    input->data(), output->data(), pool_ws->data(), nullptr);
  } else {
    create_cpu_task(this->flow_handle_, cpu::dnnl_maxpool::run_inference, dnnl_handle,
                    current_stats, input->data(), output->data(), nullptr);
  }

  return output;
}

Tensor MaxPool2DLayer::dnnl_backward(const ConstTensor &grad_output, size_t mb_id) {
  const Vec<size_t> &input_shape = micro_batch_input_shapes_[mb_id];

  build_dnnl_handle(input_shape);
  const size_t shape_key = get_shape_hash(input_shape);
  cpu::dnnl_maxpool::dnnlMaxPoolHandle_t *dnnl_handle = dnnl_handle_cache.at(shape_key);
  const MaxPoolStats &current_stats = dnnl_stats_cache.at(shape_key);

  Tensor grad_input = get_output_tensor(input_shape);
  Tensor &pool_ws = this->get_mutable_cache(mb_id, "dnnl_pool_ws");

  create_cpu_task(this->flow_handle_, cpu::dnnl_maxpool::run_backward, dnnl_handle, current_stats,
                  grad_output->data(), grad_input->data(), pool_ws->data(), nullptr);

  pool_ws = nullptr;
  return grad_input;
}
#endif  // USE_DNNL

}  // namespace tnn
