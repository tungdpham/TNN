#pragma once

#include "cpu/tensor_ops.hpp"
#include "device/task.hpp"
#ifdef USE_CUDA
#include "cuda/tensor_ops.hpp"
#endif
#include "tensor.hpp"

namespace tnn {
// im2col/col2im operations

template <typename T>
std::unique_ptr<Task> im2col(const Tensor<T, NCHW> &input_tensor, device_ptr<T[]> &col_data,
                             size_t kernel_h, size_t kernel_w, size_t stride_h = 1,
                             size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
                             const std::string &flow_id = "default") {
  if (col_data.getDeviceType() != input_tensor.device_type()) {
    throw std::runtime_error("im2col: Mismatched device types between col_data and input_tensor");
  }
  if (input_tensor.device_type() != col_data.getDeviceType()) {
    throw std::runtime_error("im2col: Mismatched device types between input tensor and col_data");
  }
  if (input_tensor.is_on_cpu()) {
    return create_cpu_task(flow_id, cpu::im2col<T>, input_tensor, col_data.get(), kernel_h,
                           kernel_w, stride_h, stride_w, pad_h, pad_w);
  }
#ifdef USE_CUDA
  else if (input_tensor.is_on_gpu()) {
    return create_gpu_task(flow_id, cuda::im2col<T>, input_tensor, col_data.get(), kernel_h,
                           kernel_w, stride_h, stride_w, pad_h, pad_w);
  }
#endif
  else {
    throw std::runtime_error("im2col: Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> col2im(const device_ptr<T[]> &col_data, device_ptr<T[]> &result_data,
                             size_t batch_size, size_t channels, size_t height, size_t width,
                             size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w,
                             size_t pad_h, size_t pad_w, const std::string &flow_id = "default") {
  if (col_data.getDeviceType() != result_data.getDeviceType()) {
    throw std::runtime_error("col2im: Mismatched device types between col_data and result_data");
  }
  if (col_data.getDeviceType() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::col2im<T>, col_data.get(), result_data.get(), batch_size,
                           channels, height, width, kernel_h, kernel_w, stride_h, stride_w, pad_h,
                           pad_w);
  }
#ifdef USE_CUDA
  else if (col_data.getDeviceType() == DeviceType::GPU) {
    return create_gpu_task(flow_id, cuda::col2im<T>, col_data.get(), result_data.get(), batch_size,
                           channels, height, width, kernel_h, kernel_w, stride_h, stride_w, pad_h,
                           pad_w);
  }
#endif
  else {
    throw std::runtime_error("col2im: Unsupported device type");
  }
}

// Padding operations

template <typename T>
std::unique_ptr<Task> pad(const Tensor<T, NCHW> &input, Tensor<T, NCHW> &result, size_t pad_h,
                          size_t pad_w, T value = T(0), const std::string &flow_id = "default") {
  if (input.is_on_cpu()) {
    return create_cpu_task(flow_id, cpu::pad<T>, input, result, pad_h, pad_w, value);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return create_gpu_task(flow_id, cuda::pad<T>, input, result, pad_h, pad_w, value);
  }
#endif
  else {
    throw std::runtime_error("pad: Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> unpad(const Tensor<T> &input, Tensor<T, NCHW> &result, size_t pad_h,
                            size_t pad_w, const std::string &flow_id = "default") {
  if (input.is_on_cpu()) {
    return create_cpu_task(flow_id, cpu::unpad<T>, input, result, pad_h, pad_w);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return create_gpu_task(flow_id, cuda::unpad<T>, input, result, pad_h, pad_w);
  }
#endif
  else {
    throw std::runtime_error("unpad: Unsupported device type");
  }
}

// Crop operation

template <typename T, Layout L>
std::unique_ptr<Task> crop(const Tensor<T, L> &input, Tensor<T, L> &result, const size_t start_h,
                           const size_t start_w, const size_t end_h, const size_t end_w,
                           const std::string &flow_id = "default") {
  if (input.is_on_cpu()) {
    return create_cpu_task(flow_id, cpu::crop<T, L>, input, result, start_h, start_w, end_h, end_w);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return create_gpu_task(flow_id, cuda::crop<T, L>, input, result, start_h, start_w, end_h,
                           end_w);
  }
#endif
  else {
    throw std::runtime_error("crop: Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> crop(const Tensor<T, NCHW> &input, Tensor<T, NCHW> &result,
                           const size_t start_h, const size_t start_w, const size_t end_h,
                           const size_t end_w, const std::string &flow_id = "default") {
  if (input.is_on_cpu()) {
    return create_cpu_task(flow_id, cpu::crop<T, NCHW>, input, result, start_h, start_w, end_h,
                           end_w);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return create_gpu_task(flow_id, cuda::crop<T, NCHW>, input, result, start_h, start_w, end_h,
                           end_w);
  }
#endif
  else {
    throw std::runtime_error("crop: Unsupported device type");
  }
}

// Slicing operations

template <typename T, Layout L>
std::unique_ptr<Task> slice_batch(const Tensor<T, L> &input, Tensor<T, L> &result,
                                  size_t start_batch, size_t end_batch,
                                  const std::string &flow_id = "default") {
  if (input.is_on_cpu()) {
    return create_cpu_task(flow_id, cpu::slice_batch<T, L>, input, result, start_batch, end_batch);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return create_gpu_task(flow_id, cuda::slice_batch<T, L>, input, result, start_batch, end_batch);
  }
#endif
  else {
    throw std::runtime_error("slice_batch: Unsupported device type");
  }
}

template <typename T, Layout L>
std::unique_ptr<Task> slice_channels(const Tensor<T, L> &input, Tensor<T, L> &result,
                                     size_t start_ch, size_t end_ch,
                                     const std::string &flow_id = "default") {
  if (input.is_on_cpu()) {
    return create_cpu_task(flow_id, cpu::slice_channels<T, L>, input, result, start_ch, end_ch);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return create_gpu_task(flow_id, cuda::slice_channels<T, L>, input, result, start_ch, end_ch);
  }
#endif
  else {
    throw std::runtime_error("slice_channels: Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> slice_channels(const Tensor<T, NCHW> &input, Tensor<T, NCHW> &result,
                                     size_t start_ch, size_t end_ch,
                                     const std::string &flow_id = "default") {
  if (input.is_on_cpu()) {
    return create_cpu_task(flow_id, cpu::slice_channels<T, NCHW>, input, result, start_ch, end_ch);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return create_gpu_task(flow_id, cuda::slice_channels<T, NCHW>, input, result, start_ch, end_ch);
  }
#endif
  else {
    throw std::runtime_error("slice_channels: Unsupported device type");
  }
}

// Split operation

template <typename T, Layout L>
std::unique_ptr<Task> split(const Tensor<T, L> &input, std::vector<Tensor<T, L>> &results,
                            size_t num_splits, const std::string &flow_id = "default") {
  if (input.is_on_cpu()) {
    return create_cpu_task(flow_id, cpu::split<T, L>, input, results, num_splits);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return create_gpu_task(flow_id, cuda::split<T, L>, input, results, num_splits);
  }
#endif
  else {
    throw std::runtime_error("split: Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> split(const Tensor<T, NCHW> &input, std::vector<Tensor<T, NCHW>> &results,
                            size_t num_splits, const std::string &flow_id = "default") {
  if (input.is_on_cpu()) {
    return create_cpu_task(flow_id, cpu::split<T, NCHW>, input, results, num_splits);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return create_gpu_task(flow_id, cuda::split<T, NCHW>, input, results, num_splits);
  }
#endif
  else {
    throw std::runtime_error("split: Unsupported device type");
  }
}

// Softmax operation

template <typename T, Layout L>
std::unique_ptr<Task> apply_softmax(Tensor<T, L> &input, const std::string &flow_id = "default") {
  if (input.is_on_cpu()) {
    return create_cpu_task(flow_id, cpu::apply_softmax<T, L>, input);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return create_gpu_task(flow_id, cuda::apply_softmax<T, L>, input);
  }
#endif
  else {
    throw std::runtime_error("apply_softmax: Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> apply_softmax(Tensor<T, NCHW> &input,
                                    const std::string &flow_id = "default") {
  if (input.is_on_cpu()) {
    return create_cpu_task(flow_id, cpu::apply_softmax<T, NCHW>, input);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return create_gpu_task(flow_id, cuda::apply_softmax<T, NCHW>, input);
  }
#endif
  else {
    throw std::runtime_error("apply_softmax: Unsupported device type");
  }
}

} // namespace tnn