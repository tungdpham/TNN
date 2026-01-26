#pragma once

#include "cpu/tensor_ops.hpp"
#include "device/task.hpp"
#ifdef USE_CUDA
#include "cuda/tensor_kernels.hpp"
#include "cuda/tensor_ops.hpp"
#endif
#include "tensor.hpp"

namespace tnn {
namespace TensorOps {
// im2col/col2im operations
template <typename T>
std::unique_ptr<Task> im2col(const Tensor &input_tensor, Tensor &col_data, size_t kernel_h,
                             size_t kernel_w, size_t stride_h = 1, size_t stride_w = 1,
                             size_t pad_h = 0, size_t pad_w = 0,
                             const std::string &flow_id = "default") {
  if (col_data->device_type() != input_tensor->device_type()) {
    throw std::runtime_error("im2col: Mismatched device types between col_data and input_tensor");
  }
  if (input_tensor->device_type() != col_data->device_type()) {
    throw std::runtime_error("im2col: Mismatched device types between input tensor and col_data");
  }
  if (input_tensor->is_on_cpu()) {
    return create_cpu_task(flow_id, cpu::im2col<T>, input_tensor, col_data->data_as<T>(), kernel_h,
                           kernel_w, stride_h, stride_w, pad_h, pad_w);
  }
#ifdef USE_CUDA
  else if (input_tensor->is_on_gpu()) {
    return create_cuda_task(flow_id, cuda::im2col<T>, input_tensor, col_data->data_as<T>(),
                            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
  }
#endif
  else {
    throw std::runtime_error("im2col: Unsupported device type");
  }
} // namespace std::unique_ptr

template <typename T>
std::unique_ptr<Task> col2im(const Tensor &col_data, Tensor &result_data, size_t batch_size,
                             size_t channels, size_t height, size_t width, size_t kernel_h,
                             size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h,
                             size_t pad_w, const std::string &flow_id = "default") {
  if (col_data->device_type() != result_data->device_type()) {
    throw std::runtime_error("col2im: Mismatched device types between col_data and result_data");
  }
  if (col_data->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::col2im<T>, col_data->data_as<T>(),
                           result_data->data_as<T>(), batch_size, channels, height, width, kernel_h,
                           kernel_w, stride_h, stride_w, pad_h, pad_w);
  }
#ifdef USE_CUDA
  else if (col_data->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::col2im<T>, col_data->data_as<T>(),
                            result_data->data_as<T>(), batch_size, channels, height, width,
                            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
  }
#endif
  else {
    throw std::runtime_error("col2im: Unsupported device type");
  }
}

// Padding operations
template <typename T>
std::unique_ptr<Task> pad(const Tensor &input, Tensor &result, size_t pad_h, size_t pad_w,
                          T value = T(0), const std::string &flow_id = "default") {
  if (input->is_on_cpu()) {
    return create_cpu_task(flow_id, cpu::pad<T>, input, result, pad_h, pad_w, value);
  }
#ifdef USE_CUDA
  else if (input->is_on_gpu()) {
    return create_cuda_task(flow_id, cuda::pad<T>, input, result, pad_h, pad_w, value);
  }
#endif
  else {
    throw std::runtime_error("pad: Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> unpad(const Tensor &input, Tensor &result, size_t pad_h, size_t pad_w,
                            const std::string &flow_id = "default") {
  if (input->is_on_cpu()) {
    return create_cpu_task(flow_id, cpu::unpad<T>, input, result, pad_h, pad_w);
  }
#ifdef USE_CUDA
  else if (input->is_on_gpu()) {
    return create_cuda_task(flow_id, cuda::unpad<T>, input, result, pad_h, pad_w);
  }
#endif
  else {
    throw std::runtime_error("unpad: Unsupported device type");
  }
}

// Crop operation
template <typename T>
std::unique_ptr<Task> crop(const Tensor &input, Tensor &result, const size_t start_h,
                           const size_t start_w, const size_t end_h, const size_t end_w,
                           const std::string &flow_id = "default") {
  if (input->is_on_cpu()) {
    return create_cpu_task(flow_id, cpu::crop<T>, input, result, start_h, start_w, end_h, end_w);
  }
#ifdef USE_CUDA
  else if (input->is_on_gpu()) {
    return create_cuda_task(flow_id, cuda::crop<T>, input, result, start_h, start_w, end_h, end_w);
  }
#endif
  else {
    throw std::runtime_error("crop: Unsupported device type");
  }
}

// Slice batch operation
template <typename T>
std::unique_ptr<Task> slice_batch(const Tensor &input, Tensor &result, size_t start_batch,
                                  size_t end_batch, const std::string &flow_id = "default") {
  if (input->is_on_cpu()) {
    return create_cpu_task(flow_id, cpu::slice_batch<T>, input, result, start_batch, end_batch);
  }
#ifdef USE_CUDA
  else if (input->is_on_gpu()) {
    return create_cuda_task(flow_id, cuda::slice_batch<T>, input, result, start_batch, end_batch);
  }
#endif
  else {
    throw std::runtime_error("slice_batch: Unsupported device type");
  }
}

// Split operation
template <typename T>
std::unique_ptr<Task> split(const Tensor &input, std::vector<Tensor> &results, size_t num_splits,
                            const std::string &flow_id = "default") {
  if (input->is_on_cpu()) {
    return create_cpu_task(flow_id, cpu::split<T>, input, results, num_splits);
  }
#ifdef USE_CUDA
  else if (input->is_on_gpu()) {
    return create_cuda_task(flow_id, cuda::split<T>, input, results, num_splits);
  }
#endif
  else {
    throw std::runtime_error("split: Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> transpose_2d(const Tensor &input, Tensor &output, size_t rows, size_t cols,
                                   const std::string &flow_id = "default") {
  if (!input->device() || !output->device()) {
    throw std::runtime_error("transpose_2d: Device pointer has no associated device");
  }

  if (output->device() != input->device()) {
    throw std::runtime_error("transpose_2d: Input and output must be on the same device");
  }

  auto device = input->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::transpose_2d<T>, input->data_as<T>(), output->data_as<T>(),
                           rows, cols);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_transpose_2d<T>, input->data_as<T>(),
                            output->data_as<T>(), rows, cols);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> nchw_to_cnhw(const Tensor &input, Tensor &output, size_t n, size_t c,
                                   size_t h, size_t w, const std::string &flow_id = "default") {
  if (!input->device() || !output->device()) {
    throw std::runtime_error("nchw_to_cnhw: Device pointer has no associated device");
  }

  if (output->device() != input->device()) {
    throw std::runtime_error("nchw_to_cnhw: Input and output must be on the same device");
  }

  auto device = input->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::nchw_to_cnhw<T>, input->data_as<T>(), output->data_as<T>(),
                           n, c, h, w);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_nchw_to_cnhw<T>, input->data_as<T>(),
                            output->data_as<T>(), n, c, h, w);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> cnhw_to_nchw(const Tensor &input, Tensor &output, size_t n, size_t c,
                                   size_t h, size_t w, const std::string &flow_id = "default") {
  if (!input->device() || !output->device()) {
    throw std::runtime_error("cnhw_to_nchw: Device pointer has no associated device");
  }
  if (output->device() != input->device()) {
    throw std::runtime_error("cnhw_to_nchw: Input and output must be on the same device");
  }
  auto device = input->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::cnhw_to_nchw<T>, input->data_as<T>(), output->data_as<T>(),
                           n, c, h, w);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_cnhw_to_nchw<T>, input->data_as<T>(),
                            output->data_as<T>(), n, c, h, w);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

} // namespace TensorOps
} // namespace tnn