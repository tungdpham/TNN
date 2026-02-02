#pragma once

#include "cpu/tensor_ops.hpp"
#include "device/task.hpp"
#ifdef USE_CUDA
#include "cuda/tensor_ops.hpp"
#endif
#include "tensor.hpp"

namespace tnn {
namespace ops {

template <typename T>
std::unique_ptr<Task> im2col(const Tensor &input_tensor, Tensor &col_data, size_t kernel_h,
                             size_t kernel_w, size_t stride_h = 1, size_t stride_w = 1,
                             size_t pad_h = 0, size_t pad_w = 0,
                             const std::string &flow_id = "default") {
  if (col_data->device_type() != input_tensor->device_type()) {
    throw std::runtime_error("im2col: Mismatched device types between col_data and input_tensor");
  }

  const auto &shape = input_tensor->shape();
  if (shape.size() != 4) {
    throw std::invalid_argument("im2col: Input tensor must be 4-dimensional (NCHW)");
  }

  const size_t batch_size = shape[0];
  const size_t channels = shape[1];
  const size_t height = shape[2];
  const size_t width = shape[3];

  const size_t padded_h = height + 2 * pad_h;
  const size_t padded_w = width + 2 * pad_w;
  const size_t output_h = (padded_h - kernel_h) / stride_h + 1;
  const size_t output_w = (padded_w - kernel_w) / stride_w + 1;

  const T *input_data = input_tensor->data_as<T>();
  T *col_data_ptr = col_data->data_as<T>();

  if (input_tensor->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, tnn::cpu::cpu_im2col<T>, input_data, col_data_ptr, batch_size,
                           channels, height, width, kernel_h, kernel_w, stride_h, stride_w, pad_h,
                           pad_w, output_h, output_w);
  }
#ifdef USE_CUDA
  else if (input_tensor->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id, tnn::cuda::cuda_im2col<T>, input_data, col_data_ptr,
                            batch_size, channels, height, width, kernel_h, kernel_w, stride_h,
                            stride_w, pad_h, pad_w, output_h, output_w);
  }
#endif
  else {
    throw std::runtime_error("im2col: Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> col2im(const Tensor &col_data, Tensor &result_data, size_t batch_size,
                             size_t channels, size_t height, size_t width, size_t kernel_h,
                             size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h,
                             size_t pad_w, const std::string &flow_id = "default") {
  if (col_data->device_type() != result_data->device_type()) {
    throw std::runtime_error("col2im: Mismatched device types between col_data and result_data");
  }

  const size_t padded_h = height + 2 * pad_h;
  const size_t padded_w = width + 2 * pad_w;
  const size_t output_h = (padded_h - kernel_h) / stride_h + 1;
  const size_t output_w = (padded_w - kernel_w) / stride_w + 1;

  const T *col_data_ptr = col_data->data_as<T>();
  T *result_data_ptr = result_data->data_as<T>();

  if (col_data->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, tnn::cpu::cpu_col2im<T>, col_data_ptr, result_data_ptr,
                           batch_size, channels, height, width, kernel_h, kernel_w, stride_h,
                           stride_w, pad_h, pad_w, output_h, output_w);
  }
#ifdef USE_CUDA
  else if (col_data->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id, tnn::cuda::cuda_col2im<T>, col_data_ptr, result_data_ptr,
                            batch_size, channels, height, width, kernel_h, kernel_w, stride_h,
                            stride_w, pad_h, pad_w, output_h, output_w);
  }
#endif
  else {
    throw std::runtime_error("col2im: Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> pad(const Tensor &input, Tensor &result, size_t pad_h, size_t pad_w,
                          T value = T(0), const std::string &flow_id = "default") {
  const auto &shape = input->shape();
  if (shape.size() != 4) {
    throw std::invalid_argument("pad: Input tensor must be 4-dimensional (NCHW)");
  }

  const size_t batch_size = shape[0];
  const size_t channels = shape[1];
  const size_t height = shape[2];
  const size_t width = shape[3];

  const T *input_data = input->data_as<T>();
  T *result_data = result->data_as<T>();

  if (input->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, tnn::cpu::cpu_pad<T>, input_data, result_data, batch_size,
                           channels, height, width, pad_h, pad_w, value);
  }
#ifdef USE_CUDA
  else if (input->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id, tnn::cuda::cuda_pad<T>, input_data, result_data, batch_size,
                            channels, height, width, pad_h, pad_w, value);
  }
#endif
  else {
    throw std::runtime_error("pad: Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> unpad(const Tensor &input, Tensor &result, size_t pad_h, size_t pad_w,
                            const std::string &flow_id = "default") {
  const auto &shape = input->shape();
  if (shape.size() != 4) {
    throw std::invalid_argument("unpad: Input tensor must be 4-dimensional (NCHW)");
  }

  const size_t batch_size = shape[0];
  const size_t channels = shape[1];
  const size_t padded_height = shape[2];
  const size_t padded_width = shape[3];

  if (padded_height <= 2 * pad_h || padded_width <= 2 * pad_w) {
    throw std::invalid_argument("Padding size too large for unpadding");
  }

  const size_t height = padded_height - 2 * pad_h;
  const size_t width = padded_width - 2 * pad_w;

  const T *input_data = input->data_as<T>();
  T *result_data = result->data_as<T>();

  if (input->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, tnn::cpu::cpu_unpad<T>, input_data, result_data, batch_size,
                           channels, height, width, pad_h, pad_w);
  }
#ifdef USE_CUDA
  else if (input->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id, tnn::cuda::cuda_unpad<T>, input_data, result_data, batch_size,
                            channels, height, width, pad_h, pad_w);
  }
#endif
  else {
    throw std::runtime_error("unpad: Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> crop(const Tensor &input, Tensor &result, const size_t start_h,
                           const size_t start_w, const size_t end_h, const size_t end_w,
                           const std::string &flow_id = "default") {
  const auto &shape = input->shape();
  if (shape.size() != 4) {
    throw std::invalid_argument("crop: Input tensor must be 4-dimensional (NCHW)");
  }

  const size_t batch_size = shape[0];
  const size_t channels = shape[1];
  const size_t height = shape[2];
  const size_t width = shape[3];

  if (end_h >= height || end_w >= width || start_h > end_h || start_w > end_w) {
    throw std::invalid_argument("Invalid crop dimensions");
  }

  const size_t new_height = end_h - start_h + 1;
  const size_t new_width = end_w - start_w + 1;

  const T *input_data = input->data_as<T>();
  T *result_data = result->data_as<T>();

  if (input->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, tnn::cpu::cpu_crop<T>, input_data, result_data, batch_size,
                           channels, height, width, start_h, start_w, new_height, new_width);
  }
#ifdef USE_CUDA
  else if (input->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id, tnn::cuda::cuda_crop<T>, input_data, result_data, batch_size,
                            channels, height, width, start_h, start_w, new_height, new_width);
  }
#endif
  else {
    throw std::runtime_error("crop: Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> slice_batch(const Tensor &input, Tensor &result, size_t start_batch,
                                  size_t end_batch, const std::string &flow_id = "default") {
  const auto &shape = input->shape();
  const size_t batch_size = shape[0];

  if (end_batch > batch_size || start_batch > end_batch) {
    throw std::invalid_argument("Invalid batch slice range");
  }

  size_t batch_stride = 1;
  for (size_t i = 1; i < shape.size(); ++i) {
    batch_stride *= shape[i];
  }

  std::vector<size_t> result_shape = shape;
  result_shape[0] = end_batch - start_batch;
  result->resize(result_shape);

  const T *input_data = input->data_as<T>();
  T *result_data = result->data_as<T>();

  const size_t copy_size = (end_batch - start_batch) * batch_stride;

  if (input->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, [=]() {
      std::copy(&input_data[start_batch * batch_stride], &input_data[end_batch * batch_stride],
                result_data);
    });
  }
#ifdef USE_CUDA
  else if (input->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_copy<T>,
                            &input_data[start_batch * batch_stride], result_data, copy_size);
  }
#endif
  else {
    throw std::runtime_error("slice_batch: Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> split(const Tensor &input, std::vector<Tensor> &results, size_t num_splits,
                            const std::string &flow_id = "default") {
  const auto &shape = input->shape();
  const size_t batch_size = shape[0];

  if (num_splits == 0 || num_splits > batch_size) {
    throw std::invalid_argument("Invalid number of splits");
  }

  results.clear();
  results.reserve(num_splits);
  const size_t split_size = batch_size / num_splits;

  for (size_t i = 0; i < num_splits; ++i) {
    size_t start = i * split_size;
    size_t end = (i == num_splits - 1) ? batch_size : start + split_size;

    // Calculate the shape for this split
    std::vector<size_t> split_shape = shape;
    split_shape[0] = end - start;

    // Create a properly initialized tensor for this split
    Tensor split_tensor = make_tensor<T>(split_shape, input->device());
    slice_batch<T>(input, split_tensor, start, end, flow_id);
    results.push_back(split_tensor);
  }

  return nullptr;
}

template <typename T>
std::unique_ptr<Task> transpose_2d(const Tensor &input, Tensor &output, size_t rows, size_t cols,
                                   const std::string &flow_id = "default") {
  if (output->device() != input->device()) {
    throw std::runtime_error("transpose_2d: Input and output must be on the same device");
  }

  const T *input_data = input->data_as<T>();
  T *output_data = output->data_as<T>();

  const auto &device = input->device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, tnn::cpu::cpu_transpose_2d<T>, input_data, output_data, rows,
                           cols);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, tnn::cuda::cuda_transpose_2d<T>, input_data, output_data, rows,
                            cols);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> nchw_to_cnhw(const Tensor &input, Tensor &output, size_t n, size_t c,
                                   size_t h, size_t w, const std::string &flow_id = "default") {
  if (output->device() != input->device()) {
    throw std::runtime_error("nchw_to_cnhw: Input and output must be on the same device");
  }

  const T *input_data = input->data_as<T>();
  T *output_data = output->data_as<T>();

  const auto &device = input->device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, tnn::cpu::cpu_nchw_to_cnhw<T>, input_data, output_data, n, c, h,
                           w);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, tnn::cuda::cuda_nchw_to_cnhw<T>, input_data, output_data, n, c,
                            h, w);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> cnhw_to_nchw(const Tensor &input, Tensor &output, size_t n, size_t c,
                                   size_t h, size_t w, const std::string &flow_id = "default") {
  if (output->device() != input->device()) {
    throw std::runtime_error("cnhw_to_nchw: Input and output must be on the same device");
  }

  const T *input_data = input->data_as<T>();
  T *output_data = output->data_as<T>();

  const auto &device = input->device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, tnn::cpu::cpu_cnhw_to_nchw<T>, input_data, output_data, n, c, h,
                           w);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, tnn::cuda::cuda_cnhw_to_nchw<T>, input_data, output_data, n, c,
                            h, w);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

}  // namespace ops
}  // namespace tnn
