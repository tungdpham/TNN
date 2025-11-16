#pragma once

#include "cpu/tensor_ops.hpp"
#ifdef USE_CUDA
#include "cuda/tensor_ops.hpp"
#endif
#include "tensor.hpp"

namespace tnn {
// im2col/col2im operations

template <typename T>
void im2col(const Tensor<T, NCHW> &input_tensor, device_ptr<T[]> &col_data, size_t kernel_h,
            size_t kernel_w, size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0,
            size_t pad_w = 0) {
  if (input_tensor.device_type() != col_data.getDeviceType()) {
    throw std::runtime_error("im2col: Mismatched device types between input tensor and col_data");
  }
  if (input_tensor.is_on_cpu()) {
    cpu::im2col(input_tensor, col_data.get(), kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
  }
#ifdef USE_CUDA
  else if (input_tensor.is_on_gpu()) {
    cuda::im2col(input_tensor, col_data.get(), kernel_h, kernel_w, stride_h, stride_w, pad_h,
                 pad_w);
  }
#endif
  else {
    throw std::runtime_error("im2col: Unsupported device type");
  }
}

template <typename T>
void col2im(const device_ptr<T[]> &col_data, const device_ptr<T[]> &result_data, size_t batch_size,
            size_t channels, size_t height, size_t width, size_t kernel_h, size_t kernel_w,
            size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w) {
  if (col_data.getDeviceType() == DeviceType::CPU) {
    cpu::col2im(col_data.get(), result_data.get(), batch_size, channels, height, width, kernel_h,
                kernel_w, stride_h, stride_w, pad_h, pad_w);
  }
#ifdef USE_CUDA
  else if (col_data.getDeviceType() == DeviceType::GPU) {
    cuda::col2im(col_data.get(), result_data.get(), batch_size, channels, height, width, kernel_h,
                 kernel_w, stride_h, stride_w, pad_h, pad_w);
  }
#endif
  else {
    throw std::runtime_error("col2im: Unsupported device type");
  }
}

// Padding operations

template <typename T, Layout L>
Tensor<T, L> pad(const Tensor<T, L> &input, size_t pad_h, size_t pad_w, T value = T(0)) {
  if (input.is_on_cpu()) {
    return cpu::pad(input, pad_h, pad_w, value);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return cuda::pad(input, pad_h, pad_w, value);
  }
#endif
  else {
    throw std::runtime_error("pad: Unsupported device type");
  }
}

template <typename T>
Tensor<T, NCHW> pad(const Tensor<T, NCHW> &input, size_t pad_h, size_t pad_w, T value = T(0)) {
  if (input.is_on_cpu()) {
    return cpu::pad(input, pad_h, pad_w, value);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return cuda::pad(input, pad_h, pad_w, value);
  }
#endif
  else {
    throw std::runtime_error("pad: Unsupported device type");
  }
}

template <typename T, Layout L>
Tensor<T, NCHW> unpad(const Tensor<T, L> &input, size_t pad_h, size_t pad_w) {
  if (input.is_on_cpu()) {
    return cpu::unpad(input, pad_h, pad_w);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return cuda::unpad(input, pad_h, pad_w);
  }
#endif
  else {
    throw std::runtime_error("unpad: Unsupported device type");
  }
}

template <typename T>
Tensor<T, NCHW> unpad(const Tensor<T, NCHW> &input, size_t pad_h, size_t pad_w) {
  if (input.is_on_cpu()) {
    return cpu::unpad(input, pad_h, pad_w);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return cuda::unpad(input, pad_h, pad_w);
  }
#endif
  else {
    throw std::runtime_error("unpad: Unsupported device type");
  }
}

// Crop operation

template <typename T, Layout L>
Tensor<T, L> crop(const Tensor<T, L> &input, const size_t start_h, const size_t start_w,
                  const size_t end_h, const size_t end_w) {
  if (input.is_on_cpu()) {
    return cpu::crop(input, start_h, start_w, end_h, end_w);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return cuda::crop(input, start_h, start_w, end_h, end_w);
  }
#endif
  else {
    throw std::runtime_error("crop: Unsupported device type");
  }
}

template <typename T>
Tensor<T, NCHW> crop(const Tensor<T, NCHW> &input, const size_t start_h, const size_t start_w,
                     const size_t end_h, const size_t end_w) {
  if (input.is_on_cpu()) {
    return cpu::crop(input, start_h, start_w, end_h, end_w);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return cuda::crop(input, start_h, start_w, end_h, end_w);
  }
#endif
  else {
    throw std::runtime_error("crop: Unsupported device type");
  }
}

// Slicing operations

template <typename T, Layout L>
Tensor<T, L> slice_batch(const Tensor<T, L> &input, size_t start_batch, size_t end_batch) {
  if (input.is_on_cpu()) {
    return cpu::slice_batch(input, start_batch, end_batch);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return cuda::slice_batch(input, start_batch, end_batch);
  }
#endif
  else {
    throw std::runtime_error("slice_batch: Unsupported device type");
  }
}

template <typename T, Layout L>
Tensor<T, L> slice_channels(const Tensor<T, L> &input, size_t start_ch, size_t end_ch) {
  if (input.is_on_cpu()) {
    return cpu::slice_channels(input, start_ch, end_ch);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return cuda::slice_channels(input, start_ch, end_ch);
  }
#endif
  else {
    throw std::runtime_error("slice_channels: Unsupported device type");
  }
}

template <typename T>
Tensor<T, NCHW> slice_channels(const Tensor<T, NCHW> &input, size_t start_ch, size_t end_ch) {
  if (input.is_on_cpu()) {
    return cpu::slice_channels(input, start_ch, end_ch);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return cuda::slice_channels(input, start_ch, end_ch);
  }
#endif
  else {
    throw std::runtime_error("slice_channels: Unsupported device type");
  }
}

// Split operation

template <typename T, Layout L>
std::vector<Tensor<T, L>> split(const Tensor<T, L> &input, size_t num_splits) {
  if (input.is_on_cpu()) {
    return cpu::split(input, num_splits);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return cuda::split(input, num_splits);
  }
#endif
  else {
    throw std::runtime_error("split: Unsupported device type");
  }
}

template <typename T>
std::vector<Tensor<T, NCHW>> split(const Tensor<T, NCHW> &input, size_t num_splits) {
  if (input.is_on_cpu()) {
    return cpu::split(input, num_splits);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    return cuda::split(input, num_splits);
  }
#endif
  else {
    throw std::runtime_error("split: Unsupported device type");
  }
}

// Softmax operation

template <typename T, Layout L> void apply_softmax(Tensor<T, L> &input) {
  if (input.is_on_cpu()) {
    cpu::apply_softmax(input);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    cuda::apply_softmax(input);
  }
#endif
  else {
    throw std::runtime_error("apply_softmax: Unsupported device type");
  }
}

template <typename T> void apply_softmax(Tensor<T, NCHW> &input) {
  if (input.is_on_cpu()) {
    cpu::apply_softmax(input);
  }
#ifdef USE_CUDA
  else if (input.is_on_gpu()) {
    cuda::apply_softmax(input);
  }
#endif
  else {
    throw std::runtime_error("apply_softmax: Unsupported device type");
  }
}

} // namespace tnn