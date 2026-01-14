/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/device_ptr.hpp"
#include "device/task.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <malloc.h>
#include <windows.h>
#endif

#include "device/device.hpp"
#include "device/device_manager.hpp"
#include "device/device_ptr.hpp"
#include "ops/ops.hpp"

namespace tnn {

/**
 * @brief A tensor class dedicated for ML and DL applications.
 * @tparam T Data type (e.g., float, double, int)
 * For now only NCHW is supported. A lot of changes are needed to support other
 * layouts.
 */
template <typename T = float> struct Tensor {
  static_assert(std::is_arithmetic<T>::value, "Tensor type must be arithmetic");
  static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
                "Tensor type must be floating point or integral");

private:
  const Device *device_;
  size_t data_size_;
  device_ptr<T[]> data_;
  std::vector<size_t> shape_;

  inline size_t compute_stride(size_t index) const {
    size_t stride = 1;
    for (size_t i = index + 1; i < shape_.size(); ++i) {
      stride *= shape_[i];
    }
    return stride;
  }

  template <typename... Indices> inline size_t compute_index(Indices... indices) const {
    assert(sizeof...(indices) == shape_.size());

    size_t index = [this, ... indices = indices]<size_t... I>(std::index_sequence<I...>) {
      return ((static_cast<size_t>(indices) * compute_stride(I)) + ... + 0);
    }(std::make_index_sequence<sizeof...(Indices)>{});

    return index;
  }

  void allocate_data(size_t size) { data_ = make_array_ptr<T[]>(device_, size); }

public:
  // Constructors and Destructor
  Tensor(const Device *device = &getCPU()) : device_(device), data_size_(0) {
    for (size_t i = 0; i < shape_.size(); ++i) {
      shape_[i] = 0;
    }
    allocate_data(0);
  }

  Tensor(std::initializer_list<size_t> shape_list, const Device *dt = &getCPU())
      : device_(dt), shape_(shape_list) {
    data_size_ =
        std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    allocate_data(data_size_);
  }

  Tensor(std::initializer_list<size_t> shape_list, const device_ptr<T[]> &data,
         const Device *dt = &getCPU())
      : device_(dt), shape_(shape_list) {
    data_size_ =
        std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    allocate_data(data_size_);
    if (data.get() != nullptr) {
      ops::copy(data, data_, data_size_);
    }
  }

  Tensor(std::vector<size_t> shape, const Device *dt = &getCPU())
      : device_(dt), shape_(std::move(shape)) {
    data_size_ =
        std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    allocate_data(data_size_);
  }

  Tensor(std::vector<size_t> shape, const device_ptr<T[]> &data, const Device *dt = &getCPU())
      : device_(dt), shape_(std::move(shape)) {
    data_size_ =
        std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    allocate_data(data_size_);
    if (data.get() != nullptr) {
      ops::copy(data, data_, data_size_);
    }
  }

  ~Tensor() { data_.reset(); }

  Tensor(const Tensor &other)
      : device_(other.device_), data_size_(other.data_size_), shape_(other.shape_) {
    if (data_size_ > 0) {
      allocate_data(data_size_);
      ops::copy(other.data_, data_, data_size_);
    }
  }

  Tensor(Tensor &&other) noexcept
      : device_(other.device_), data_size_(other.data_size_), shape_(std::move(other.shape_)) {
    data_ = std::move(other.data_);
  }

  template <typename... Indices> T &operator()(Indices... indices) {
    assert(device_->device_type() == DeviceType::CPU &&
           "Operator() is only available for CPU tensors");
    return data_.get()[compute_index(indices...)];
  }

  template <typename... Indices> const T &operator()(Indices... indices) const {
    assert(device_->device_type() == DeviceType::CPU &&
           "Operator() is only available for CPU tensors");
    return data_.get()[compute_index(indices...)];
  }

  T *data() { return data_.get(); }
  const T *data() const { return data_.get(); }

  // Operators
  Tensor<T> &operator=(const Tensor<T> &other) {
    if (this != &other) {
      device_ = other.device_;
      data_size_ = other.data_size_;
      allocate_data(data_size_);
    }
    return *this;
  }

  Tensor<T> &operator=(Tensor<T> &&other) noexcept {
    if (this != &other) {
      device_ = other.device_;
      shape_ = std::move(other.shape_);
      data_ = std::move(other.data_);
      data_size_ = other.data_size_;

      other.data_size_ = 0;
    }
    return *this;
  }

  bool same_shape(const Tensor<T> &other) const { return shape_ == other.shape_; }

  Tensor<T> operator+(const Tensor<T> &other) const {
    if (!same_shape(other)) {
      throw std::invalid_argument("Tensor shapes must match for addition");
    }

    Tensor<T> result(shape_, device_);
    ops::add(data_, other.data_, result.data_, data_size_);
    return result;
  }

  Tensor<T> operator-(const Tensor<T> &other) const {
    if (!same_shape(other)) {
      throw std::invalid_argument("Tensor shapes must match for subtraction");
    }

    Tensor<T> result(shape_, device_);
    ops::sub(data_, other.data_, result.data_, data_size_);
    return result;
  }

  Tensor<T> operator*(const Tensor<T> &other) const {
    if (!same_shape(other)) {
      throw std::invalid_argument("Tensor shapes must match for element-wise multiplication");
    }

    Tensor<T> result(shape_, device_);
    ops::mul(data_, other.data_, result.data_, data_size_);
    return result;
  }

  Tensor<T> operator/(const Tensor<T> &other) const {
    if (!same_shape(other)) {
      throw std::invalid_argument("Tensor shapes must match for element-wise division");
    }

    Tensor<T> result(shape_, device_);
    ops::div(data_, other.data_, result.data_, data_size_);
    return result;
  }

  Tensor<T> operator+(T scalar) const {
    Tensor<T> result(shape_, device_);
    ops::add_scalar(data_, scalar, result.data_, data_size_);
    return result;
  }

  Tensor<T> operator-(T scalar) const {
    Tensor<T> result(shape_, device_);
    ops::sub_scalar(data_, scalar, result.data_, data_size_);
    return result;
  }

  Tensor<T> operator*(T scalar) const {
    Tensor<T> result(shape_, device_);
    ops::mul_scalar(data_, scalar, result.data_, data_size_);
    return result;
  }

  Tensor<T> operator/(T scalar) const {
    if (scalar == T(0)) {
      throw std::invalid_argument("Division by zero");
    }

    Tensor<T> result(shape_, device_);
    ops::div_scalar(data_, scalar, result.data_, data_size_);
    return result;
  }

  Tensor<T> &operator+=(const Tensor<T> &other) {
    if (!same_shape(other)) {
      throw std::invalid_argument("Tensor shapes must match for addition");
    }
    ops::add(data_, other.data_, data_, data_size_);
    return *this;
  }

  Tensor<T> &operator-=(const Tensor<T> &other) {
    if (!same_shape(other)) {
      throw std::invalid_argument("Tensor shapes must match for subtraction");
    }

    ops::sub(data_, other.data_, data_, data_size_);

    return *this;
  }

  Tensor<T> &operator*=(const Tensor<T> &other) {
    if (!same_shape(other)) {
      throw std::invalid_argument("Tensor shapes must match for element-wise multiplication");
    }

    ops::mul(data_, other.data_, data_, data_size_);

    return *this;
  }

  Tensor<T> &operator+=(T scalar) {
    ops::add_scalar(data_, scalar, data_, data_size_);
    return *this;
  }

  Tensor<T> &operator-=(T scalar) {
    ops::sub_scalar(data_, scalar, data_, data_size_);
    return *this;
  }

  Tensor<T> &operator*=(T scalar) {
    ops::mul_scalar(data_, scalar, data_, data_size_);
    return *this;
  }

  Tensor<T> &operator/=(T scalar) {
    if (scalar == T(0)) {
      throw std::invalid_argument("Division by zero");
    }
    ops::div_scalar(data_, scalar, data_, data_size_);
    return *this;
  }

  const std::vector<size_t> &shape() const { return shape_; }

  std::string shape_str() const {
    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0; i < shape_.size(); ++i) {
      oss << shape_[i];
      if (i < shape_.size() - 1) {
        oss << ", ";
      }
    }
    oss << "}";
    return oss.str();
  }

  const size_t dims() const { return shape_.size(); }

  const size_t dimension(const size_t index) const { return shape_[index]; }

  const size_t stride(const size_t index) const { return compute_stride(index); }

  const size_t size() const { return data_size_; }

  const size_t capacity() const { return data_.capacity(); }

  bool is_aligned(size_t alignment = 32) const {
    return (reinterpret_cast<uintptr_t>(data_.get()) % alignment) == 0;
  }

  const Device *device() const { return device_; }

  DeviceType device_type() const { return device_->device_type(); }

  bool is_on_cpu() const { return device_->device_type() == DeviceType::CPU; }

  bool is_on_gpu() const { return device_->device_type() == DeviceType::GPU; }

  const device_ptr<T[]> &data_ptr() const { return data_; }

  device_ptr<T[]> &data_ptr() { return data_; }

  Tensor<T> to_cpu() const {
    if (device_type() == DeviceType::CPU) {
      return clone();
    }

    if (device_type() == DeviceType::GPU) {
      std::vector<size_t> shape_vec(shape_);
      Tensor<T> cpu_tensor(shape_vec, &getCPU());
      // Copy from GPU to CPU
      data_.getDevice()->copyToHost(cpu_tensor.data_.get(), data_.get(), data_size_ * sizeof(T));
      return cpu_tensor;
    }
    throw std::runtime_error("Unsupported device type for to_cpu()");
  }

  Tensor<T> to_gpu(int gpu_id = 0) const {
    if (device_type() == DeviceType::GPU) {
      return clone();
    }

    if (device_type() == DeviceType::CPU) {
      std::vector<size_t> shape_vec(shape_);
      Tensor<T> gpu_tensor(shape_vec, &getGPU(gpu_id));

      // Copy from CPU to GPU
      getGPU(gpu_id).copyToDevice(gpu_tensor.data_.get(), data_.get(), data_size_ * sizeof(T));
      return gpu_tensor;
    }

    throw std::runtime_error("Unsupported device type for to_gpu()");
  }

  Tensor<T> to_device(const Device *target_device) const {
    if (device_ == target_device) {
      return clone();
    }

    if (device_type() == DeviceType::CPU && target_device->device_type() == DeviceType::GPU) {
      return to_gpu(target_device->getID());
    }

    if (device_type() == DeviceType::GPU && target_device->device_type() == DeviceType::CPU) {
      return to_cpu();
    }

    throw std::runtime_error("Unsupported device type for to_device()");
  }

  Tensor<T> clone() const { return Tensor<T>(shape_, data_, device_); }

  std::unique_ptr<Task> fill(T value) { return ops::set_scalar(data_, value, data_size_); }

  void fill_random_uniform(T range) {
    // Generate seed from current time and pointer address for uniqueness
    unsigned long long seed = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
        reinterpret_cast<uintptr_t>(data_.get()));
    ops::fill_random_uniform(data_, data_size_, T(0), range, seed);
  }

  void fill_random_uniform(T min_val, T max_val) {
    // Generate seed from current time and pointer address for uniqueness
    unsigned long long seed = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
        reinterpret_cast<uintptr_t>(data_.get()));
    ops::fill_random_uniform(data_, data_size_, min_val, max_val, seed);
  }

  void fill_random_uniform(T min_val, T max_val, unsigned long long seed) {
    ops::fill_random_uniform(data_, data_size_, min_val, max_val, seed);
  }

  void fill_random_normal(T mean, T stddev) {
    // Generate seed from current time and pointer address for uniqueness
    unsigned long long seed = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
        reinterpret_cast<uintptr_t>(data_.get()));
    ops::fill_random_normal(data_, data_size_, mean, stddev, seed);
  }

  void fill_random_normal(T mean, T stddev, unsigned long long seed) {
    ops::fill_random_normal(data_, data_size_, mean, stddev, seed);
  }

  void resize(const std::vector<size_t> &new_shape, const Device *new_device = nullptr) {
    if (new_device != nullptr && new_device != device_) {
      // Change device
      device_ = new_device;
      data_.reset();
      shape_ = new_shape;
      data_size_ =
          std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
      allocate_data(data_size_);
      return;
    }
    if (new_shape == shape_) {
      return;
    }

    size_t new_size =
        std::accumulate(new_shape.begin(), new_shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_size != data_size_) {
      data_.reset();
      allocate_data(new_size);
      data_size_ = new_size;
    }
    shape_ = new_shape;
  }

  /**
   * Similar to resize but only reallocates if the new size is larger than
   * the current allocated size. Good for caching.
   */
  void ensure(const std::vector<size_t> &new_shape, const Device *new_device = nullptr) {
    if (new_device != nullptr && new_device != device_) {
      device_ = new_device;
      shape_ = new_shape;
      data_size_ =
          std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
      allocate_data(data_size_);
      return;
    }
    size_t new_size =
        std::accumulate(new_shape.begin(), new_shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_size > data_.capacity()) {
      allocate_data(new_size);
    }
    data_size_ = new_size;
    shape_ = new_shape;
  }

  void copy_batch(Tensor<T> &other, size_t src_batch_idx, size_t dest_batch_idx) {
    size_t batch_size = shape_[0];
    if (dest_batch_idx >= batch_size || src_batch_idx >= other.shape_[0]) {
      throw std::invalid_argument("Invalid batch index for copy");
    }

    if (device_ != other.device_) {
      throw std::runtime_error("Cannot copy batch between tensors on different devices. Transfer "
                               "to same device first.");
    }

    size_t batch_stride = compute_stride(0);
    size_t src_offset = src_batch_idx * other.compute_stride(0);
    size_t dest_offset = dest_batch_idx * batch_stride;

    ops::copy(other.data_, data_, batch_stride, src_offset, dest_offset);
  }

  T min() const {
    Tensor<T> cpu_tensor = to_cpu();
    T min_value = cpu_tensor.data_.get()[0];
    for (size_t i = 1; i < cpu_tensor.data_size_; ++i) {
      if (cpu_tensor.data_.get()[i] < min_value) {
        min_value = cpu_tensor.data_.get()[i];
      }
    }
    return min_value;
  }

  T max() const {
    Tensor<T> cpu_tensor = to_cpu();
    T max_value = cpu_tensor.data_.get()[0];
    for (size_t i = 1; i < cpu_tensor.data_size_; ++i) {
      if (cpu_tensor.data_.get()[i] > max_value) {
        max_value = cpu_tensor.data_.get()[i];
      }
    }
    return max_value;
  }

  T mean() const {
    T sum = ops::sum(data_, data_size_);
    return sum / static_cast<T>(data_size_);
  }

  T variance() const {
    T m = mean();
    T sum_sq_diff = ops::sum_squared_diff(data_, m, data_size_);
    return sum_sq_diff / static_cast<T>(data_size_);
  }

  void print_data() const {
    Tensor<T> cpu_tensor = to_cpu();
    size_t total_elements = cpu_tensor.size();
    std::cout << "Tensor data (shape " << cpu_tensor.shape_str() << "):\n";
    T *data = cpu_tensor.data_.get();
    for (size_t i = 0; i < total_elements; ++i) {
      std::cout << std::setprecision(3) << data[i] << " ";
    }
    std::cout << std::endl;
  }

  void head(size_t n = 10) const {
    Tensor<T> cpu_tensor = to_cpu();
    size_t total_elements = cpu_tensor.size();
    n = std::min(n, total_elements);
    std::cout << "Tensor head (first " << n << " elements of shape " << cpu_tensor.shape_str()
              << "):\n";
    for (size_t i = 0; i < n; ++i) {
      std::cout << std::setprecision(3) << cpu_tensor.data_.get()[i] << " ";
    }
    std::cout << std::endl;
  }

  void save(std::ofstream &out) const {
    if (!out.is_open()) {
      throw std::runtime_error("File is not open for writing");
    }

    // write dims, shape
    size_t dims = shape_.size();
    out.write(reinterpret_cast<const char *>(&dims), sizeof(size_t));
    out.write(reinterpret_cast<const char *>(shape_.data()), shape_.size() * sizeof(size_t));

    if (device_type() == DeviceType::CPU) {
      out.write(reinterpret_cast<const char *>(data_.get()), data_size_ * sizeof(T));
    } else {
      // GPU case: copy to host buffer first then write
      std::vector<T> host_buffer(data_size_);
      device_->copyToHost(host_buffer.data(), data_.get(), data_size_ * sizeof(T));
      out.write(reinterpret_cast<const char *>(host_buffer.data()), data_size_ * sizeof(T));
    }
  }

  static Tensor<T> load(std::ifstream &in, const Device *device = &getCPU()) {
    if (!in.is_open()) {
      throw std::runtime_error("File is not open for reading");
    }
    // read dims, shape, and data
    size_t dims;
    in.read(reinterpret_cast<char *>(&dims), sizeof(size_t));
    std::vector<size_t> shape(dims);
    in.read(reinterpret_cast<char *>(shape.data()), dims * sizeof(size_t));
    if (in.gcount() != static_cast<std::streamsize>(dims * sizeof(size_t))) {
      throw std::runtime_error("Failed to read tensor shape from file");
    }

    Tensor<T> tensor(shape, device);
    if (device->device_type() == DeviceType::CPU) {
      in.read(reinterpret_cast<char *>(tensor.data_.get()), tensor.size() * sizeof(T));
      if (in.gcount() != static_cast<std::streamsize>(tensor.size() * sizeof(T))) {
        throw std::runtime_error("Failed to read tensor data from file");
      }
    } else {
      // GPU case: read into host buffer then copy to device
      std::vector<T> host_buffer(tensor.size());
      in.read(reinterpret_cast<char *>(host_buffer.data()), tensor.size() * sizeof(T));
      if (in.gcount() != static_cast<std::streamsize>(tensor.size() * sizeof(T))) {
        throw std::runtime_error("Failed to read tensor data from file");
      }
      device->copyToDevice(tensor.data_.get(), host_buffer.data(), tensor.size() * sizeof(T));
    }
    return tensor;
  }
};

template <typename T>
void check_nan_and_inf(const T *data, size_t size, const std::string &tensor_name = "") {
  for (size_t i = 0; i < size; ++i) {
    if (std::isnan(data[i]) || std::isinf(data[i])) {
      std::cerr << "Tensor " << tensor_name << " contains NaN or Inf at index " << i << std::endl;
      return;
    }
  }
}

template <typename T>
void check_nan_and_inf(const Tensor<T> &tensor, const std::string &tensor_name = "") {
  Tensor<T> cpu_tensor = tensor.to_cpu();
  size_t total_elements = cpu_tensor.size();
  T *data = cpu_tensor.data_ptr().get();
  check_nan_and_inf(data, total_elements, tensor_name);
}

} // namespace tnn