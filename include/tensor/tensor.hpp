/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/device_ptr.hpp"
#include "device/task.hpp"
#include "layout_trait.hpp"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
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

enum ALIGNMENT_TYPE { MKL = 64, AVX2 = 32, DEFAULT = 16 };

/**
 * @brief A tensor class dedicated for ML and DL applications.
 * @tparam T Data type (e.g., float, double, int)
 * @tparam L Memory layout (NCHW, NHWC, NCDHW, NDHWC)
 * For now only NCHW is supported. A lot of changes are needed to support other
 * layouts.
 */
template <typename T = float, Layout L = NCHW> struct Tensor {
  static_assert(std::is_arithmetic<T>::value, "Tensor type must be arithmetic");
  static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
                "Tensor type must be floating point or integral");

private:
  LayoutTrait<L> layout_trait_;
  const Device *device_;
  device_ptr<T[]> data_;
  static constexpr size_t dims_ = LayoutTrait<L>::dims;
  size_t (&shape_)[LayoutTrait<L>::dims] = layout_trait_.shape;
  size_t (&strides_)[LayoutTrait<L>::dims] = layout_trait_.strides;

  size_t data_size_;

  template <typename... Indices> inline size_t compute_index(Indices... indices) const {
    static_assert(sizeof...(indices) == dims_, "Incorrect number of dimensions");
    size_t index = 0;
    short count = 0;
    ((index += indices * strides_[count++]), ...);
    return index;
  }

  void allocate_data(size_t size) { data_ = make_array_ptr<T[]>(device_, size); }

public:
  // Constructors and Destructor
  Tensor(const Device *device = &getCPU()) : device_(device), data_size_(0) {
    for (size_t i = 0; i < dims_; ++i) {
      shape_[i] = 0;
      strides_[i] = 0;
    }
    allocate_data(0);
  }

  Tensor(std::initializer_list<size_t> shape_list, const Device *dt = &getCPU()) : device_(dt) {
    assert(shape_list.size() == dims_ && "Initializer list size must match tensor dimensions");
    std::copy(shape_list.begin(), shape_list.end(), shape_);
    layout_trait_.compute_strides();
    data_size_ = std::accumulate(shape_, shape_ + dims_, size_t(1), std::multiplies<size_t>());
    allocate_data(data_size_);
  }

  Tensor(std::initializer_list<size_t> shape_list, const device_ptr<T[]> &data,
         const Device *dt = &getCPU())
      : device_(dt) {
    assert(shape_list.size() == dims_ && "Initializer list size must match dimensions");
    std::copy(shape_list.begin(), shape_list.end(), shape_);
    layout_trait_.compute_strides();
    data_size_ = std::accumulate(shape_, shape_ + dims_, size_t(1), std::multiplies<size_t>());
    allocate_data(data_size_);
    if (data.get() != nullptr) {
      ops::copy(data, data_, data_size_)->sync();
    }
  }

  Tensor(std::vector<size_t> shape, const Device *dt = &getCPU()) : device_(dt) {
    assert(shape.size() == dims_ && "Shape vector size must match tensor dimensions");
    std::copy(shape.begin(), shape.end(), shape_);
    layout_trait_.compute_strides();
    data_size_ = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    allocate_data(data_size_);
  }

  Tensor(std::vector<size_t> shape, const device_ptr<T[]> &data, const Device *dt = &getCPU())
      : device_(dt) {
    assert(shape.size() == dims_ && "Shape vector size must match dimensions");
    std::copy(shape.begin(), shape.end(), shape_);
    layout_trait_.compute_strides();
    data_size_ = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    allocate_data(data_size_);
    if (data.get() != nullptr) {
      ops::copy(data, data_, data_size_)->sync();
    }
  }

  ~Tensor() { data_.reset(); }

  Tensor(const Tensor &other)
      : layout_trait_(other.layout_trait_), device_(other.device_), data_size_(other.data_size_) {
    if (data_size_ > 0) {
      allocate_data(data_size_);
      ops::copy(other.data_, data_, data_size_)->sync();
    }
  }
  // Tensor(const Tensor &other) = delete;

  Tensor(Tensor &&other) noexcept : device_(other.device_), data_size_(other.data_size_) {
    layout_trait_ = other.layout_trait_;
    data_ = std::move(other.data_);
  }

  template <typename... Indices> T &operator()(Indices... indices) {
    static_assert(sizeof...(indices) == dims_, "Incorrect number of dimensions");
    assert(device_->getDeviceType() == DeviceType::CPU &&
           "Operator() is only available for CPU tensors");
    return data_.get()[compute_index(indices...)];
  }

  template <typename... Indices> const T &operator()(Indices... indices) const {
    static_assert(sizeof...(indices) == dims_, "Incorrect number of dimensions");
    assert(device_->getDeviceType() == DeviceType::CPU &&
           "Operator() is only available for CPU tensors");
    return data_.get()[compute_index(indices...)];
  }

  T *data() { return data_.get(); }
  const T *data() const { return data_.get(); }

  // Operators
  Tensor<T, L> &operator=(const Tensor<T, L> &other) = delete;

  Tensor<T, L> &operator=(Tensor<T, L> &&other) noexcept {
    if (this != &other) {
      data_.reset();

      device_ = other.device_;
      layout_trait_ = other.layout_trait_;
      data_ = std::move(other.data_);
      data_size_ = other.data_size_;

      other.data_size_ = 0;
    }
    return *this;
  }

  bool same_shape(const Tensor<T, L> &other) const {
    for (size_t i = 0; i < dims_; ++i) {
      if (shape_[i] != other.shape_[i]) {
        return false;
      }
    }
    return true;
  }

  Tensor<T, L> operator+(const Tensor<T, L> &other) const {
    if (!same_shape(other)) {
      throw std::invalid_argument("Tensor shapes must match for addition");
    }

    std::vector<size_t> shape_vec(shape_, shape_ + dims_);
    Tensor<T, L> result(shape_vec, device_);

    ops::add(data_, other.data_, result.data_, data_size_)->sync();

    return result;
  }

  Tensor<T, L> operator-(const Tensor<T, L> &other) const {
    if (!same_shape(other)) {
      throw std::invalid_argument("Tensor shapes must match for subtraction");
    }

    std::vector<size_t> shape_vec(shape_, shape_ + dims_);
    Tensor<T, L> result(shape_vec, device_);

    ops::sub(data_, other.data_, result.data_, data_size_)->sync();

    return result;
  }

  Tensor<T, L> operator*(const Tensor<T, L> &other) const {
    if (!same_shape(other)) {
      throw std::invalid_argument("Tensor shapes must match for element-wise multiplication");
    }

    std::vector<size_t> shape_vec(shape_, shape_ + dims_);
    Tensor<T, L> result(shape_vec, device_);

    ops::mul(data_, other.data_, result.data_, data_size_)->sync();

    return result;
  }

  Tensor<T, L> operator/(const Tensor<T, L> &other) const {
    if (!same_shape(other)) {
      throw std::invalid_argument("Tensor shapes must match for element-wise division");
    }

    std::vector<size_t> shape_vec(shape_, shape_ + dims_);
    Tensor<T, L> result(shape_vec, device_);

    ops::div(data_, other.data_, result.data_, data_size_)->sync();

    return result;
  }

  Tensor<T, L> operator+(T scalar) const {
    std::vector<size_t> shape_vec(shape_, shape_ + dims_);
    Tensor<T, L> result(shape_vec, device_);

    ops::add_scalar(data_, scalar, result.data_, data_size_)->sync();

    return result;
  }

  Tensor<T, L> operator-(T scalar) const {
    std::vector<size_t> shape_vec(shape_, shape_ + dims_);
    Tensor<T, L> result(shape_vec, device_);

    ops::sub_scalar(data_, scalar, result.data_, data_size_)->sync();

    return result;
  }

  Tensor<T, L> operator*(T scalar) const {
    std::vector<size_t> shape_vec(shape_, shape_ + dims_);
    Tensor<T, L> result(shape_vec, device_);

    ops::mul_scalar(data_, scalar, result.data_, data_size_)->sync();

    return result;
  }

  Tensor<T, L> operator/(T scalar) const {
    if (scalar == T(0)) {
      throw std::invalid_argument("Division by zero");
    }

    std::vector<size_t> shape_vec(shape_, shape_ + dims_);
    Tensor<T, L> result(shape_vec, device_);

    ops::div_scalar(data_, scalar, result.data_, data_size_)->sync();

    return result;
  }

  Tensor<T, L> &operator+=(const Tensor<T, L> &other) {
    if (!same_shape(other)) {
      throw std::invalid_argument("Tensor shapes must match for addition");
    }

    ops::add(data_, other.data_, data_, data_size_)->sync();

    return *this;
  }

  Tensor<T, L> &operator-=(const Tensor<T, L> &other) {
    if (!same_shape(other)) {
      throw std::invalid_argument("Tensor shapes must match for subtraction");
    }

    ops::sub(data_, other.data_, data_, data_size_)->sync();

    return *this;
  }

  Tensor<T, L> &operator*=(const Tensor<T, L> &other) {
    if (!same_shape(other)) {
      throw std::invalid_argument("Tensor shapes must match for element-wise multiplication");
    }

    ops::mul(data_, other.data_, data_, data_size_)->sync();

    return *this;
  }

  Tensor<T, L> &operator+=(T scalar) {
    ops::add_scalar(data_, scalar, data_, data_size_)->sync();
    return *this;
  }

  Tensor<T, L> &operator-=(T scalar) {
    ops::sub_scalar(data_, scalar, data_, data_size_)->sync();
    return *this;
  }

  Tensor<T, L> &operator*=(T scalar) {
    ops::mul_scalar(data_, scalar, data_, data_size_)->sync();
    return *this;
  }

  Tensor<T, L> &operator/=(T scalar) {
    if (scalar == T(0)) {
      throw std::invalid_argument("Division by zero");
    }
    ops::div_scalar(data_, scalar, data_, data_size_)->sync();
    return *this;
  }

  std::vector<size_t> shape() const { return std::vector<size_t>(shape_, shape_ + dims_); }

  std::vector<size_t> strides() const { return std::vector<size_t>(strides_, strides_ + dims_); }

  std::string shape_str() const {
    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0; i < dims_; ++i) {
      oss << shape_[i];
      if (i < dims_ - 1) {
        oss << ", ";
      }
    }
    oss << "}";
    return oss.str();
  }

  const size_t batch_size() const { return layout_trait_.batch_size(); }

  const size_t channels() const { return layout_trait_.channels(); }

  const size_t height() const { return layout_trait_.height(); }

  const size_t width() const { return layout_trait_.width(); }

  const size_t depth() const {
    if constexpr (dims_ == 5) {
      return layout_trait_.depth();
    } else {
      return 1;
    }
  }

  const size_t dimension(const size_t index) const { return shape_[index]; }

  const size_t stride(const size_t index) const { return strides_[index]; }

  const size_t size() const { return data_size_; }

  bool is_aligned(size_t alignment = 32) const {
    return (reinterpret_cast<uintptr_t>(data_.get()) % alignment) == 0;
  }

  const Device *device() const { return device_; }

  DeviceType device_type() const { return device_->getDeviceType(); }

  bool is_on_cpu() const { return device_->getDeviceType() == DeviceType::CPU; }

  bool is_on_gpu() const { return device_->getDeviceType() == DeviceType::GPU; }

  const device_ptr<T[]> &data_ptr() const { return data_; }

  device_ptr<T[]> &data_ptr() { return data_; }

  Tensor<T, L> to_cpu() const {
    if (device_type() == DeviceType::CPU) {
      return clone();
    }

    if (device_type() == DeviceType::GPU) {
      std::vector<size_t> shape_vec(shape_, shape_ + dims_);
      Tensor<T, L> cpu_tensor(shape_vec, &getCPU());
      // Copy from GPU to CPU
      data_.getDevice()->copyToHost(cpu_tensor.data_.get(), data_.get(), data_size_ * sizeof(T));
      return cpu_tensor;
    }
    throw std::runtime_error("Unsupported device type for to_cpu()");
  }

  Tensor<T, L> to_gpu(int gpu_id = 0) const {
    if (device_type() == DeviceType::GPU) {
      return clone();
    }

    if (device_type() == DeviceType::CPU) {
      std::vector<size_t> shape_vec(shape_, shape_ + dims_);
      Tensor<T, L> gpu_tensor(shape_vec, &getGPU(gpu_id));

      // Copy from CPU to GPU
      getGPU(gpu_id).copyToDevice(gpu_tensor.data_.get(), data_.get(), data_size_ * sizeof(T));
      return gpu_tensor;
    }

    throw std::runtime_error("Unsupported device type for to_gpu()");
  }

  Tensor<T, L> to_device(const Device *target_device) const {
    if (device_ == target_device) {
      return clone();
    }

    if (device_type() == DeviceType::CPU && target_device->getDeviceType() == DeviceType::GPU) {
      return to_gpu(target_device->getID());
    }

    if (device_type() == DeviceType::GPU && target_device->getDeviceType() == DeviceType::CPU) {
      return to_cpu();
    }

    throw std::runtime_error("Unsupported device type for to_device()");
  }

  Tensor<T, L> clone() const {
    return Tensor<T, L>(std::vector<size_t>(shape_, shape_ + dims_), data_, device_);
  }

  std::unique_ptr<Task> fill(T value) { return ops::set_scalar(data_, value, data_size_); }

  void fill_random_uniform(T range) {
    // Generate seed from current time and pointer address for uniqueness
    unsigned long long seed = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
        reinterpret_cast<uintptr_t>(data_.get()));
    ops::fill_random_uniform(data_, data_size_, T(0), range, seed)->sync();
  }

  void fill_random_normal(T mean, T stddev) {
    // Generate seed from current time and pointer address for uniqueness
    unsigned long long seed = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
        reinterpret_cast<uintptr_t>(data_.get()));
    ops::fill_random_normal(data_, data_size_, mean, stddev, seed)->sync();
  }

  void resize(const std::vector<size_t> &new_shape) {
    assert(new_shape.size() == dims_ && "New shape size must match tensor dims size");
    if (new_shape == std::vector<size_t>(shape_, shape_ + dims_)) {
      return;
    }
    size_t new_size =
        std::accumulate(new_shape.begin(), new_shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_size != data_size_) {
      data_.reset();
      allocate_data(new_size);
      data_size_ = new_size;
    }
    std::copy(new_shape.begin(), new_shape.end(), shape_);
    layout_trait_.compute_strides();
  }

  Tensor<T, L> reshape(const std::vector<size_t> &new_shape) const {
    size_t new_size =
        std::accumulate(new_shape.begin(), new_shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_size != size()) {
      throw std::invalid_argument("New shape must have same total size");
    }
    return Tensor<T, L>(new_shape, data_, device_);
  }

  void copy_batch(Tensor<T, L> &other, size_t src_batch_idx, size_t dest_batch_idx) {
    if (dest_batch_idx >= batch_size() || src_batch_idx >= other.batch_size()) {
      throw std::invalid_argument("Invalid batch index for copy");
    }

    if (device_ != other.device_) {
      throw std::runtime_error(
          "Cannot copy batch between tensors on different devices. Transfer to same device first.");
    }

    size_t batch_stride = strides_[0];
    size_t src_offset = src_batch_idx * other.strides_[0];
    size_t dest_offset = dest_batch_idx * batch_stride;

    ops::copy(other.data_, data_, batch_stride, src_offset, dest_offset)->sync();
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
    Tensor<T, L> cpu_tensor = to_cpu();
    size_t total_elements = cpu_tensor.size();
    std::cout << "Tensor data (shape " << cpu_tensor.shape_str() << "):\n";
    for (size_t i = 0; i < total_elements; ++i) {
      std::cout << cpu_tensor.data_.get()[i] << " ";
    }
    std::cout << std::endl;
  }

  void save(std::ofstream &out) const {
    if (!out.is_open()) {
      throw std::runtime_error("File is not open for writing");
    }

    if (device_type() != DeviceType::CPU) {
      throw std::runtime_error("Tensor must be on CPU to save to file");
    }

    out.write(reinterpret_cast<const char *>(shape_), dims_ * sizeof(size_t));

    out.write(reinterpret_cast<const char *>(data_.get()), data_size_ * sizeof(T));
  }

  static Tensor<T, L> load(std::ifstream &in) {
    if (!in.is_open()) {
      throw std::runtime_error("File is not open for reading");
    }
    std::vector<size_t> shape(dims_);
    in.read(reinterpret_cast<char *>(shape.data()), dims_ * sizeof(size_t));
    if (in.gcount() != dims_ * sizeof(size_t)) {
      throw std::runtime_error("Failed to read tensor shape from file");
    }

    Tensor<T, L> tensor(shape);
    in.read(reinterpret_cast<char *>(tensor.data_.get()), tensor.size() * sizeof(T));
    if (in.gcount() != static_cast<std::streamsize>(tensor.size() * sizeof(T))) {
      throw std::runtime_error("Failed to read tensor data from file");
    }
    return tensor;
  }
};

} // namespace tnn