/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "common/blob.hpp"
#include "device/device.hpp"
#include "device/dptr.hpp"
#include "device/flow.hpp"
#include "device/iallocator.hpp"
#include "device/pool_allocator.hpp"
#include "device/task.hpp"
#include "ops/ops.hpp"
#include "type/type.hpp"

namespace tnn {

using Tensor = std::shared_ptr<class TensorImpl>;
using ConstTensor = std::shared_ptr<const class TensorImpl>;

/**
 * @brief A TensorImpl class dedicated for ML and DL applications.
 * Data layout is assumed to be row-major (C-style) by default.
 * Generic N-dimensional TensorImpl with various functions. How layers interpret
 * the dimensions is up to the layer's implementation.
 */
class TensorImpl {
protected:
  DType_t dtype_;
  sref<IAllocator> allocator_;
  size_t data_size_;
  dptr data_;
  Vec<size_t> shape_;
  Vec<size_t> strides_;  // Precomputed strides for each dimension

  inline void compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) {
      return;
    }
    size_t stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
      strides_[i] = stride;
      stride *= shape_[i];
    }
  }

  inline size_t compute_stride(size_t index) const { return strides_[index]; }

  inline size_t compute_index(std::initializer_list<size_t> indices) const {
    assert(indices.size() == shape_.size());
    size_t index = 0;
    size_t i = 0;
    for (auto idx : indices) {
      index += idx * compute_stride(i++);
    }
    return index;
  }

  dptr allocate_data(size_t size) {
    dptr data = allocator_->allocate(size * get_dtype_size(dtype_));
    return data;
  }

public:
  // Constructors and Destructor
  TensorImpl(IAllocator &allocator, DType_t dtype)
      : dtype_(dtype),
        allocator_(allocator),
        data_size_(0) {
    for (size_t i = 0; i < shape_.size(); ++i) {
      shape_[i] = 0;
    }
    compute_strides();
    data_ = allocate_data(0);
  }

  TensorImpl(IAllocator &allocator, DType_t dtype, std::initializer_list<size_t> shape_list)
      : dtype_(dtype),
        allocator_(allocator),
        shape_(shape_list) {
    if (shape_.empty()) {
      shape_.push_back(0);
    }
    data_size_ =
        std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    compute_strides();
    data_ = allocate_data(data_size_);
  }

  TensorImpl(IAllocator &allocator, DType_t dtype, std::initializer_list<size_t> shape_list,
             const dptr &data)
      : dtype_(dtype),
        allocator_(allocator),
        shape_(shape_list) {
    if (shape_.empty()) {
      shape_.push_back(0);
    }
    data_size_ =
        std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    compute_strides();
    data_ = allocate_data(data_size_);
    if (data.get<void>() != nullptr) {
      DISPATCH_ANY_DTYPE(dtype_, T, ops::cd_copy<T>(data, data_, data_size_));
    }
  }

  TensorImpl(IAllocator &allocator, DType_t dtype, const Vec<size_t> &shape)
      : dtype_(dtype),
        allocator_(allocator),
        shape_(shape) {
    if (shape_.empty()) {
      shape_.push_back(0);
    }
    data_size_ =
        std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    compute_strides();
    data_ = allocate_data(data_size_);
  }

  TensorImpl(IAllocator &allocator, DType_t dtype, const Vec<size_t> &shape, const dptr &data)
      : dtype_(dtype),
        allocator_(allocator),
        shape_(shape) {
    if (shape_.empty()) {
      shape_.push_back(0);
    }
    data_size_ =
        std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    compute_strides();
    data_ = allocate_data(data_size_);
    if (data.get<void>() != nullptr) {
      DISPATCH_ANY_DTYPE(dtype_, T, ops::cd_copy<T>(data, data_, data_size_));
    }
  }

  TensorImpl(IAllocator &allocator, DType_t dtype, const Vec<size_t> &shape, dptr &&data)
      : dtype_(dtype),
        allocator_(allocator),
        data_(std::move(data)),
        shape_(shape) {
    if (shape_.empty()) {
      shape_.push_back(0);
    }
    data_size_ =
        std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    compute_strides();
  }

  ~TensorImpl() = default;

  TensorImpl(const TensorImpl &other)
      : dtype_(other.dtype_),
        allocator_(other.allocator_),
        data_size_(other.data_size_),
        data_(other.data_),
        shape_(other.shape_),
        strides_(other.strides_) {}

  TensorImpl(TensorImpl &&other) noexcept
      : dtype_(other.dtype_),
        allocator_(other.allocator_),
        data_size_(other.data_size_),
        data_(std::move(other.data_)),
        shape_(std::move(other.shape_)),
        strides_(std::move(other.strides_)) {
    other.data_size_ = 0;
  }

  void *data() { return data_.get<void>(); }
  const void *data() const { return data_.get<void>(); }

  template <typename T>
  T *data_as() {
    return reinterpret_cast<T *>(data_.get<void>());
  }

  template <typename T>
  const T *data_as() const {
    return reinterpret_cast<const T *>(data_.get<void>());
  }

  template <typename T>
  T &at(std::initializer_list<size_t> indices) {
    size_t index = compute_index(indices);
    return data_as<T>()[index];
  }

  template <typename T>
  const T &at(std::initializer_list<size_t> indices) const {
    size_t index = compute_index(indices);
    return data_as<T>()[index];
  }

  dptr data_ptr() { return data_; }
  const dptr data_ptr() const { return data_; }

  // Operators
  TensorImpl &operator=(const TensorImpl &other) {
    if (this != &other) {
      dtype_ = other.dtype_;
      allocator_ = other.allocator_;
      shape_ = other.shape_;
      strides_ = other.strides_;
      data_size_ = other.data_size_;
      data_ = other.data_;
    }
    return *this;
  }

  TensorImpl &operator=(TensorImpl &&other) noexcept {
    if (this != &other) {
      dtype_ = other.dtype_;
      allocator_ = other.allocator_;
      shape_ = std::move(other.shape_);
      strides_ = std::move(other.strides_);
      data_ = std::move(other.data_);
      data_size_ = other.data_size_;
      other.data_size_ = 0;
    }
    return *this;
  }

  bool same_shape(const TensorImpl &other) const { return shape_ == other.shape_; }

  const Vec<size_t> &shape() const { return shape_; }

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

  size_t dims() const { return shape_.size(); }

  size_t dimension(const size_t index) const { return shape_[index]; }

  size_t stride(const size_t index) const { return compute_stride(index); }

  size_t size() const { return data_size_; }

  size_t capacity() const { return data_.capacity() / get_dtype_size(dtype_); }

  bool is_aligned(size_t alignment = 32) const {
    return (reinterpret_cast<uintptr_t>(data_.get<void>()) % alignment) == 0;
  }

  IAllocator &allocator() const { return allocator_; }

  const Device &device() const { return data_.device(); }

  DeviceType device_type() const { return device().device_type(); }

  Tensor to_device(const Device &target_device) const {
    if (device() == target_device) {
      return clone();
    }
    auto &allocator = PoolAllocator::instance(target_device, defaultFlowHandle);
    if (device_type() == DeviceType::CPU && target_device.device_type() == DeviceType::GPU) {
      Vec<size_t> shape_vec(shape_);
      Tensor gpu_tensor = std::make_shared<TensorImpl>(allocator, dtype_, shape_vec);
      DISPATCH_ANY_DTYPE(dtype_, T, ops::cd_copy<T>(data_, gpu_tensor->data_, data_size_));
      return gpu_tensor;
    }
    if (device_type() == DeviceType::GPU && target_device.device_type() == DeviceType::CPU) {
      Vec<size_t> shape_vec(shape_);
      Tensor cpu_tensor = std::make_shared<TensorImpl>(allocator, dtype_, shape_vec);
      DISPATCH_ANY_DTYPE(dtype_, T, ops::cd_copy<T>(data_, cpu_tensor->data_, data_size_));
      return cpu_tensor;
    }
    throw std::runtime_error("Unsupported device type for to_device()");
  }

  Tensor to_dtype(DType_t target_dtype) const {
    if (dtype_ == target_dtype) {
      return clone();
    }
    auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
    Vec<size_t> shape_vec(shape_);
    Tensor converted_tensor = std::make_shared<TensorImpl>(allocator, target_dtype, shape_vec);
    DISPATCH_ANY_DTYPE2(dtype_, target_dtype, T, U,
                        ops::cast<T, U>(data_, converted_tensor->data_, data_size_));
    return converted_tensor;
  }

  Tensor to_host() const {
    if (device_type() == DeviceType::CPU) {
      return clone();
    }
    auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
    Vec<size_t> shape_vec(shape_);
    Tensor cpu_tensor = std::make_shared<TensorImpl>(allocator, dtype_, shape_vec);
    DISPATCH_ANY_DTYPE(dtype_, T, ops::cd_copy<T>(data_, cpu_tensor->data_, data_size_));
    return cpu_tensor;
  }

  Tensor clone() const { return std::make_shared<TensorImpl>(*allocator_, dtype_, shape_, data_); }

  Tensor span(Vec<size_t> start_offset, Vec<size_t> span_sizes) const {
    if (start_offset.size() != shape_.size() || span_sizes.size() != shape_.size()) {
      throw std::invalid_argument("Span offsets and sizes must match TensorImpl dimensions");
    }

    for (size_t i = 0; i < shape_.size(); ++i) {
      if (start_offset[i] + span_sizes[i] > shape_[i]) {
        throw std::out_of_range("Span exceeds TensorImpl dimensions");
      }
    }

    for (size_t i = 0; i < shape_.size(); ++i) {
      if (span_sizes[i] > 1) {
        for (size_t j = i + 1; j < shape_.size(); ++j) {
          if (span_sizes[j] != shape_[j]) {
            throw std::invalid_argument(
                "Non-contiguous span: dimension " + std::to_string(i) +
                " has span_size > 1, so all subsequent dimensions must be fully "
                "spanned (span_size == shape[j])");
          }
        }
        break;
      }
    }

    size_t offset = 0;
    size_t span_size = 1;
    for (size_t i = 0; i < shape_.size(); ++i) {
      offset += start_offset[i] * compute_stride(i);
      span_size *= span_sizes[i];
    }
    size_t dtype_size = get_dtype_size(dtype_);
    dptr span_data = data_.span(offset * dtype_size, span_size * dtype_size);
    return std::make_shared<TensorImpl>(allocator_, dtype_, span_sizes, std::move(span_data));
  }

  Vec<Tensor> split(size_t num_splits) const {
    if (num_splits == 0) {
      throw std::invalid_argument("num_splits must be greater than 0");
    }
    if (num_splits > shape_[0]) {
      throw std::invalid_argument("num_splits exceeds the size of dimension 0");
    }
    if (shape_[0] % num_splits != 0) {
      throw std::invalid_argument("Dimension 0 (size " + std::to_string(shape_[0]) +
                                  ") is not evenly divisible by num_splits (" +
                                  std::to_string(num_splits) + ")");
    }
    size_t split_size = shape_[0] / num_splits;
    Vec<size_t> split_shape = shape_;
    split_shape[0] = split_size;

    Vec<Tensor> splits;
    splits.reserve(num_splits);
    for (size_t i = 0; i < num_splits; ++i) {
      Vec<size_t> start_offset(shape_.size(), 0);
      start_offset[0] = i * split_size;
      splits.push_back(span(start_offset, split_shape));
    }
    return splits;
  }

  std::unique_ptr<Task> fill(double value, flowHandle_t handle = defaultFlowHandle) {
    std::unique_ptr<Task> result;
    DISPATCH_ANY_DTYPE(
        dtype_, T, result = ops::set_scalar<T>(data_, static_cast<T>(value), data_size_, handle));
    return result;
  }

  // Arithmetic operations returning shared_ptr
  void add(const ConstTensor &other) {
    if (!same_shape(*other)) {
      throw std::invalid_argument("TensorImpl shapes must match for addition");
    }
    if (dtype_ != other->dtype_) {
      throw std::runtime_error("DType mismatch in TensorImpl addition");
    }
    DISPATCH_ANY_DTYPE(dtype_, T, ops::add<T>(data_, other->data_, data_, data_size_));
  }

  void sub(const ConstTensor &other) {
    if (!same_shape(*other)) {
      throw std::invalid_argument("TensorImpl shapes must match for subtraction");
    }
    if (dtype_ != other->dtype_) {
      throw std::runtime_error("DType mismatch in TensorImpl subtraction");
    }
    DISPATCH_ANY_DTYPE(dtype_, T, ops::sub<T>(data_, other->data_, data_, data_size_));
  }

  void mul(const ConstTensor &other) {
    if (!same_shape(*other)) {
      throw std::invalid_argument("TensorImpl shapes must match for element-wise multiplication");
    }
    if (dtype_ != other->dtype_) {
      throw std::runtime_error("DType mismatch in TensorImpl multiplication");
    }
    DISPATCH_ANY_DTYPE(dtype_, T, ops::mul<T>(data_, other->data_, data_, data_size_));
  }

  void div(const ConstTensor &other) {
    if (!same_shape(*other)) {
      throw std::invalid_argument("TensorImpl shapes must match for element-wise division");
    }
    if (dtype_ != other->dtype_) {
      throw std::runtime_error("DType mismatch in TensorImpl division");
    }
    DISPATCH_ANY_DTYPE(dtype_, T, ops::div<T>(data_, other->data_, data_, data_size_));
  }

  void add_scalar(double scalar) {
    DISPATCH_ANY_DTYPE(dtype_, T,
                       ops::add_scalar<T>(data_, static_cast<T>(scalar), data_, data_size_));
  }

  void sub_scalar(double scalar) {
    DISPATCH_ANY_DTYPE(dtype_, T,
                       ops::sub_scalar<T>(data_, static_cast<T>(scalar), data_, data_size_));
  }

  void mul_scalar(double scalar) {
    DISPATCH_ANY_DTYPE(dtype_, T,
                       ops::mul_scalar<T>(data_, static_cast<T>(scalar), data_, data_size_));
  }

  void div_scalar(double scalar) {
    if (scalar == 0.0) {
      throw std::invalid_argument("Division by zero");
    }
    DISPATCH_ANY_DTYPE(dtype_, T,
                       ops::div_scalar<T>(data_, static_cast<T>(scalar), data_, data_size_));
  }

  void fill_random_uniform(double range) {
    unsigned long long seed = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
        reinterpret_cast<uintptr_t>(data_.get<void>()));
    DISPATCH_ANY_DTYPE(
        dtype_, T, ops::fill_random_uniform(data_, data_size_, T(0), static_cast<T>(range), seed));
  }

  void fill_random_uniform(double min_val, double max_val) {
    unsigned long long seed = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
        reinterpret_cast<uintptr_t>(data_.get<void>()));
    DISPATCH_ANY_DTYPE(dtype_, T,
                       ops::fill_random_uniform(data_, data_size_, static_cast<T>(min_val),
                                                static_cast<T>(max_val), seed));
  }

  void fill_random_uniform(double min_val, double max_val, unsigned long long seed) {
    DISPATCH_ANY_DTYPE(dtype_, T,
                       ops::fill_random_uniform(data_, data_size_, static_cast<T>(min_val),
                                                static_cast<T>(max_val), seed));
  }

  void fill_random_normal(double mean, double stddev) {
    unsigned long long seed = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
        reinterpret_cast<uintptr_t>(data_.get<void>()));
    DISPATCH_ANY_DTYPE(dtype_, T,
                       ops::fill_random_normal(data_, data_size_, static_cast<T>(mean),
                                               static_cast<T>(stddev), seed));
  }

  void fill_random_normal(double mean, double stddev, unsigned long long seed) {
    DISPATCH_ANY_DTYPE(dtype_, T,
                       ops::fill_random_normal(data_, data_size_, static_cast<T>(mean),
                                               static_cast<T>(stddev), seed));
  }

  /**
   * @brief Copy between typed tensors
   * @param target Target TensorImpl to copy data into
   */
  void copy_to(const Tensor &target) const {
    if (dtype_ != target->dtype_) {
      throw std::runtime_error("DType mismatch in TensorImpl copy");
    }
    DISPATCH_ANY_DTYPE(dtype_, T, ops::cd_copy<T>(data_, target->data_, data_size_));
  }

  // unsafe version of copy_to that allows copying between different const ness
  void share_from(const ConstTensor &source) {
    if (dtype_ != source->dtype_) {
      throw std::runtime_error("DType mismatch in TensorImpl share_from");
    }
    data_ = source->data_;
    data_size_ = source->data_size_;
    shape_ = source->shape_;
    strides_ = source->strides_;
  }

  void resize(const Vec<size_t> &new_shape) {
    if (new_shape == shape_) {
      return;
    }

    size_t new_size =
        std::accumulate(new_shape.begin(), new_shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_size != data_size_) {
      data_ = allocate_data(new_size);
      data_size_ = new_size;
    }
    shape_ = new_shape;
    compute_strides();
  }

  /**
   * Similar to resize but only reallocates if the new size is larger than
   * the current allocated size. Good for caching.
   */
  void ensure(const Vec<size_t> &new_shape) {
    size_t new_size =
        std::accumulate(new_shape.begin(), new_shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_size * get_dtype_size(dtype_) > data_.capacity()) {
      data_ = nullptr;  // free data
      data_ = allocate_data(new_size);
    }
    data_size_ = new_size;
    shape_ = new_shape;
    compute_strides();
  }

  void copy_batch(TensorImpl &other, size_t src_batch_idx, size_t dest_batch_idx) {
    size_t batch_size = shape_[0];
    if (dest_batch_idx >= batch_size || src_batch_idx >= other.shape_[0]) {
      throw std::invalid_argument("Invalid batch index for copy");
    }

    if (device() != other.device()) {
      throw std::runtime_error(
          "Cannot copy batch between tensors on different devices. Transfer "
          "to same device first.");
    }

    if (dtype_ != other.dtype_) {
      throw std::runtime_error("DType mismatch in copy_batch");
    }

    size_t batch_stride = compute_stride(0);
    size_t src_offset = src_batch_idx * other.compute_stride(0);
    size_t dest_offset = dest_batch_idx * batch_stride;
    size_t dtype_size = get_dtype_size(dtype_);

    DISPATCH_ANY_DTYPE(dtype_, T,
                       ops::copy<T>(other.data_ + src_offset * dtype_size,
                                    data_ + dest_offset * dtype_size, batch_stride));
  }

  double min() const {
    double result = 0.0;
    DISPATCH_ANY_DTYPE(dtype_, T, {
      auto cpu_tensor = to_device(getHost());
      T min_val = cpu_tensor->data_.template get<T>()[0];
      for (size_t i = 1; i < cpu_tensor->data_size_; ++i) {
        if (cpu_tensor->data_.template get<T>()[i] < min_val) {
          min_val = cpu_tensor->data_.template get<T>()[i];
        }
      }
      result = static_cast<double>(min_val);
    });
    return result;
  }

  double max() const {
    double result = 0.0;
    DISPATCH_ANY_DTYPE(dtype_, T, {
      auto cpu_tensor = to_device(getHost());
      T max_val = cpu_tensor->data_as<T>()[0];
      for (size_t i = 1; i < cpu_tensor->data_size_; ++i) {
        if (cpu_tensor->data_as<T>()[i] > max_val) {
          max_val = cpu_tensor->data_as<T>()[i];
        }
      }
      result = static_cast<double>(max_val);
    });
    return result;
  }

  double mean() const {
    double result = 0.0;
    DISPATCH_ANY_DTYPE(dtype_, T, {
      T sum = ops::sum<T>(data_, data_size_);
      result = static_cast<double>((double)sum / (double)data_size_);
    });
    return result;
  }

  double variance() const {
    double result = 0.0;
    DISPATCH_ANY_DTYPE(dtype_, T, {
      T m = static_cast<T>(mean());
      T sum_sq_diff = ops::sum_squared_diff<T>(data_, m, data_size_);
      result = static_cast<double>((double)sum_sq_diff / (double)data_size_);
    });
    return result;
  }

  void print_data(std::string name = "Tensor") const {
    Tensor cpu_tensor = to_device(getHost());
    size_t total_elements = cpu_tensor->size();
    std::cout << name << " data (shape " << cpu_tensor->shape_str() << "):\n";
    DISPATCH_ANY_DTYPE(dtype_, T, {
      T *data = cpu_tensor->data_as<T>();
      for (size_t i = 0; i < total_elements; ++i) {
        std::cout << static_cast<float>(data[i]) << " ";
      }
    });
    std::cout << std::endl;
  }

  void head(size_t n = 10, std::string name = "Tensor") const {
    Tensor cpu_tensor = to_device(getHost());
    size_t total_elements = cpu_tensor->size();
    n = std::min(n, total_elements);
    std::cout << name << " head (first " << n << " elements of shape " << cpu_tensor->shape_str()
              << "):\n";
    DISPATCH_ANY_DTYPE(dtype_, T, {
      T *data = cpu_tensor->data_as<T>();
      for (size_t i = 0; i < n; ++i) {
        std::cout << static_cast<float>(data[i]) << " ";
      }
    });
    std::cout << std::endl;
  }

  void save(std::ostream &out) const {
    if (!out) {
      throw std::runtime_error("Stream is not ready for writing");
    }

    // write dims, shape
    size_t dims = shape_.size();
    DType_t dtype = data_type();
    out.write(reinterpret_cast<const char *>(&dtype), sizeof(DType_t));
    out.write(reinterpret_cast<const char *>(&dims), sizeof(size_t));
    out.write(reinterpret_cast<const char *>(shape_.data()), shape_.size() * sizeof(size_t));

    DISPATCH_ANY_DTYPE(dtype_, T, {
      if (device_type() == DeviceType::CPU) {
        out.write(reinterpret_cast<const char *>(data_.get<T>()), data_size_ * sizeof(T));
      } else {
        Vec<T> host_buffer(data_size_);
        device().copyToHost(host_buffer.data(), data_.get<T>(), data_size_ * sizeof(T));
        out.write(reinterpret_cast<const char *>(host_buffer.data()), data_size_ * sizeof(T));
      }
    });
  }

  DType_t &data_type() { return dtype_; }
  const DType_t &data_type() const { return dtype_; }
};

template <typename Archiver>
void archive(Archiver &archive, const Tensor &tensor) {
  DType_t &dtype = tensor->data_type();
  Vec<size_t> shape_vec = tensor->shape();
  archive(dtype);
  archive(shape_vec);
  dptr data_ptr = tensor->data_ptr();
  archive(make_blob(data_ptr.get<unsigned char>(), tensor->size() * get_dtype_size(dtype),
                    tensor->device()));
}

inline Tensor operator+(const ConstTensor &lhs, const ConstTensor &rhs) {
  Tensor result = lhs->clone();
  result->add(rhs);
  return result;
}

inline Tensor operator-(const ConstTensor &lhs, const ConstTensor &rhs) {
  Tensor result = lhs->clone();
  result->sub(rhs);
  return result;
}

inline Tensor operator*(const ConstTensor &lhs, const ConstTensor &rhs) {
  Tensor result = lhs->clone();
  result->mul(rhs);
  return result;
}

inline Tensor operator/(const ConstTensor &lhs, const ConstTensor &rhs) {
  Tensor result = lhs->clone();
  result->div(rhs);
  return result;
}

inline Tensor operator+(const ConstTensor &lhs, double scalar) {
  Tensor result = lhs->clone();
  result->add_scalar(scalar);
  return result;
}

inline Tensor operator-(const ConstTensor &lhs, double scalar) {
  Tensor result = lhs->clone();
  result->sub_scalar(scalar);
  return result;
}

inline Tensor operator*(const ConstTensor &lhs, double scalar) {
  Tensor result = lhs->clone();
  result->mul_scalar(scalar);
  return result;
}

inline Tensor operator/(const ConstTensor &lhs, double scalar) {
  Tensor result = lhs->clone();
  result->div_scalar(scalar);
  return result;
}

inline Tensor operator+(double scalar, const ConstTensor &rhs) {
  Tensor result = rhs->clone();
  result->add_scalar(scalar);
  return result;
}

inline Tensor operator*(double scalar, const ConstTensor &rhs) {
  Tensor result = rhs->clone();
  result->mul_scalar(scalar);
  return result;
}

inline Tensor operator+=(Tensor &lhs, const ConstTensor &rhs) {
  lhs->add(rhs);
  return lhs;
}

inline Tensor operator-=(Tensor &lhs, const ConstTensor &rhs) {
  lhs->sub(rhs);
  return lhs;
}

inline Tensor operator*=(Tensor &lhs, const ConstTensor &rhs) {
  lhs->mul(rhs);
  return lhs;
}

inline Tensor operator/=(Tensor &lhs, const ConstTensor &rhs) {
  lhs->div(rhs);
  return lhs;
}

}  // namespace tnn

#include "tensor/tensor_factory.hpp"  // IWYU pragma: keep
