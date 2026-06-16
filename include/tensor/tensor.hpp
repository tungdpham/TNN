/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cassert>
#include <chrono>
#include <numeric>

#include "device/device.hpp"
#include "device/device_allocator.hpp"
#include "device/dptr.hpp"
#include "device/iallocator.hpp"
#include "device/task.hpp"
#include "ops/ops.hpp"
#include "type/type.hpp"

namespace synet {

/**
 * @brief A Tensor class dedicated for ML and DL applications.
 * Data layout is assumed to be row-major (C-style) by default.
 * Generic N-dimensional Tensor with various functions. How layers interpret
 * the dimensions is up to the layer's implementation.
 */
class Tensor {
protected:
  DType_t dtype_;
  sref<IAllocator> allocator_;
  size_t data_size_ = 0;
  dptr data_;
  Vec<size_t> shape_;
  Vec<size_t> strides_;

  void compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;
    size_t stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
      strides_[i] = stride;
      stride *= shape_[i];
    }
  }

  size_t compute_index(std::initializer_list<size_t> indices) const {
    assert(indices.size() == shape_.size());
    size_t index = 0;
    size_t i = 0;
    for (auto idx : indices) {
      index += idx * strides_[i++];
    }
    return index;
  }

public:
  Tensor(DType_t dtype = DType_t::FP32,
         IAllocator &allocator = DeviceAllocator::instance(getHost()))
      : dtype_(dtype),
        allocator_(allocator) {
    compute_strides();
  }

  Tensor(const Vec<size_t> &shape, DType_t dtype = DType_t::FP32,
         IAllocator &allocator = DeviceAllocator::instance(getHost()))
      : dtype_(dtype),
        allocator_(allocator),
        shape_(shape) {
    if (shape_.empty()) shape_.push_back(0);
    data_size_ =
        std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    compute_strides();
    data_ = allocator_->allocate(data_size_ * get_dtype_size(dtype_));
  }

  Tensor(const Vec<size_t> &shape, dptr data, DType_t dtype = DType_t::FP32,
         IAllocator &allocator = DeviceAllocator::instance(getHost()))
      : dtype_(dtype),
        allocator_(allocator),
        data_(std::move(data)),
        shape_(shape) {
    if (shape_.empty()) shape_.push_back(0);
    data_size_ =
        std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    compute_strides();
  }

  Tensor(const Vec<size_t> &shape, DType_t dtype, const Device &device)
      : Tensor(shape, dtype, DeviceAllocator::instance(device)) {}

  ~Tensor() = default;
  Tensor(const Tensor &) = default;
  Tensor(Tensor &&) noexcept = default;
  Tensor &operator=(const Tensor &) = default;
  Tensor &operator=(Tensor &&) noexcept = default;

  explicit operator bool() const { return data_.get<void>() != nullptr; }
  void *data() { return data_.get<void>(); }
  const void *data() const { return data_.get<void>(); }

  template <typename T>
  T *data_as() {
    return reinterpret_cast<T *>(data());
  }
  template <typename T>
  const T *data_as() const {
    return reinterpret_cast<const T *>(data());
  }

  template <typename T>
  T &at(std::initializer_list<size_t> indices) {
    return data_as<T>()[compute_index(indices)];
  }
  template <typename T>
  const T &at(std::initializer_list<size_t> indices) const {
    return data_as<T>()[compute_index(indices)];
  }

  dptr data_ptr() { return data_; }
  const dptr data_ptr() const { return data_; }
  const Vec<size_t> &shape() const { return shape_; }
  size_t dims() const { return shape_.size(); }
  size_t dimension(size_t index) const { return shape_[index]; }
  size_t stride(size_t index) const { return strides_[index]; }
  size_t size() const { return data_size_; }
  size_t capacity() const { return data_.capacity() / get_dtype_size(dtype_); }
  IAllocator &allocator() const { return allocator_; }
  const Device &device() const { return data_.device(); }
  DeviceType device_type() const { return device().device_type(); }
  DType_t &data_type() { return dtype_; }
  const DType_t &data_type() const { return dtype_; }

  Tensor span(Vec<size_t> start_offset, Vec<size_t> span_sizes) const {
    size_t offset = 0, span_size = 1;
    for (size_t i = 0; i < shape_.size(); ++i) {
      offset += start_offset[i] * strides_[i];
      span_size *= span_sizes[i];
    }
    size_t dtype_size = get_dtype_size(dtype_);
    return Tensor(span_sizes, data_.span(offset * dtype_size, span_size * dtype_size), dtype_,
                  *allocator_);
  }

  std::unique_ptr<Task> copy_to(Tensor &dest) const {
    if (data_size_ != dest.data_size_) {
      throw std::invalid_argument("Tensor copy_to: Shape mismatch between source and destination");
    }
    if (dtype_ != dest.dtype_) {
      throw std::invalid_argument(
          "Tensor copy_to: Data type mismatch between source and destination");
    }
    DISPATCH_ANY_DTYPE(dtype_, T, return ops::cd_copy<T>(data_, dest.data_, data_size_));
  }

  std::unique_ptr<Task> fill(double value, flowHandle_t handle = defaultFlowHandle) {
    DISPATCH_ANY_DTYPE(dtype_, T,
                       return ops::set_scalar<T>(data_, static_cast<T>(value), data_size_, handle));
  }

  std::unique_ptr<Task> fill_random_normal(double mean, double stddev, unsigned long long seed) {
    DISPATCH_ANY_DTYPE(dtype_, T,
                       return ops::fill_random_normal(data_, data_size_, static_cast<T>(mean),
                                                      static_cast<T>(stddev), seed));
  }

  std::unique_ptr<Task> fill_random_uniform(double low, double high, unsigned long long seed) {
    DISPATCH_ANY_DTYPE(dtype_, T,
                       return ops::fill_random_uniform(data_, data_size_, static_cast<T>(low),
                                                       static_cast<T>(high), seed));
  }

  std::unique_ptr<Task> fill_random_normal(double mean, double stddev) {
    unsigned long long seed = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
        reinterpret_cast<uintptr_t>(data()));
    return fill_random_normal(mean, stddev, seed);
  }

  std::unique_ptr<Task> fill_random_uniform(double low, double high) {
    unsigned long long seed = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
        reinterpret_cast<uintptr_t>(data()));
    return fill_random_uniform(low, high, seed);
  }

  Tensor to_device(const Device &target_device) const {
    if (device() == target_device) {
      return *this;
    }
    Tensor result(shape_, dtype_, DeviceAllocator::instance(target_device));
    copy_to(result);
    return result;
  }

  Tensor to_host() const { return to_device(getHost()); }

  Tensor clone() const {
    Tensor result(shape_, dtype_, allocator_);
    copy_to(result);
    return result;
  }
};

template <typename Archiver>
void archive(Archiver &archive, const Tensor &tensor) {
  const DType_t &dtype = tensor.data_type();
  Vec<size_t> shape_vec = tensor.shape();
  archive(dtype);
  archive(shape_vec);
  dptr data_ptr = tensor.data_ptr();
  archive(make_blob(data_ptr.get<unsigned char>(), tensor.size() * get_dtype_size(dtype),

                    tensor.device()));
}

}  // namespace synet

#include "tensor/tensor_arithmetic.hpp"  // IWYU pragma: export