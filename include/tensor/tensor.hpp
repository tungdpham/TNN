/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cassert>
#include <memory>
#include <vector>

#include "device/device.hpp"
#include "device/dptr.hpp"
#include "device/iallocator.hpp"
#include "device/task.hpp"
#include "type/type.hpp"

namespace tnn {

class ITensor;

using Tensor = std::shared_ptr<ITensor>;             // ref to mutable tensor
using ConstTensor = std::shared_ptr<const ITensor>;  // ref to immutable tensor

class ITensor {
public:
  virtual ~ITensor() = default;

  // basic properties
  virtual DType_t data_type() const = 0;
  virtual size_t size() const = 0;
  virtual size_t capacity() const = 0;
  virtual const std::vector<size_t> &shape() const = 0;
  virtual std::string shape_str() const = 0;
  virtual size_t dims() const = 0;
  virtual size_t dimension(const size_t index) const = 0;
  virtual size_t stride(const size_t index) const = 0;

  // device properties
  virtual IAllocator &allocator() const = 0;
  virtual const Device &device() const = 0;
  virtual DeviceType device_type() const = 0;
  virtual bool is_aligned(size_t alignment = 32) const = 0;

  virtual Tensor clone() const = 0;
  virtual void head(size_t n = 10) const = 0;
  virtual void print_data() const = 0;
  virtual void save(std::ofstream &out) const = 0;

  // stats
  virtual double min() const = 0;
  virtual double max() const = 0;
  virtual double mean() const = 0;
  virtual double variance() const = 0;

  // Data access
  template <typename U>
  U &at(std::initializer_list<size_t> indices) {
    assert(device().device_type() == DeviceType::CPU && "at() is only available for CPU tensors");
    assert(indices.size() == shape().size());
    size_t index = compute_index(indices);
    return data_as<U>()[index];
  }

  template <typename U>
  const U &at(std::initializer_list<size_t> indices) const {
    assert(device().device_type() == DeviceType::CPU && "at() is only available for CPU tensors");
    assert(indices.size() == shape().size());
    size_t index = compute_index(indices);
    return data_as<U>()[index];
  }

  template <typename U>
  U *data_as() {
    if (this->data_type() != dtype_of<U>()) {
      throw std::runtime_error("Tensor data type mismatch in data_as()");
    }
    return static_cast<U *>(data());
  }
  template <typename U>
  const U *data_as() const {
    if (this->data_type() != dtype_of<U>()) {
      throw std::runtime_error("Tensor data type mismatch in data_as()");
    }
    return static_cast<const U *>(data());
  }
  virtual void *data() = 0;
  virtual const void *data() const = 0;
  virtual dptr data_ptr() = 0;
  virtual const dptr data_ptr() const = 0;

  // Operations
  virtual Tensor span(std::vector<size_t> start_offset, std::vector<size_t> span_sizes) const = 0;
  virtual void resize(const std::vector<size_t> &new_shape) = 0;
  virtual void ensure(const std::vector<size_t> &new_shape) = 0;
  virtual void copy_to(const Tensor &target) const = 0;
  virtual Tensor to_cpu() const = 0;
  virtual Tensor to_gpu(int gpu_id = 0) const = 0;
  virtual Tensor to_device(const Device &target_device) const = 0;

  virtual void add(const ConstTensor &other) = 0;
  virtual void sub(const ConstTensor &other) = 0;
  virtual void mul(const ConstTensor &other) = 0;
  virtual void div(const ConstTensor &other) = 0;
  virtual void add_scalar(double scalar) = 0;
  virtual void sub_scalar(double scalar) = 0;
  virtual void mul_scalar(double scalar) = 0;
  virtual void div_scalar(double scalar) = 0;

  virtual std::unique_ptr<Task> fill(double value) = 0;
  virtual void fill_random_uniform(double range) = 0;
  virtual void fill_random_uniform(double min_val, double max_val) = 0;
  virtual void fill_random_uniform(double min_val, double max_val, unsigned long long seed) = 0;
  virtual void fill_random_normal(double mean, double stddev) = 0;
  virtual void fill_random_normal(double mean, double stddev, unsigned long long seed) = 0;

private:
  virtual size_t compute_index(std::initializer_list<size_t> indices) const = 0;
};

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

}  // namespace tnn

#include "tensor/tensor_factory.hpp"  // IWYU pragma: keep
