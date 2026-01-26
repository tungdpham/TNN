/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/device.hpp"
#include "device/device_manager.hpp"
#include "device/device_ptr.hpp"
#include "device/mem_pool.hpp"
#include "device/task.hpp"
#include "ops/ops.hpp"
#include "type/type.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
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

namespace tnn {

class ITensor;

class Tensor;

template <typename T> class TypedTensor;

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
  virtual bool is_on_cpu() const = 0;
  virtual bool is_on_gpu() const = 0;
  virtual const Device *device() const = 0;
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
  template <typename U> U &at(std::initializer_list<size_t> indices) {
    assert(device()->device_type() == DeviceType::CPU && "at() is only available for CPU tensors");
    assert(indices.size() == shape().size());
    size_t index = compute_index(indices);
    return data_as<U>()[index];
  }

  template <typename U> const U &at(std::initializer_list<size_t> indices) const {
    assert(device()->device_type() == DeviceType::CPU && "at() is only available for CPU tensors");
    assert(indices.size() == shape().size());
    size_t index = compute_index(indices);
    return data_as<U>()[index];
  }

  template <typename U> U *data_as() {
    if (this->data_type() != dtype_of<U>()) {
      throw std::runtime_error("Tensor data type mismatch in data_as()");
    }
    return static_cast<U *>(data());
  }
  template <typename U> const U *data_as() const {
    if (this->data_type() != dtype_of<U>()) {
      throw std::runtime_error("Tensor data type mismatch in data_as()");
    }
    return static_cast<const U *>(data());
  }
  virtual void *data() = 0;
  virtual const void *data() const = 0;

  // Operations
  virtual void resize(const std::vector<size_t> &new_shape, const Device *new_device = nullptr) = 0;
  virtual void ensure(const std::vector<size_t> &new_shape, const Device *new_device = nullptr) = 0;
  virtual void copy_to(Tensor &target) const = 0;
  virtual Tensor to_cpu() const = 0;
  virtual Tensor to_gpu(int gpu_id = 0) const = 0;
  virtual Tensor to_device(const Device *target_device) const = 0;

  virtual void add(const Tensor &other) = 0;
  virtual void sub(const Tensor &other) = 0;
  virtual void mul(const Tensor &other) = 0;
  virtual void div(const Tensor &other) = 0;
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

class Tensor : public std::shared_ptr<ITensor> {
public:
  using std::shared_ptr<ITensor>::shared_ptr;

  Tensor() : std::shared_ptr<ITensor>() {}

  Tensor(const std::shared_ptr<ITensor> &ptr) : std::shared_ptr<ITensor>(ptr) {}

  Tensor(std::shared_ptr<ITensor> &&ptr) : std::shared_ptr<ITensor>(std::move(ptr)) {}

  template <typename T>
  static Tensor create(std::vector<size_t> shape, const Device *device = &getCPU());

  template <typename T>
  static Tensor create(std::vector<size_t> shape, const device_ptr &data,
                       const Device *device = &getCPU());

  template <typename T>
  static Tensor create(std::initializer_list<size_t> shape = {}, const Device *device = &getCPU());

  template <typename T>
  static Tensor create(std::initializer_list<size_t> shape, const device_ptr &data,
                       const Device *device = &getCPU());

  static Tensor create(DType_t dtype, std::vector<size_t> shape, const Device *device = &getCPU());

  static Tensor create(DType_t dtype, std::initializer_list<size_t> shape = {},
                       const Device *device = &getCPU());

  template <typename T> static Tensor create_pooled(MemPool &mem_pool, std::vector<size_t> shape);

  template <typename T>
  static Tensor create_pooled(MemPool &mem_pool, std::initializer_list<size_t> shape);

  static Tensor create_pooled(MemPool &mem_pool, DType_t dtype, std::vector<size_t> shape);

  static Tensor create_pooled(MemPool &mem_pool, DType_t dtype,
                              std::initializer_list<size_t> shape);

  template <typename T> static std::shared_ptr<TypedTensor<T>> cast(const Tensor &tensor);

  template <typename T> static Tensor load(std::ifstream &in, const Device *device = &getCPU());

  static void load_into(std::ifstream &in, Tensor &target, const Device *device = &getCPU());
};

/**
 * @brief A tensor class dedicated for ML and DL applications.
 * @tparam T Data type (e.g., float, double, int)
 * Data layout is assumed to be row-major (C-style) by default.
 * Generic N-dimensional tensor with various lity functions. How layers interpret
 * the dimensions is up to the layer implementation.
 */
template <typename T = float> class TypedTensor : public ITensor {
protected:
  size_t data_size_;
  device_ptr data_;
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

  inline size_t compute_index(std::initializer_list<size_t> indices) const override {
    assert(indices.size() == shape_.size());
    size_t index = 0;
    size_t i = 0;
    for (auto idx : indices) {
      index += idx * compute_stride(i++);
    }
    return index;
  }

  device_ptr allocate_data(const Device *device, size_t size) {
    return make_dptr_t<T>(device, size);
  }

public:
  // Constructors and Destructor
  TypedTensor(const Device *device = &getCPU()) : data_size_(0) {
    for (size_t i = 0; i < shape_.size(); ++i) {
      shape_[i] = 0;
    }
    data_ = allocate_data(device, 0);
  }

  TypedTensor(std::initializer_list<size_t> shape_list, const Device *device = &getCPU())
      : shape_(shape_list) {
    data_size_ =
        std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    data_ = allocate_data(device, data_size_);
  }

  TypedTensor(std::initializer_list<size_t> shape_list, const device_ptr &data,
              const Device *device = &getCPU())
      : shape_(shape_list) {
    data_size_ =
        std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    data_ = allocate_data(device, data_size_);
    if (data.get<T>() != nullptr) {
      ops::copy<T>(data, data_, data_size_);
    }
  }

  TypedTensor(std::vector<size_t> shape, const Device *device = &getCPU())
      : shape_(std::move(shape)) {
    data_size_ =
        std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    data_ = allocate_data(device, data_size_);
  }

  TypedTensor(std::vector<size_t> shape, const device_ptr &data, const Device *device = &getCPU())
      : shape_(std::move(shape)) {
    data_size_ =
        std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    data_ = allocate_data(device, data_size_);
    if (data.get<T>() != nullptr) {
      ops::copy<T>(data, data_, data_size_);
    }
  }

  ~TypedTensor() {
    // data_ will be automatically released by device_ptr destructor
  }

  TypedTensor(const TypedTensor &other) : data_size_(other.data_size_), shape_(other.shape_) {
    if (data_size_ > 0) {
      data_ = allocate_data(other.device(), data_size_);
      ops::copy<T>(other.data_, data_, data_size_);
    }
  }

  TypedTensor(TypedTensor &&other) noexcept
      : data_size_(other.data_size_), shape_(std::move(other.shape_)) {
    data_ = std::move(other.data_);
    other.data_size_ = 0;
  }

  template <typename... Indices> T &operator()(Indices... indices) {
    assert(device()->device_type() == DeviceType::CPU &&
           "Operator() is only available for CPU tensors");
    return data_.get<T>()[compute_index(indices...)];
  }

  T &operator()(std::initializer_list<size_t> indices) {
    assert(device()->device_type() == DeviceType::CPU &&
           "Operator() is only available for CPU tensors");
    return data_.get<T>()[compute_index(indices)];
  }

  template <typename... Indices> const T &operator()(Indices... indices) const {
    assert(device()->device_type() == DeviceType::CPU &&
           "Operator() is only available for CPU tensors");
    return data_.get<T>()[compute_index(indices...)];
  }

  const T &operator()(std::initializer_list<size_t> indices) const {
    assert(device()->device_type() == DeviceType::CPU &&
           "Operator() is only available for CPU tensors");
    return data_.get<T>()[compute_index(indices)];
  }

  void *data() override { return static_cast<void *>(data_.get<T>()); }
  const void *data() const override { return static_cast<const void *>(data_.get<T>()); }

  // Operators
  TypedTensor<T> &operator=(const TypedTensor<T> &other) {
    if (this != &other) {
      data_ = device_ptr(nullptr);
      data_size_ = other.data_size_;
      shape_ = other.shape_;
      data_ = allocate_data(other.device(), data_size_);
      if (data_size_ > 0) {
        ops::copy<T>(other.data_, data_, data_size_);
      }
    }
    return *this;
  }

  TypedTensor<T> &operator=(TypedTensor<T> &&other) noexcept {
    if (this != &other) {
      shape_ = std::move(other.shape_);
      data_ = std::move(other.data_);
      data_size_ = other.data_size_;

      other.data_size_ = 0;
    }
    return *this;
  }

  bool same_shape(const TypedTensor<T> &other) const { return shape_ == other.shape_; }

  TypedTensor<T> operator+(const TypedTensor<T> &other) const {
    if (!same_shape(other)) {
      throw std::invalid_argument("TypedTensor shapes must match for addition");
    }

    TypedTensor<T> result(shape_, device());
    ops::add<T>(data_, other.data_, result.data_, data_size_);
    return result;
  }

  TypedTensor<T> operator-(const TypedTensor<T> &other) const {
    if (!same_shape(other)) {
      throw std::invalid_argument("TypedTensor shapes must match for subtraction");
    }

    TypedTensor<T> result(shape_, device());
    ops::sub<T>(data_, other.data_, result.data_, data_size_);
    return result;
  }

  TypedTensor<T> operator*(const TypedTensor<T> &other) const {
    if (!same_shape(other)) {
      throw std::invalid_argument("TypedTensor shapes must match for element-wise multiplication");
    }

    TypedTensor<T> result(shape_, device());
    ops::mul<T>(data_, other.data_, result.data_, data_size_);
    return result;
  }

  TypedTensor<T> operator/(const TypedTensor<T> &other) const {
    if (!same_shape(other)) {
      throw std::invalid_argument("TypedTensor shapes must match for element-wise division");
    }

    TypedTensor<T> result(shape_, device());
    ops::div<T>(data_, other.data_, result.data_, data_size_);
    return result;
  }

  TypedTensor<T> operator+(T scalar) const {
    TypedTensor<T> result(shape_, device());
    ops::add_scalar<T>(data_, scalar, result.data_, data_size_);
    return result;
  }

  TypedTensor<T> operator-(T scalar) const {
    TypedTensor<T> result(shape_, device());
    ops::sub_scalar(data_, scalar, result.data_, data_size_);
    return result;
  }

  TypedTensor<T> operator*(T scalar) const {
    TypedTensor<T> result(shape_, device());
    ops::mul_scalar(data_, scalar, result.data_, data_size_);
    return result;
  }

  TypedTensor<T> operator/(T scalar) const {
    if (scalar == T(0)) {
      throw std::invalid_argument("Division by zero");
    }

    TypedTensor<T> result(shape_, device());
    ops::div_scalar(data_, scalar, result.data_, data_size_);
    return result;
  }

  TypedTensor<T> &operator+=(const TypedTensor<T> &other) {
    if (!same_shape(other)) {
      throw std::invalid_argument("TypedTensor shapes must match for addition");
    }
    ops::add<T>(data_, other.data_, data_, data_size_);
    return *this;
  }

  TypedTensor<T> &operator-=(const TypedTensor<T> &other) {
    if (!same_shape(other)) {
      throw std::invalid_argument("TypedTensor shapes must match for subtraction");
    }
    ops::sub<T>(data_, other.data_, data_, data_size_);
    return *this;
  }

  TypedTensor<T> &operator*=(const TypedTensor<T> &other) {
    if (!same_shape(other)) {
      throw std::invalid_argument("TypedTensor shapes must match for element-wise multiplication");
    }
    ops::mul<T>(data_, other.data_, data_, data_size_);
    return *this;
  }

  TypedTensor<T> &operator+=(T scalar) {
    ops::add_scalar<T>(data_, scalar, data_, data_size_);
    return *this;
  }

  TypedTensor<T> &operator-=(T scalar) {
    ops::sub_scalar<T>(data_, scalar, data_, data_size_);
    return *this;
  }

  TypedTensor<T> &operator*=(T scalar) {
    ops::mul_scalar<T>(data_, scalar, data_, data_size_);
    return *this;
  }

  TypedTensor<T> &operator/=(T scalar) {
    if (scalar == T(0)) {
      throw std::invalid_argument("Division by zero");
    }
    ops::div_scalar(data_, scalar, data_, data_size_);
    return *this;
  }

  const std::vector<size_t> &shape() const override { return shape_; }

  std::string shape_str() const override {
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

  size_t dims() const override { return shape_.size(); }

  size_t dimension(const size_t index) const override { return shape_[index]; }

  size_t stride(const size_t index) const override { return compute_stride(index); }

  size_t size() const override { return data_size_; }

  size_t capacity() const override { return data_.capacity() / sizeof(T); }

  bool is_aligned(size_t alignment = 32) const override {
    return (reinterpret_cast<uintptr_t>(data_.get<T>()) % alignment) == 0;
  }

  const Device *device() const override { return data_.getDevice(); }

  DeviceType device_type() const override { return device()->device_type(); }

  bool is_on_cpu() const override { return device()->device_type() == DeviceType::CPU; }

  bool is_on_gpu() const override { return device()->device_type() == DeviceType::GPU; }

  const device_ptr &data_ptr() const { return data_; }

  device_ptr &data_ptr() { return data_; }

  Tensor to_cpu() const override {
    if (device_type() == DeviceType::CPU) {
      return clone();
    }

    if (device_type() == DeviceType::GPU) {
      std::vector<size_t> shape_vec(shape_);
      std::shared_ptr<TypedTensor<T>> cpu_tensor =
          std::make_shared<TypedTensor<T>>(shape_vec, &getCPU());
      // Copy from GPU to CPU
      data_.getDevice()->copyToHost(cpu_tensor->data_.template get<T>(), data_.template get<T>(),
                                    data_size_ * sizeof(T));
      return cpu_tensor;
    }
    throw std::runtime_error("Unsupported device type for to_cpu()");
  }

  Tensor to_gpu(int gpu_id = 0) const override {
    if (device_type() == DeviceType::GPU) {
      return clone();
    }

    if (device_type() == DeviceType::CPU) {
      std::vector<size_t> shape_vec(shape_);
      auto gpu_tensor = std::make_shared<TypedTensor<T>>(shape_vec, &getGPU(gpu_id));

      // Copy from CPU to GPU
      getGPU(gpu_id).copyToDevice(gpu_tensor->data_.template get<T>(), data_.template get<T>(),
                                  data_size_ * sizeof(T));
      return gpu_tensor;
    }

    throw std::runtime_error("Unsupported device type for to_gpu()");
  }

  Tensor to_device(const Device *target_device) const override {
    if (device() == target_device) {
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

  Tensor clone() const override {
    return std::make_shared<TypedTensor<T>>(shape_, data_, device());
  }

  std::unique_ptr<Task> fill(double value) override {
    return ops::set_scalar<T>(data_, static_cast<T>(value), data_size_);
  }

  // Arithmetic operations returning shared_ptr
  void add(const Tensor &other) override {
    auto other_typed = std::dynamic_pointer_cast<TypedTensor<T>>(other);
    if (!other_typed) {
      throw std::runtime_error("Type mismatch in tensor addition");
    }
    if (!same_shape(*other_typed)) {
      throw std::invalid_argument("Tensor shapes must match for addition");
    }
    ops::add<T>(data_, other_typed->data_, data_, data_size_);
  }

  void sub(const Tensor &other) override {
    auto other_typed = std::dynamic_pointer_cast<TypedTensor<T>>(other);
    if (!other_typed) {
      throw std::runtime_error("Type mismatch in tensor subtraction");
    }
    if (!same_shape(*other_typed)) {
      throw std::invalid_argument("Tensor shapes must match for subtraction");
    }
    ops::sub<T>(data_, other_typed->data_, data_, data_size_);
  }

  void mul(const Tensor &other) override {
    auto other_typed = std::dynamic_pointer_cast<TypedTensor<T>>(other);
    if (!other_typed) {
      throw std::runtime_error("Type mismatch in tensor multiplication");
    }
    if (!same_shape(*other_typed)) {
      throw std::invalid_argument("Tensor shapes must match for element-wise multiplication");
    }
    ops::mul<T>(data_, other_typed->data_, data_, data_size_);
  }

  void div(const Tensor &other) override {
    auto other_typed = std::dynamic_pointer_cast<TypedTensor<T>>(other);
    if (!other_typed) {
      throw std::runtime_error("Type mismatch in tensor division");
    }
    if (!same_shape(*other_typed)) {
      throw std::invalid_argument("Tensor shapes must match for element-wise division");
    }
    ops::div<T>(data_, other_typed->data_, data_, data_size_);
  }

  void add_scalar(double scalar) override {
    ops::add_scalar<T>(data_, static_cast<T>(scalar), data_, data_size_);
  }

  void sub_scalar(double scalar) override {
    ops::sub_scalar<T>(data_, static_cast<T>(scalar), data_, data_size_);
  }

  void mul_scalar(double scalar) override {
    ops::mul_scalar(data_, static_cast<T>(scalar), data_, data_size_);
  }

  void div_scalar(double scalar) override {
    if (scalar == 0.0) {
      throw std::invalid_argument("Division by zero");
    }
    ops::div_scalar<T>(data_, static_cast<T>(scalar), data_, data_size_);
  }

  void fill_random_uniform(double range) override {
    unsigned long long seed = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
        reinterpret_cast<uintptr_t>(data_.get<T>()));
    ops::fill_random_uniform(data_, data_size_, T(0), static_cast<T>(range), seed);
  }

  void fill_random_uniform(double min_val, double max_val) override {
    unsigned long long seed = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
        reinterpret_cast<uintptr_t>(data_.get<T>()));
    ops::fill_random_uniform(data_, data_size_, static_cast<T>(min_val), static_cast<T>(max_val),
                             seed);
  }

  void fill_random_uniform(double min_val, double max_val, unsigned long long seed) override {
    ops::fill_random_uniform(data_, data_size_, static_cast<T>(min_val), static_cast<T>(max_val),
                             seed);
  }

  void fill_random_normal(double mean, double stddev) override {
    unsigned long long seed = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
        reinterpret_cast<uintptr_t>(data_.get<T>()));
    ops::fill_random_normal(data_, data_size_, static_cast<T>(mean), static_cast<T>(stddev), seed);
  }

  void fill_random_normal(double mean, double stddev, unsigned long long seed) override {
    ops::fill_random_normal(data_, data_size_, static_cast<T>(mean), static_cast<T>(stddev), seed);
  }

  /**
   * @brief Copy between typed tensors
   * @param target Target tensor to copy data into
   */
  void copy_to(Tensor &target) const override {
    if (target == nullptr || target->size() < size()) {
      throw std::invalid_argument("Target tensor is null or smaller than source in tensor copy");
    }
    auto target_typed = std::dynamic_pointer_cast<TypedTensor<T>>(target);
    if (!target_typed) {
      throw std::runtime_error("Type mismatch in tensor copy");
    }
    ops::cd_copy<T>(data_, target_typed->data_, data_size_);
  }

  void resize(const std::vector<size_t> &new_shape, const Device *new_device = nullptr) override {
    if (new_device != nullptr && new_device != device()) {
      data_.reset();
      shape_ = new_shape;
      data_size_ =
          std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
      data_ = allocate_data(new_device, data_size_);
      return;
    }
    if (new_shape == shape_) {
      return;
    }

    size_t new_size =
        std::accumulate(new_shape.begin(), new_shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_size != data_size_) {
      data_ = allocate_data(device(), new_size);
      data_size_ = new_size;
    }
    shape_ = new_shape;
  }

  /**
   * Similar to resize but only reallocates if the new size is larger than
   * the current allocated size. Good for caching.
   */
  void ensure(const std::vector<size_t> &new_shape, const Device *new_device = nullptr) override {
    if (new_device != nullptr && new_device != device()) {
      shape_ = new_shape;
      data_size_ =
          std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
      data_ = allocate_data(new_device, data_size_);
      return;
    }
    size_t new_size =
        std::accumulate(new_shape.begin(), new_shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_size * sizeof(T) > data_.capacity()) {
      data_ = allocate_data(device(), new_size);
    }
    data_size_ = new_size;
    shape_ = new_shape;
  }

  void copy_batch(TypedTensor<T> &other, size_t src_batch_idx, size_t dest_batch_idx) {
    size_t batch_size = shape_[0];
    if (dest_batch_idx >= batch_size || src_batch_idx >= other.shape_[0]) {
      throw std::invalid_argument("Invalid batch index for copy");
    }

    if (device() != other.device()) {
      throw std::runtime_error("Cannot copy batch between tensors on different devices. Transfer "
                               "to same device first.");
    }

    size_t batch_stride = compute_stride(0);
    size_t src_offset = src_batch_idx * other.compute_stride(0);
    size_t dest_offset = dest_batch_idx * batch_stride;

    ops::copy<T>(other.data_, data_, batch_stride, src_offset, dest_offset);
  }

  double min() const override {
    auto cpu_tensor = std::dynamic_pointer_cast<TypedTensor<T>>(to_cpu());
    T min_val = cpu_tensor->data_.template get<T>()[0];
    for (size_t i = 1; i < cpu_tensor->data_size_; ++i) {
      if (cpu_tensor->data_.template get<T>()[i] < min_val) {
        min_val = cpu_tensor->data_.template get<T>()[i];
      }
    }
    return static_cast<double>(min_val);
  }

  double max() const override {
    auto cpu_tensor = std::dynamic_pointer_cast<TypedTensor<T>>(to_cpu());
    T max_val = cpu_tensor->data_.template get<T>()[0];
    for (size_t i = 1; i < cpu_tensor->data_size_; ++i) {
      if (cpu_tensor->data_.template get<T>()[i] > max_val) {
        max_val = cpu_tensor->data_.template get<T>()[i];
      }
    }
    return static_cast<double>(max_val);
  }

  double mean() const override {
    T sum = ops::sum<T>(data_, data_size_);
    return static_cast<double>(sum / static_cast<T>((double)data_size_));
  }

  double variance() const override {
    T m = static_cast<T>(mean());
    T sum_sq_diff = ops::sum_squared_diff<T>(data_, m, data_size_);
    return static_cast<double>(sum_sq_diff / static_cast<T>((double)data_size_));
  }

  void print_data() const override {
    Tensor cpu_tensor = to_cpu();
    size_t total_elements = cpu_tensor->size();
    std::cout << "TypedTensor data (shape " << cpu_tensor->shape_str() << "):\n";
    T *data = cpu_tensor->data_as<T>();
    for (size_t i = 0; i < total_elements; ++i) {
      std::cout << std::setprecision(3) << static_cast<float>(data[i]) << " ";
    }
    std::cout << std::endl;
  }

  void head(size_t n = 10) const override {
    Tensor cpu_tensor = to_cpu();
    size_t total_elements = cpu_tensor->size();
    n = std::min(n, total_elements);
    std::cout << "TypedTensor head (first " << n << " elements of shape " << cpu_tensor->shape_str()
              << "):\n";
    T *data = cpu_tensor->data_as<T>();
    for (size_t i = 0; i < n; ++i) {
      std::cout << std::setprecision(3) << static_cast<float>(data[i]) << " ";
    }
    std::cout << std::endl;
  }

  void save(std::ofstream &out) const override {
    if (!out.is_open()) {
      throw std::runtime_error("File is not open for writing");
    }

    // write dims, shape
    size_t dims = shape_.size();
    DType_t data_type = this->data_type();
    out.write(reinterpret_cast<const char *>(&data_type), sizeof(DType_t));
    out.write(reinterpret_cast<const char *>(&dims), sizeof(size_t));
    out.write(reinterpret_cast<const char *>(shape_.data()), shape_.size() * sizeof(size_t));

    if (device_type() == DeviceType::CPU) {
      out.write(reinterpret_cast<const char *>(data_.get<T>()), data_size_ * sizeof(T));
    } else {
      std::vector<T> host_buffer(data_size_);
      device()->copyToHost(host_buffer.data(), data_.get<T>(), data_size_ * sizeof(T));
      out.write(reinterpret_cast<const char *>(host_buffer.data()), data_size_ * sizeof(T));
    }
  }

  DType_t data_type() const override {
    if (std::is_same<T, float>::value) {
      return DType_t::FP32;
    } else if (std::is_same<T, double>::value) {
      return DType_t::FP64;
    } else if (std::is_same<T, fp16>::value) {
      return DType_t::FP16;
    } else if (std::is_same<T, int32_t>::value) {
      return DType_t::INT32_T;
    } else if (std::is_same<T, int64_t>::value) {
      return DType_t::INT64_T;
    } else if (std::is_same<T, size_t>::value) {
      return DType_t::SIZE_T;
    } else {
      throw std::runtime_error("Unsupported data type for TypedTensor");
    }
  }
};

template <typename T> class PooledTypedTensor : public TypedTensor<T> {
public:
  using TypedTensor<T>::TypedTensor;

  // ! Do not call default TypedTensor constructor with shape as that will allocate memory
  PooledTypedTensor(MemPool &mem_pool, std::vector<size_t> shape)
      : TypedTensor<T>(&mem_pool.device()), mem_pool_(mem_pool) {
    this->shape_ = std::move(shape);
    this->data_size_ = std::accumulate(this->shape_.begin(), this->shape_.end(), size_t(1),
                                       std::multiplies<size_t>());
    this->data_ = mem_pool_.get(this->data_size_ * sizeof(T));
  }

  ~PooledTypedTensor() override { mem_pool_.release(std::move(this->data_)); }

private:
  MemPool &mem_pool_;
};

template <typename T>
void check_nan_and_inf(const T *data, size_t size, const std::string &tensor_name = "") {
  for (size_t i = 0; i < size; ++i) {
    if (std::isnan(data[i]) || std::isinf(data[i])) {
      std::cerr << "TypedTensor " << tensor_name << " contains NaN or Inf at index " << i
                << std::endl;
      return;
    }
  }
}

template <typename T>
void check_nan_and_inf(const TypedTensor<T> &tensor, const std::string &tensor_name = "") {
  auto cpu_tensor = std::dynamic_pointer_cast<TypedTensor<T>>(tensor.to_cpu());
  size_t total_elements = cpu_tensor->size();
  T *data = cpu_tensor->data_ptr().template get<T>();
  check_nan_and_inf(data, total_elements, tensor_name);
}

inline void check_nan_and_inf(const Tensor &tensor, const std::string &tensor_name = "") {
  DType_t dtype = tensor->data_type();
  switch (dtype) {
  case DType_t::FP32: {
    auto typed_tensor = std::dynamic_pointer_cast<TypedTensor<float>>(tensor);
    check_nan_and_inf<float>(*typed_tensor, tensor_name);
    break;
  }
  case DType_t::FP64: {
    auto typed_tensor = std::dynamic_pointer_cast<TypedTensor<double>>(tensor);
    check_nan_and_inf<double>(*typed_tensor, tensor_name);
    break;
  }
  case DType_t::FP16: {
    throw std::runtime_error("check_nan_and_inf not implemented for FP16 tensors");
    break;
  }
  default:
    throw std::runtime_error("Unsupported data type for check_nan_and_inf");
  }
}

// Prints data density at ranges (2^-32, 2^-31, ..., 2^31, 2^32)
inline void print_data_distribution(const Tensor &tensor) {
  if (!tensor) {
    std::cerr << "Cannot print distribution of null tensor" << std::endl;
    return;
  }

  Tensor cpu_tensor = tensor->to_cpu();
  DType_t dtype = cpu_tensor->data_type();

  constexpr int min_exp = -32;
  constexpr int max_exp = 32;
  constexpr int num_buckets = max_exp - min_exp + 1;

  // buckets[0] = values < 2^-32 (including zeros)
  // buckets[1..num_buckets] = values in [2^exp, 2^(exp+1))
  // buckets[num_buckets+1] = values >= 2^32
  std::vector<size_t> buckets(num_buckets + 2, 0);

  auto process_data = [&]<typename T>() {
    auto typed_tensor = std::dynamic_pointer_cast<TypedTensor<T>>(cpu_tensor);
    if (!typed_tensor) {
      throw std::runtime_error("Failed to cast tensor in print_data_distribution");
    }

    T *data = typed_tensor->data_ptr().template get<T>();
    size_t size = typed_tensor->size();

    for (size_t i = 0; i < size; ++i) {
      T val = data[i];
      double abs_val = std::abs(static_cast<double>(val));

      if (abs_val == 0.0 || abs_val < std::pow(2.0, min_exp)) {
        buckets[0]++;
      } else if (abs_val >= std::pow(2.0, max_exp + 1)) {
        buckets[num_buckets + 1]++;
      } else {
        double log2_val = std::log2(abs_val);
        int exp = static_cast<int>(std::floor(log2_val));

        exp = std::max(min_exp, std::min(max_exp, exp));
        int bucket_idx = exp - min_exp + 1;
        buckets[bucket_idx]++;
      }
    }
  };

  switch (dtype) {
  case DType_t::FP32:
    process_data.template operator()<float>();
    break;
  case DType_t::FP64:
    process_data.template operator()<double>();
    break;
  case DType_t::FP16:
    process_data.template operator()<fp16>();
    break;
  default:
    std::cerr << "Unsupported data type for print_data_distribution" << std::endl;
    return;
  }

  // Print distribution
  size_t total = cpu_tensor->size();
  std::cout << "\nData Distribution (shape " << cpu_tensor->shape_str() << ", " << total
            << " elements):\n";
  std::cout << std::setw(20) << "Range" << std::setw(15) << "Count" << std::setw(15)
            << "Percentage\n";
  std::cout << std::string(50, '-') << "\n";

  // Zero/very small values
  if (buckets[0] > 0) {
    double pct = 100.0 * buckets[0] / total;
    std::cout << std::setw(20) << "< 2^-32 (or zero)" << std::setw(15) << buckets[0]
              << std::setw(14) << std::fixed << std::setprecision(2) << pct << "%\n";
  }

  // Regular buckets - only show non-empty buckets
  for (int exp = min_exp; exp <= max_exp; ++exp) {
    int bucket_idx = exp - min_exp + 1;
    if (buckets[bucket_idx] > 0) {
      double pct = 100.0 * buckets[bucket_idx] / total;
      std::ostringstream range;
      range << "[2^" << exp << ", 2^" << (exp + 1) << ")";
      std::cout << std::setw(20) << range.str() << std::setw(15) << buckets[bucket_idx]
                << std::setw(14) << std::fixed << std::setprecision(2) << pct << "%\n";
    }
  }

  // Very large values
  if (buckets[num_buckets + 1] > 0) {
    double pct = 100.0 * buckets[num_buckets + 1] / total;
    std::cout << std::setw(20) << ">= 2^33" << std::setw(15) << buckets[num_buckets + 1]
              << std::setw(14) << std::fixed << std::setprecision(2) << pct << "%\n";
  }

  std::cout << std::endl;
}

// Tensor static method implementations
template <typename T>
inline Tensor Tensor::create(std::vector<size_t> shape, const Device *device) {
  return std::make_shared<TypedTensor<T>>(shape, device);
}

template <typename T>
inline Tensor Tensor::create(std::vector<size_t> shape, const device_ptr &data,
                             const Device *device) {
  return std::make_shared<TypedTensor<T>>(shape, data, device);
}

template <typename T>
inline Tensor Tensor::create(std::initializer_list<size_t> shape, const Device *device) {
  return std::make_shared<TypedTensor<T>>(shape, device);
}

template <typename T>
inline Tensor Tensor::create(std::initializer_list<size_t> shape, const device_ptr &data,
                             const Device *device) {
  return std::make_shared<TypedTensor<T>>(shape, data, device);
}

inline Tensor Tensor::create(DType_t dtype, std::vector<size_t> shape, const Device *device) {
  switch (dtype) {
  case DType_t::FP32:
    return create<float>(shape, device);
  case DType_t::FP64:
    return create<double>(shape, device);
  case DType_t::FP16:
    return create<fp16>(shape, device);
  default:
    throw std::runtime_error("Unsupported data type for Tensor::create_from_dtype");
  }
}

inline Tensor Tensor::create(DType_t dtype, std::initializer_list<size_t> shape,
                             const Device *device) {
  switch (dtype) {
  case DType_t::FP32:
    return create<float>(shape, device);
  case DType_t::FP64:
    return create<double>(shape, device);
  case DType_t::FP16:
    return create<fp16>(shape, device);
  default:
    throw std::runtime_error("Unsupported data type for Tensor::create_from_dtype");
  }
}

template <typename T>
inline Tensor Tensor::create_pooled(MemPool &mem_pool, std::vector<size_t> shape) {
  return std::make_shared<PooledTypedTensor<T>>(mem_pool, shape);
}

template <typename T>
inline Tensor Tensor::create_pooled(MemPool &mem_pool, std::initializer_list<size_t> shape) {
  return std::make_shared<PooledTypedTensor<T>>(mem_pool, std::vector<size_t>(shape));
}

inline Tensor Tensor::create_pooled(MemPool &mem_pool, DType_t dtype, std::vector<size_t> shape) {
  switch (dtype) {
  case DType_t::FP32:
    return create_pooled<float>(mem_pool, shape);
  case DType_t::FP64:
    return create_pooled<double>(mem_pool, shape);
  case DType_t::FP16:
    return create_pooled<fp16>(mem_pool, shape);
  default:
    throw std::runtime_error("Unsupported data type for Tensor::create_pooled");
  }
}

inline Tensor Tensor::create_pooled(MemPool &mem_pool, DType_t dtype,
                                    std::initializer_list<size_t> shape) {
  switch (dtype) {
  case DType_t::FP32:
    return create_pooled<float>(mem_pool, shape);
  case DType_t::FP64:
    return create_pooled<double>(mem_pool, shape);
  case DType_t::FP16:
    return create_pooled<fp16>(mem_pool, shape);
  default:
    throw std::runtime_error("Unsupported data type for Tensor::create_pooled");
  }
}

template <typename T> inline std::shared_ptr<TypedTensor<T>> Tensor::cast(const Tensor &tensor) {
  auto typed = std::dynamic_pointer_cast<TypedTensor<T>>(tensor);
  if (!typed) {
    throw std::runtime_error("Invalid tensor type cast");
  }
  return typed;
}

template <typename T> inline Tensor Tensor::load(std::ifstream &in, const Device *device) {
  if (!in.is_open()) {
    throw std::runtime_error("File is not open for reading");
  }
  size_t dims;
  in.read(reinterpret_cast<char *>(&dims), sizeof(size_t));
  std::vector<size_t> shape(dims);
  in.read(reinterpret_cast<char *>(shape.data()), dims * sizeof(size_t));
  if (in.gcount() != static_cast<std::streamsize>(dims * sizeof(size_t))) {
    throw std::runtime_error("Failed to read tensor shape from file");
  }

  auto tensor = std::make_shared<TypedTensor<T>>(shape, device);
  if (device->device_type() == DeviceType::CPU) {
    in.read(reinterpret_cast<char *>(tensor->data()), tensor->size() * sizeof(T));
    if (in.gcount() != static_cast<std::streamsize>(tensor->size() * sizeof(T))) {
      throw std::runtime_error("Failed to read tensor data from file");
    }
  } else {
    std::vector<T> host_buffer(tensor->size());
    in.read(reinterpret_cast<char *>(host_buffer.data()), tensor->size() * sizeof(T));
    if (in.gcount() != static_cast<std::streamsize>(tensor->size() * sizeof(T))) {
      throw std::runtime_error("Failed to read tensor data from file");
    }
    device->copyToDevice(tensor->data(), host_buffer.data(), tensor->size() * sizeof(T));
  }
  return tensor;
}

inline void Tensor::load_into(std::ifstream &in, Tensor &target, const Device *device) {
  if (!in.is_open()) {
    throw std::runtime_error("File is not open for reading");
  }
  // read dtype, dims, shape, and data
  DType_t dtype;
  in.read(reinterpret_cast<char *>(&dtype), sizeof(DType_t));
  size_t dims;
  in.read(reinterpret_cast<char *>(&dims), sizeof(size_t));
  std::vector<size_t> shape(dims);
  in.read(reinterpret_cast<char *>(shape.data()), dims * sizeof(size_t));
  if (in.gcount() != static_cast<std::streamsize>(dims * sizeof(size_t))) {
    throw std::runtime_error("Failed to read tensor shape from file");
  }

  target->resize(shape, device);

  if (device->device_type() == DeviceType::CPU) {
    in.read(reinterpret_cast<char *>(target->data()), target->size() * get_dtype_size(dtype));
    if (in.gcount() != static_cast<std::streamsize>(target->size() * get_dtype_size(dtype))) {
      throw std::runtime_error("Failed to read tensor data from file");
    }
  } else {
    void *host_buffer = malloc(target->size() * get_dtype_size(dtype));
    in.read(reinterpret_cast<char *>(host_buffer), target->size() * get_dtype_size(dtype));
    if (in.gcount() != static_cast<std::streamsize>(target->size() * get_dtype_size(dtype))) {
      throw std::runtime_error("Failed to read tensor data from file");
    }
    device->copyToDevice(target->data(), host_buffer, target->size() * get_dtype_size(dtype));
    free(host_buffer);
  }
}
inline Tensor operator+(const Tensor &lhs, const Tensor &rhs) {
  Tensor result = lhs->clone();
  result->add(rhs);
  return result;
}

inline Tensor operator-(const Tensor &lhs, const Tensor &rhs) {
  Tensor result = lhs->clone();
  result->sub(rhs);
  return result;
}

inline Tensor operator*(const Tensor &lhs, const Tensor &rhs) {
  Tensor result = lhs->clone();
  result->mul(rhs);
  return result;
}

inline Tensor operator/(const Tensor &lhs, const Tensor &rhs) {
  Tensor result = lhs->clone();
  result->div(rhs);
  return result;
}

inline Tensor operator+(const Tensor &lhs, double scalar) {
  Tensor result = lhs->clone();
  result->add_scalar(scalar);
  return result;
}

inline Tensor operator-(const Tensor &lhs, double scalar) {
  Tensor result = lhs->clone();
  result->sub_scalar(scalar);
  return result;
}

inline Tensor operator*(const Tensor &lhs, double scalar) {
  Tensor result = lhs->clone();
  result->mul_scalar(scalar);
  return result;
}

inline Tensor operator/(const Tensor &lhs, double scalar) {
  Tensor result = lhs->clone();
  result->div_scalar(scalar);
  return result;
}

inline Tensor operator+(double scalar, const Tensor &rhs) {
  Tensor result = rhs->clone();
  result->add_scalar(scalar);
  return result;
}

inline Tensor operator*(double scalar, const Tensor &rhs) {
  Tensor result = rhs->clone();
  result->mul_scalar(scalar);
  return result;
}

// Convenience wrapper for backward compatibility
template <typename T>
inline Tensor load_tensor(std::ifstream &in, const Device *device = &getCPU()) {
  return Tensor::load<T>(in, device);
}

template <typename T> DType_t get_dtype_if_tensor(const T &val) {
  if constexpr (std::is_convertible_v<T, tnn::Tensor>) {
    return val ? val->data_type() : DType_t::UNKNOWN;
  }
  return DType_t::UNKNOWN;
}

template <typename... Args> void check_all_match(DType_t expected, const Args &...args) {
  auto validator = [&](DType_t current) {
    if (current != DType_t::UNKNOWN && current != expected) {
      throw std::runtime_error("Tensor DType mismatch in operation!");
    }
  };
  (validator(get_dtype_if_tensor(args)), ...);
}

template <typename... Args> DType_t find_and_verify_dtype(const Args &...args) {
  DType_t found = DType_t::UNKNOWN;

  auto find_first = [&](DType_t current) {
    if (found == DType_t::UNKNOWN && current != DType_t::UNKNOWN) {
      found = current;
    }
  };
  (find_first(get_dtype_if_tensor(args)), ...);

  if (found == DType_t::UNKNOWN) {
    throw std::runtime_error("No Tensor found in arguments.");
  }

  check_all_match(found, args...);
  return found;
}

#define DISPATCH_ON_DTYPE(dtype_value, type_alias, ...)                                            \
  do {                                                                                             \
    switch (dtype_value) {                                                                         \
    case DType_t::FP32: {                                                                          \
      using type_alias = float;                                                                    \
      __VA_ARGS__;                                                                                 \
      break;                                                                                       \
    }                                                                                              \
    case DType_t::FP64: {                                                                          \
      using type_alias = double;                                                                   \
      __VA_ARGS__;                                                                                 \
      break;                                                                                       \
    }                                                                                              \
    case DType_t::FP16: {                                                                          \
      using type_alias = fp16;                                                                     \
      __VA_ARGS__;                                                                                 \
      break;                                                                                       \
    }                                                                                              \
    default:                                                                                       \
      throw std::runtime_error("Unknown dtype in dispatch");                                       \
    }                                                                                              \
  } while (0)

#define DISPATCH_AUTO(type_alias, func_body, ...)                                                  \
  DISPATCH_ON_DTYPE(tnn::find_and_verify_dtype(__VA_ARGS__), type_alias, func_body)

#define DISPATCH_AUTO_T(func, ...) DISPATCH_AUTO(T, func<T>(__VA_ARGS__), __VA_ARGS__)

} // namespace tnn