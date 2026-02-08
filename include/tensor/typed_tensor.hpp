#pragma once

#include <chrono>
#include <cstddef>
#include <fstream>
#include <sstream>

#include "device/device.hpp"
#include "device/device_manager.hpp"
#include "device/dptr.hpp"
#include "device/flow.hpp"
#include "device/iallocator.hpp"
#include "device/pool_allocator.hpp"
#include "device/sref.hpp"
#include "device/task.hpp"
#include "ops/ops.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tnn {
/**
 * @brief A tensor class dedicated for ML and DL applications.
 * @tparam T Data type (e.g., float, double, int)
 * Data layout is assumed to be row-major (C-style) by default.
 * Generic N-dimensional tensor with various functions. How layers interpret
 * the dimensions is up to the layer's implementation.
 */
template <typename T = float>
class TypedTensor : public ITensor {
protected:
  sref<IAllocator> allocator_;
  size_t data_size_;
  dptr data_;
  std::vector<size_t> shape_;

  inline size_t compute_stride(size_t index) const {
    size_t stride = 1;
    for (size_t i = index + 1; i < shape_.size(); ++i) {
      stride *= shape_[i];
    }
    return stride;
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

  dptr allocate_data(size_t size) {
    dptr data = allocator_->allocate(size * sizeof(T));
    return data;
  }

public:
  // Constructors and Destructor
  TypedTensor(IAllocator &allocator)
      : allocator_(allocator),
        data_size_(0) {
    for (size_t i = 0; i < shape_.size(); ++i) {
      shape_[i] = 0;
    }
    data_ = allocate_data(0);
  }

  TypedTensor(IAllocator &allocator, std::initializer_list<size_t> shape_list)
      : allocator_(allocator),
        shape_(shape_list) {
    data_size_ =
        std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    data_ = allocate_data(data_size_);
  }

  TypedTensor(IAllocator &allocator, std::initializer_list<size_t> shape_list, const dptr &data)
      : allocator_(allocator),
        shape_(shape_list) {
    data_size_ =
        std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    data_ = allocate_data(data_size_);
    if (data.get<T>() != nullptr) {
      ops::cd_copy<T>(data, data_, data_size_);
    }
  }

  TypedTensor(IAllocator &allocator, const std::vector<size_t> &shape)
      : allocator_(allocator),
        shape_(shape) {
    data_size_ =
        std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    data_ = allocate_data(data_size_);
  }

  TypedTensor(IAllocator &allocator, const std::vector<size_t> &shape, const dptr &data)
      : allocator_(allocator),
        shape_(shape) {
    data_size_ =
        std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    data_ = allocate_data(data_size_);
    if (data.get<T>() != nullptr) {
      ops::cd_copy<T>(data, data_, data_size_);
    }
  }

  TypedTensor(IAllocator &allocator, dptr &&data, const std::vector<size_t> &shape)
      : allocator_(allocator),
        data_(std::move(data)),
        shape_(shape) {
    data_size_ =
        std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
  }

  ~TypedTensor() = default;

  TypedTensor(const TypedTensor &other)
      : allocator_(other.allocator_),
        data_size_(other.data_size_),
        shape_(other.shape_) {
    if (data_size_ > 0) {
      data_ = allocate_data(data_size_);
      ops::copy<T>(other.data_, data_, data_size_);
    }
  }

  TypedTensor(TypedTensor &&other) noexcept
      : allocator_(other.allocator_),
        data_size_(other.data_size_),
        shape_(std::move(other.shape_)) {
    data_ = std::move(other.data_);
    other.data_size_ = 0;
  }

  T &operator()(std::initializer_list<size_t> indices) {
    assert(device().device_type() == DeviceType::CPU &&
           "Operator() is only available for CPU tensors");
    return data_.get<T>()[compute_index(indices)];
  }

  const T &operator()(std::initializer_list<size_t> indices) const {
    assert(device().device_type() == DeviceType::CPU &&
           "Operator() is only available for CPU tensors");
    return data_.get<T>()[compute_index(indices)];
  }

  void *data() override { return static_cast<void *>(data_.get<T>()); }
  const void *data() const override { return static_cast<const void *>(data_.get<T>()); }

  dptr data_ptr() override { return data_; }
  const dptr data_ptr() const override { return data_; }

  // Operators
  TypedTensor<T> &operator=(const TypedTensor<T> &other) = delete;

  TypedTensor<T> &operator=(TypedTensor<T> &&other) noexcept {
    if (this != &other) {
      allocator_ = other.allocator_;
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

    TypedTensor<T> result(allocator_, shape_);
    ops::add<T>(data_, other.data_, result.data_, data_size_);
    return result;
  }

  TypedTensor<T> operator-(const TypedTensor<T> &other) const {
    if (!same_shape(other)) {
      throw std::invalid_argument("TypedTensor shapes must match for subtraction");
    }

    TypedTensor<T> result(allocator_, shape_);
    ops::sub<T>(data_, other.data_, result.data_, data_size_);
    return result;
  }

  TypedTensor<T> operator*(const TypedTensor<T> &other) const {
    if (!same_shape(other)) {
      throw std::invalid_argument("TypedTensor shapes must match for element-wise multiplication");
    }

    TypedTensor<T> result(allocator_, shape_);
    ops::mul<T>(data_, other.data_, result.data_, data_size_);
    return result;
  }

  TypedTensor<T> operator/(const TypedTensor<T> &other) const {
    if (!same_shape(other)) {
      throw std::invalid_argument("TypedTensor shapes must match for element-wise division");
    }

    TypedTensor<T> result(allocator_, shape_);
    ops::div<T>(data_, other.data_, result.data_, data_size_);
    return result;
  }

  TypedTensor<T> operator+(T scalar) const {
    TypedTensor<T> result(allocator_, shape_);
    ops::add_scalar<T>(data_, scalar, result.data_, data_size_);
    return result;
  }

  TypedTensor<T> operator-(T scalar) const {
    TypedTensor<T> result(allocator_, shape_);
    ops::sub_scalar(data_, scalar, result.data_, data_size_);
    return result;
  }

  TypedTensor<T> operator*(T scalar) const {
    TypedTensor<T> result(allocator_, shape_);
    ops::mul_scalar(data_, scalar, result.data_, data_size_);
    return result;
  }

  TypedTensor<T> operator/(T scalar) const {
    if (scalar == T(0)) {
      throw std::invalid_argument("Division by zero");
    }

    TypedTensor<T> result(allocator_, shape_);
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

  IAllocator &allocator() const override { return allocator_; }

  const Device &device() const override { return data_.getDevice(); }

  DeviceType device_type() const override { return device().device_type(); }

  Tensor to_device(const Device &target_device) const override {
    if (device() == target_device) {
      return clone();
    }
    auto &allocator = PoolAllocator::instance(target_device, defaultFlowHandle);
    if (device_type() == DeviceType::CPU && target_device.device_type() == DeviceType::GPU) {
      std::vector<size_t> shape_vec(shape_);
      auto gpu_tensor = std::make_shared<TypedTensor<T>>(allocator, shape_vec);
      ops::cd_copy<T>(data_, gpu_tensor->data_, data_size_);
      return gpu_tensor;
    }
    if (device_type() == DeviceType::GPU && target_device.device_type() == DeviceType::CPU) {
      std::vector<size_t> shape_vec(shape_);
      std::shared_ptr<TypedTensor<T>> cpu_tensor =
          std::make_shared<TypedTensor<T>>(allocator, shape_vec);
      ops::cd_copy<T>(data_, cpu_tensor->data_, data_size_);
      return cpu_tensor;
    }
    throw std::runtime_error("Unsupported device type for to_device()");
  }

  Tensor clone() const override {
    return std::make_shared<TypedTensor<T>>(allocator_, shape_, data_);
  }

  Tensor span(std::vector<size_t> start_offset, std::vector<size_t> span_sizes) const override {
    if (start_offset.size() != shape_.size() || span_sizes.size() != shape_.size()) {
      throw std::invalid_argument("Span offsets and sizes must match tensor dimensions");
    }

    bool found_partial = false;
    for (size_t i = 0; i < shape_.size(); ++i) {
      if (start_offset[i] + span_sizes[i] > shape_[i]) {
        throw std::out_of_range("Span exceeds tensor dimensions");
      }
      bool is_partial = (start_offset[i] != 0) || (span_sizes[i] != shape_[i]);
      if (found_partial && is_partial) {
        throw std::invalid_argument(
            "Non-contiguous span: after a partial dimension, all subsequent dimensions "
            "must be complete (start_offset=0, span_size=shape[i])");
      }
      if (is_partial) {
        found_partial = true;
      }
    }

    size_t offset = 0;
    size_t span_size = 1;
    for (size_t i = 0; i < shape_.size(); ++i) {
      offset += start_offset[i] * compute_stride(i);
      span_size *= span_sizes[i];
    }
    dptr span_data = data_.span(offset * sizeof(T), span_size * sizeof(T));
    return std::make_shared<TypedTensor<T>>(allocator_, std::move(span_data), span_sizes);
  }

  std::unique_ptr<Task> fill(double value, flowHandle_t handle) override {
    return ops::set_scalar<T>(data_, static_cast<T>(value), data_size_, handle);
  }

  // Arithmetic operations returning shared_ptr
  void add(const ConstTensor &other) override {
    auto other_typed = std::dynamic_pointer_cast<const TypedTensor<T>>(other);
    if (!other_typed) {
      throw std::runtime_error("Type mismatch in tensor addition");
    }
    if (!same_shape(*other_typed)) {
      throw std::invalid_argument("Tensor shapes must match for addition");
    }
    ops::add<T>(data_, other_typed->data_, data_, data_size_);
  }

  void sub(const ConstTensor &other) override {
    auto other_typed = std::dynamic_pointer_cast<const TypedTensor<T>>(other);
    if (!other_typed) {
      throw std::runtime_error("Type mismatch in tensor subtraction");
    }
    if (!same_shape(*other_typed)) {
      throw std::invalid_argument("Tensor shapes must match for subtraction");
    }
    ops::sub<T>(data_, other_typed->data_, data_, data_size_);
  }

  void mul(const ConstTensor &other) override {
    auto other_typed = std::dynamic_pointer_cast<const TypedTensor<T>>(other);
    if (!other_typed) {
      throw std::runtime_error("Type mismatch in tensor multiplication");
    }
    if (!same_shape(*other_typed)) {
      throw std::invalid_argument("Tensor shapes must match for element-wise multiplication");
    }
    ops::mul<T>(data_, other_typed->data_, data_, data_size_);
  }

  void div(const ConstTensor &other) override {
    auto other_typed = std::dynamic_pointer_cast<const TypedTensor<T>>(other);
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
  void copy_to(const Tensor &target) const override {
    if (target == nullptr || target->size() < size()) {
      throw std::invalid_argument("Target tensor is null or smaller than source in tensor copy");
    }
    auto target_typed = std::dynamic_pointer_cast<TypedTensor<T>>(target);
    if (!target_typed) {
      throw std::runtime_error("Type mismatch in tensor copy");
    }
    ops::cd_copy<T>(data_, target_typed->data_, data_size_);
  }

  void resize(const std::vector<size_t> &new_shape) override {
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
  }

  /**
   * Similar to resize but only reallocates if the new size is larger than
   * the current allocated size. Good for caching.
   */
  void ensure(const std::vector<size_t> &new_shape) override {
    size_t new_size =
        std::accumulate(new_shape.begin(), new_shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_size * sizeof(T) > data_.capacity()) {
      data_ = allocate_data(new_size);
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
      throw std::runtime_error(
          "Cannot copy batch between tensors on different devices. Transfer "
          "to same device first.");
    }

    size_t batch_stride = compute_stride(0);
    size_t src_offset = src_batch_idx * other.compute_stride(0);
    size_t dest_offset = dest_batch_idx * batch_stride;

    ops::copy<T>(other.data_ + src_offset * sizeof(T), data_ + dest_offset * sizeof(T),
                 batch_stride);
  }

  double min() const override {
    auto cpu_tensor = std::dynamic_pointer_cast<TypedTensor<T>>(to_device(getCPU()));
    T min_val = cpu_tensor->data_.template get<T>()[0];
    for (size_t i = 1; i < cpu_tensor->data_size_; ++i) {
      if (cpu_tensor->data_.template get<T>()[i] < min_val) {
        min_val = cpu_tensor->data_.template get<T>()[i];
      }
    }
    return static_cast<double>(min_val);
  }

  double max() const override {
    auto cpu_tensor = std::dynamic_pointer_cast<TypedTensor<T>>(to_device(getCPU()));
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
    Tensor cpu_tensor = to_device(getCPU());
    size_t total_elements = cpu_tensor->size();
    std::cout << "TypedTensor data (shape " << cpu_tensor->shape_str() << "):\n";
    T *data = cpu_tensor->data_as<T>();
    for (size_t i = 0; i < total_elements; ++i) {
      std::cout << static_cast<float>(data[i]) << " ";
    }
    std::cout << std::endl;
  }

  void head(size_t n = 10) const override {
    Tensor cpu_tensor = to_device(getCPU());
    size_t total_elements = cpu_tensor->size();
    n = std::min(n, total_elements);
    std::cout << "TypedTensor head (first " << n << " elements of shape " << cpu_tensor->shape_str()
              << "):\n";
    T *data = cpu_tensor->data_as<T>();
    for (size_t i = 0; i < n; ++i) {
      std::cout << static_cast<float>(data[i]) << " ";
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
      device().copyToHost(host_buffer.data(), data_.get<T>(), data_size_ * sizeof(T));
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
    } else if (std::is_same<T, bf16>::value) {
      return DType_t::BF16;
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
}  // namespace tnn