/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <fmt/core.h>

#include <cstddef>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "common/config.hpp"
#include "device/del_allocator_v2.hpp"
#include "device/engine.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace synet {
using LayerConfig = TConfig;

struct ParamDescriptor {
  DType_t dtype;  // data type of the parameter
  Vec<size_t> shape;
  Tensor *data_ptr;  // pointer to the actual param
  Tensor *grad_ptr;  // pointer to the actual grad_output
};

inline size_t get_shapes_bytes(const Vec<Vec<size_t>> &shapes, DType_t dtype) {
  size_t total_bytes = 0;
  size_t dtype_size = get_dtype_size(dtype);
  for (const auto &shape : shapes) {
    size_t shape_bytes =
        std::accumulate(shape.begin(), shape.end(), dtype_size, std::multiplies<size_t>());
    shape_bytes = align_up(shape_bytes, 256);
    total_bytes += shape_bytes;
  }
  return total_bytes;
}

namespace detail {
struct ResidualsMap {
public:
  using ResidualValue = std::variant<std::monostate, std::map<std::string, ResidualsMap>, Tensor>;

  ResidualValue data_;

  ResidualsMap()
      : data_(std::monostate{}) {}
  ~ResidualsMap() = default;
  ResidualsMap(const ResidualsMap &) = delete;
  ResidualsMap &operator=(const ResidualsMap &) = delete;
  ResidualsMap(ResidualsMap &&) = default;
  ResidualsMap &operator=(ResidualsMap &&) = default;

  ResidualsMap &operator[](const std::string &key) {
    if (data_.index() == 0) {
      data_ = std::map<std::string, ResidualsMap>{};
    } else if (data_.index() == 1) {
      // already a map, do nothing
    } else if (data_.index() == 2) {
      throw std::runtime_error("ResidualsMap: Attempting to index into a leaf node");
    }

    return std::get<1>(data_)[key];
  }

  ResidualsMap &operator=(const Tensor &tensor) {
    if (data_.index() == 0) {
      data_ = tensor;
    } else if (data_.index() == 1) {
      throw std::runtime_error("ResidualsMap: Attempting to assign a Tensor to a non-leaf node");
    }
    return *this;
  }

  operator Tensor &() {
    if (data_.index() != 2) {
      throw std::runtime_error("ResidualsMap: Attempting to convert a non-leaf node to Tensor");
    }
    return std::get<2>(data_);
  }
};

}  // namespace detail

using Residuals = detail::ResidualsMap;

// Single input/output layer interface. Can be easily extended to multiple inputs/outputs later if
// needed.
class LayerImpl : public virtual std::enable_shared_from_this<LayerImpl> {
public:
  LayerImpl() = default;
  LayerImpl(const std::string &name)
      : name_(name) {}

  virtual ~LayerImpl() = default;

  void set_engine_type(EngineType engine_type);
  EngineType get_engine_type() const;

  void init();

  Vec<Tensor> forward(const Vec<Tensor> &inputs);
  Vec<Tensor> forward(const Vec<Tensor> &inputs, Residuals &residuals);
  Vec<Tensor> backward(const Vec<Tensor> &grad_outputs, Residuals &residuals);

  // Note: have to call init again after changing param dtype
  LayerImpl &set_allocator(DELAllocatorV2 &allocator);
  DELAllocatorV2 *get_allocator() const;
  LayerImpl &set_flow_handle(flowHandle_t handle);
  flowHandle_t get_flow_handle() const;
  LayerImpl &set_seed(unsigned long long seed);
  LayerImpl &set_io_dtype(DType_t dtype);
  DType_t get_io_dtype() const;
  LayerImpl &set_param_dtype(DType_t dtype);
  DType_t get_param_dtype() const;
  LayerImpl &set_compute_dtype(DType_t dtype);
  DType_t get_compute_dtype() const;
  LayerImpl &set_training(bool training);
  bool is_training() const;

  virtual Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const = 0;
  std::string name() const { return name_; }
  void save_state(std::ostream &out) const;
  virtual Vec<ParamDescriptor> param_descriptors() { return {}; }
  const Vec<ParamDescriptor> param_descriptors() const {
    return const_cast<LayerImpl *>(this)->param_descriptors();
  }
  virtual std::string type() const = 0;
  virtual LayerConfig get_config() const = 0;

  Vec<Tensor *> parameters() {
    Vec<Tensor *> params;
    for (const auto &desc : param_descriptors()) {
      params.push_back(desc.data_ptr);
    }
    return params;
  }
  Vec<Tensor *> gradients() {
    Vec<Tensor *> grads;
    for (const auto &desc : param_descriptors()) {
      grads.push_back(desc.grad_ptr);
    }
    return grads;
  }

  const Device &device() const {
    if (!allocator_) {
      throw std::runtime_error("LayerImpl: Allocator is not set to get device.");
    }
    return allocator_->device();
  }

protected:
  virtual void on_set_engine_type(EngineType engine_type) {}
  virtual void init_impl() {}
  virtual void on_set_allocator(DELAllocatorV2 &allocator) {}
  virtual void on_set_flow_handle(flowHandle_t handle) {}
  virtual void on_set_seed(unsigned long long seed) {}
  virtual void on_set_training(bool training) {}
  virtual void on_set_io_dtype(DType_t dtype) {}
  virtual void on_set_param_dtype(DType_t dtype) {}
  virtual void on_set_compute_dtype(DType_t dtype) {}
  virtual Vec<Tensor> forward_impl(const Vec<Tensor> &inputs, Residuals &residuals) = 0;
  virtual Vec<Tensor> backward_impl(const Vec<Tensor> &grad_outputs, Residuals &residuals) = 0;

protected:
  bool initialized_ = false;
  EngineType engine_type_ = EngineType::UNKNOWN;
  DELAllocatorV2 *allocator_ = nullptr;
  bool is_training_ = true;
  bool is_fwd_ = false;
  bool use_seed_ = false;
  unsigned long long srand_seed_ = 0;
  flowHandle_t flow_handle_;
  std::string name_;
  DType_t io_dtype_ = DType_t::FP32;       // data type for input/output tensors
  DType_t param_dtype_ = DType_t::FP32;    // data type for parameters/gradients
  DType_t compute_dtype_ = DType_t::FP32;  // data type for internal computations

  // helpers
  Tensor get_tensor(const Vec<size_t> &shape, DType_t dtype);
};

template <typename LayerType>
class LayerRef {
public:
  using impl_type = LayerType;

  template <typename>
  friend class LayerRef;

  LayerRef() = default;

  LayerRef(std::nullptr_t)
      : impl_(nullptr) {}

  LayerRef(std::shared_ptr<LayerType> layer)
      : impl_(layer) {}

  template <typename U, typename = std::enable_if_t<std::is_convertible_v<U *, LayerType *>>>
  LayerRef(std::shared_ptr<U> layer)
      : impl_(std::move(layer)) {}

  template <typename U, typename = std::enable_if_t<std::is_convertible_v<U *, LayerType *>>>
  LayerRef(const LayerRef<U> &other)
      : impl_(std::static_pointer_cast<LayerType>(other.impl_)) {}

  template <typename U, typename = std::enable_if_t<std::is_convertible_v<U *, LayerType *>>>
  LayerRef(LayerRef<U> &&other)
      : impl_(std::static_pointer_cast<LayerType>(std::move(other.impl_))) {}

  template <typename T, typename U = typename std::decay_t<T>::impl_type,
            std::enable_if_t<std::is_convertible_v<U *, LayerType *> &&
                                 std::is_base_of_v<LayerRef<U>, std::decay_t<T>> &&
                                 !std::is_same_v<std::decay_t<T>, LayerRef<U>>,
                             int> = 0>
  LayerRef(T &&other)
      : impl_(std::static_pointer_cast<LayerType>(static_cast<const LayerRef<U> &>(other).impl_)) {}

  template <typename... Args>
  LayerRef(Args &&...args)
      : impl_(std::make_shared<LayerType>(std::forward<Args>(args)...)) {}

  LayerType &operator*() const { return *impl_; }

  operator std::shared_ptr<LayerType>() const { return impl_; }
  LayerType *get() const { return impl_.get(); }
  LayerType *release() { return impl_.release(); }

  explicit operator bool() const { return impl_ != nullptr; }
  bool operator!() const { return impl_ == nullptr; }

  bool operator==(const LayerRef &other) const { return impl_ == other.impl_; }
  bool operator!=(const LayerRef &other) const { return impl_ != other.impl_; }

  template <typename U>
  bool is() const {
    return std::dynamic_pointer_cast<U>(impl_) != nullptr;
  }

  template <typename U>
  auto as() const -> LayerRef<U> {
    auto casted = std::dynamic_pointer_cast<U>(impl_);
    if (!casted) {
      throw std::runtime_error("LayerRef: incompatible layer cast");
    }
    return LayerRef<U>(std::move(casted));
  }

  template <typename... Args>
  decltype(auto) operator()(Args &&...args) const {
    check_layer("operator()");
    return (*impl_)(std::forward<Args>(args)...);
  }

  void set_engine_type(EngineType engine_type) {
    check_layer("set_engine_type");
    impl_->set_engine_type(engine_type);
  }

  EngineType get_engine_type() const {
    check_layer("get_engine_type");
    return impl_->get_engine_type();
  }

  void init() {
    check_layer("init");
    impl_->init();
  }

  Vec<Tensor> forward(const Vec<Tensor> &inputs) {
    check_layer("forward");
    return impl_->forward(inputs);
  }

  Vec<Tensor> forward(const Vec<Tensor> &inputs, Residuals &residuals) {
    check_layer("forward");
    return impl_->forward(inputs, residuals);
  }

  Vec<Tensor> backward(const Vec<Tensor> &grad_outputs, Residuals &residuals) {
    check_layer("backward");
    return impl_->backward(grad_outputs, residuals);
  }

  LayerRef &set_allocator(DELAllocatorV2 &allocator) {
    check_layer("set_allocator");
    impl_->set_allocator(allocator);
    return *this;
  }

  DELAllocatorV2 *get_allocator() const {
    check_layer("get_allocator");
    return impl_->get_allocator();
  }

  LayerRef &set_flow_handle(flowHandle_t handle) {
    check_layer("set_flow_handle");
    impl_->set_flow_handle(handle);
    return *this;
  }

  flowHandle_t get_flow_handle() const {
    check_layer("get_flow_handle");
    return impl_->get_flow_handle();
  }

  LayerRef &set_seed(unsigned long long seed) {
    check_layer("set_seed");
    impl_->set_seed(seed);
    return *this;
  }

  LayerRef &set_io_dtype(DType_t dtype) {
    check_layer("set_io_dtype");
    impl_->set_io_dtype(dtype);
    return *this;
  }

  DType_t get_io_dtype() const {
    check_layer("get_io_dtype");
    return impl_->get_io_dtype();
  }

  LayerRef &set_param_dtype(DType_t dtype) {
    check_layer("set_param_dtype");
    impl_->set_param_dtype(dtype);
    return *this;
  }

  DType_t get_param_dtype() const {
    check_layer("get_param_dtype");
    return impl_->get_param_dtype();
  }

  LayerRef &set_compute_dtype(DType_t dtype) {
    check_layer("set_compute_dtype");
    impl_->set_compute_dtype(dtype);
    return *this;
  }

  DType_t get_compute_dtype() const {
    check_layer("get_compute_dtype");
    return impl_->get_compute_dtype();
  }

  LayerRef &set_training(bool training) {
    check_layer("set_training");
    impl_->set_training(training);
    return *this;
  }

  bool is_training() const {
    check_layer("is_training");
    return impl_->is_training();
  }

  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
    check_layer("output_shapes");
    return impl_->output_shapes(input_shapes);
  }

  std::string name() const {
    check_layer("name");
    return impl_->name();
  }

  void save_state(std::ofstream &file) {
    check_layer("save_state");
    impl_->save_state(file);
  }

  Vec<ParamDescriptor> param_descriptors() {
    check_layer("param_descriptors");
    return impl_->param_descriptors();
  }

  const std::vector<ParamDescriptor> param_descriptors() const {
    check_layer("param_descriptors");
    return impl_->param_descriptors();
  }

  std::string type() const {
    check_layer("type");
    return impl_->type();
  }

  LayerConfig get_config() const {
    check_layer("get_config");
    return impl_->get_config();
  }

  Vec<Tensor *> parameters() {
    check_layer("parameters");
    return impl_->parameters();
  }

  Vec<Tensor *> gradients() {
    check_layer("gradients");
    return impl_->gradients();
  }

  const Device &device() const {
    check_layer("device");
    return impl_->device();
  }

  static LayerRef<LayerType> create_from_config(const LayerConfig &config) {
    std::shared_ptr<LayerType> layer = LayerType::create_from_config(config);
    if (!layer) {
      throw std::runtime_error("LayerRef: create_from_config returned nullptr");
    }
    return LayerRef<LayerType>(std::move(layer));
  }

protected:
  std::shared_ptr<LayerType> impl_;

private:
  void check_layer(const char *method_name) const {
    if (!impl_) {
      throw std::runtime_error(
          fmt::format("LayerRef {}: underlying shared_ptr is null", method_name));
    }
  }
};

class Layer : public LayerRef<LayerImpl> {
public:
  using LayerRef<LayerImpl>::LayerRef;
};

template <typename LayerType, typename... Args>
auto make_layer(Args &&...args) -> LayerRef<LayerType> {
  return LayerRef<LayerType>(std::make_shared<LayerType>(std::forward<Args>(args)...));
}

#define DISPATCH_IO_DTYPE(method_name, ...)                                \
  do {                                                                     \
    DISPATCH_DTYPE(this->io_dtype_, IO_T, method_name<IO_T>(__VA_ARGS__)); \
  } while (0)

#define DISPATCH_ON_3_DTYPES_TO_METHOD(method_name, ...)                                   \
  do {                                                                                     \
    DISPATCH_DTYPE(                                                                        \
        this->io_dtype_, IO_T,                                                             \
        DISPATCH_DTYPE(this->param_dtype_, PARAM_T,                                        \
                       DISPATCH_DTYPE(this->compute_dtype_, COMP_T,                        \
                                      method_name<IO_T, PARAM_T, COMP_T>(__VA_ARGS__);))); \
  } while (0)
}  // namespace synet