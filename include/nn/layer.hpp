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

struct Cache {
  struct CacheKey {
    size_t mb_id;
    std::string key;

    bool operator==(const CacheKey &other) const {
      return mb_id == other.mb_id && key == other.key;
    }
  };

  struct CacheKeyHash {
    size_t operator()(const CacheKey &cache_key) const noexcept {
      size_t seed = std::hash<size_t>{}(cache_key.mb_id);
      seed ^= std::hash<std::string>{}(cache_key.key) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      return seed;
    }
  };

  std::unordered_map<CacheKey, Tensor, CacheKeyHash> cache;

  void set(size_t mb_id, const std::string &key, const Tensor &value) {
    cache[{mb_id, key}] = value;
  }

  Tensor &get(size_t mb_id, const std::string &key) {
    auto it = cache.find({mb_id, key});
    if (it == cache.end()) {
      throw std::runtime_error(fmt::format("Cache miss for key: {} (mb_id={})", key, mb_id));
    }
    return it->second;
  }

  void clear(size_t mb_id) {
    for (auto it = cache.begin(); it != cache.end();) {
      if (it->first.mb_id == mb_id) {
        it = cache.erase(it);
      } else {
        ++it;
      }
    }
  }
};

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
  Vec<Tensor> forward(const Vec<ConstTensor> &inputs, size_t mb_id = 0);
  Vec<Tensor> backward(const Vec<ConstTensor> &grad_outputs, size_t mb_id = 0);

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
  void save_state(std::ofstream &file);
  virtual Vec<ParamDescriptor> param_descriptors() { return {}; }
  virtual std::string type() const = 0;
  virtual LayerConfig get_config() const = 0;

  Vec<Tensor> parameters();
  Vec<Tensor> gradients();
  void clear_cache(size_t mb_id);

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
  virtual Vec<Tensor> forward_impl(const Vec<ConstTensor> &inputs, size_t mb_id) = 0;
  virtual Vec<Tensor> backward_impl(const Vec<ConstTensor> &grad_outputs, size_t mb_id) = 0;

protected:
  bool initialized_ = false;
  EngineType engine_type_ = EngineType::UNKNOWN;
  DELAllocatorV2 *allocator_ = nullptr;
  bool is_training_ = true;
  bool is_fwd_ = false;
  bool use_seed_ = false;
  unsigned long long srand_seed_ = 0;
  std::map<std::pair<size_t, std::string>, ConstTensor> immutable_cache_;
  std::map<std::pair<size_t, std::string>, Tensor> mutable_cache_;
  flowHandle_t flow_handle_;
  std::string name_;
  DType_t io_dtype_ = DType_t::FP32;       // data type for input/output tensors
  DType_t param_dtype_ = DType_t::FP32;    // data type for parameters/gradients
  DType_t compute_dtype_ = DType_t::FP32;  // data type for internal computations

  // helpers
  void set_immutable_cache(size_t mb_id, const std::string &key, ConstTensor value);
  ConstTensor &get_immutable_cache(size_t mb_id, const std::string &key);
  void set_mutable_cache(size_t mb_id, const std::string &key, Tensor value);
  Tensor &get_mutable_cache(size_t mb_id, const std::string &key);
  Tensor get_tensor(const Vec<size_t> &shape, DType_t dtype);
};

template <typename LayerType>
class LayerRef {
public:
  using impl_type = LayerType;

  template <typename>
  friend class LayerRef;

  LayerRef() = default;

  LayerRef(std::shared_ptr<LayerType> layer)
      : layer_(layer) {}

  template <typename U, typename = std::enable_if_t<std::is_convertible_v<U *, LayerType *>>>
  LayerRef(std::shared_ptr<U> layer)
      : layer_(std::move(layer)) {}

  template <typename U, typename = std::enable_if_t<std::is_convertible_v<U *, LayerType *>>>
  LayerRef(const LayerRef<U> &other)
      : layer_(std::static_pointer_cast<LayerType>(other.layer_)) {}

  template <typename U, typename = std::enable_if_t<std::is_convertible_v<U *, LayerType *>>>
  LayerRef(LayerRef<U> &&other)
      : layer_(std::static_pointer_cast<LayerType>(std::move(other.layer_))) {}

  template <typename T, typename U = typename std::decay_t<T>::impl_type,
            std::enable_if_t<std::is_convertible_v<U *, LayerType *> &&
                                 std::is_base_of_v<LayerRef<U>, std::decay_t<T>> &&
                                 !std::is_same_v<std::decay_t<T>, LayerRef<U>>,
                             int> = 0>
  LayerRef(T &&other)
      : layer_(
            std::static_pointer_cast<LayerType>(static_cast<const LayerRef<U> &>(other).layer_)) {}

  template <typename... Args>
  LayerRef(Args &&...args)
      : layer_(std::make_shared<LayerType>(std::forward<Args>(args)...)) {}

  LayerType *operator->() const { return layer_.get(); }
  LayerType &operator*() const { return *layer_; }

  operator std::shared_ptr<LayerType>() const { return layer_; }
  LayerType *get() const { return layer_.get(); }
  LayerType *release() { return layer_.release(); }

  explicit operator bool() const { return layer_ != nullptr; }
  bool operator!() const { return layer_ == nullptr; }

  bool operator==(const LayerRef &other) const { return layer_ == other.layer_; }
  bool operator!=(const LayerRef &other) const { return layer_ != other.layer_; }

  template <typename U>
  bool is() const {
    return std::dynamic_pointer_cast<U>(layer_) != nullptr;
  }

  template <typename U>
  auto as() const -> LayerRef<U> {
    auto casted = std::dynamic_pointer_cast<U>(layer_);
    if (!casted) {
      throw std::runtime_error("LayerRef: incompatible layer cast");
    }
    return LayerRef<U>(std::move(casted));
  }

  template <typename... Args>
  decltype(auto) operator()(Args &&...args) const {
    if (!layer_) {
      throw std::runtime_error("LayerRef: underlying shared_ptr is null");
    }
    return (*layer_)(std::forward<Args>(args)...);
  }

protected:
  std::shared_ptr<LayerType> layer_;
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