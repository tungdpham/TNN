/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/layer.hpp"

#include "logging/logger.hpp"

namespace tnn {

template <typename T>
T LayerConfig::get(const std::string &key, const T &default_value) const {
  auto it = parameters.find(key);
  if (it != parameters.end()) {
    try {
      return std::any_cast<T>(it->second);
    } catch (const std::bad_any_cast &) {
      std::cerr << "Warning: LayerConfig parameter '" << key
                << "' type mismatch. Returning default value." << std::endl;
      return default_value;
    }
  }
  return default_value;
}

template size_t LayerConfig::get<size_t>(const std::string &, const size_t &) const;
template int LayerConfig::get<int>(const std::string &, const int &) const;
template long LayerConfig::get<long>(const std::string &, const long &) const;
template long long LayerConfig::get<long long>(const std::string &, const long long &) const;
template unsigned int LayerConfig::get<unsigned int>(const std::string &,
                                                     const unsigned int &) const;
template unsigned long long LayerConfig::get<unsigned long long>(const std::string &,
                                                                 const unsigned long long &) const;
template float LayerConfig::get<float>(const std::string &, const float &) const;
template double LayerConfig::get<double>(const std::string &, const double &) const;
template bool LayerConfig::get<bool>(const std::string &, const bool &) const;
template std::string LayerConfig::get<std::string>(const std::string &, const std::string &) const;
template nlohmann::json LayerConfig::get<nlohmann::json>(const std::string &,
                                                         const nlohmann::json &) const;

nlohmann::json LayerConfig::to_json() const {
  nlohmann::json j;
  j["name"] = name;
  j["type"] = type;
  nlohmann::json param_json;
  for (const auto &[key, value] : parameters) {
    if (value.type() == typeid(size_t)) {
      param_json[key] = std::any_cast<size_t>(value);
    } else if (value.type() == typeid(int)) {
      param_json[key] = std::any_cast<int>(value);
    } else if (value.type() == typeid(long)) {
      param_json[key] = std::any_cast<long>(value);
    } else if (value.type() == typeid(long long)) {
      param_json[key] = std::any_cast<long long>(value);
    } else if (value.type() == typeid(unsigned int)) {
      param_json[key] = std::any_cast<unsigned int>(value);
    } else if (value.type() == typeid(unsigned long)) {
      param_json[key] = std::any_cast<unsigned long>(value);
    } else if (value.type() == typeid(unsigned long long)) {
      param_json[key] = std::any_cast<unsigned long long>(value);
    } else if (value.type() == typeid(float)) {
      param_json[key] = std::any_cast<float>(value);
    } else if (value.type() == typeid(double)) {
      param_json[key] = std::any_cast<double>(value);
    } else if (value.type() == typeid(bool)) {
      param_json[key] = std::any_cast<bool>(value);
    } else if (value.type() == typeid(std::string)) {
      param_json[key] = std::any_cast<std::string>(value);
    } else if (value.type() == typeid(nlohmann::json)) {
      param_json[key] = std::any_cast<nlohmann::json>(value);
    } else if (value.type() == typeid(std::vector<nlohmann::json>)) {
      param_json[key] = std::any_cast<std::vector<nlohmann::json>>(value);
    }
  }
  j["parameters"] = param_json;
  return j;
}

LayerConfig LayerConfig::from_json(const nlohmann::json &j) {
  LayerConfig config;
  config.name = j.value("name", "");
  config.type = j.value("type", "");
  nlohmann::json param_json = j.value("parameters", nlohmann::json::object());
  if (j.contains("parameters") && !param_json.is_null()) {
    for (const auto &[key, value] : param_json.items()) {
      if (value.is_number_integer()) {
        int64_t int_val = value.template get<int64_t>();
        if (int_val >= 0) {
          config.parameters[key] = static_cast<size_t>(int_val);
        } else {
          config.parameters[key] = static_cast<int>(int_val);
        }
      } else if (value.is_number_float()) {
        config.parameters[key] = value.template get<float>();
      } else if (value.is_boolean()) {
        config.parameters[key] = value.template get<bool>();
      } else if (value.is_string()) {
        config.parameters[key] = value.template get<std::string>();
      } else if (value.is_array() || value.is_object()) {
        config.parameters[key] = value;
      }
    }
  }
  return config;
}

Layer::Layer() {
  this->device_ = getCPU();
  this->mem_pool_ = &PoolAllocator::instance(*device_);
}

void Layer::set_device(const Device &device) {
  device_ = device;
  mem_pool_ = &PoolAllocator::instance(*device_);
  on_set_device(device);
}

const Device &Layer::get_device() const { return device_; }

void Layer::set_io_dtype(DType_t dtype) {
  io_dtype_ = dtype;
  on_set_io_dtype(dtype);
}

DType_t Layer::get_io_dtype() const { return io_dtype_; }

void Layer::set_param_dtype(DType_t dtype) {
  param_dtype_ = dtype;
  on_set_param_dtype(dtype);
}

DType_t Layer::get_param_dtype() const { return param_dtype_; }

void Layer::set_compute_dtype(DType_t dtype) {
  compute_dtype_ = dtype;
  on_set_compute_dtype(dtype);
}

DType_t Layer::get_compute_dtype() const { return compute_dtype_; }

void Layer::init() {
  if (initialized_) {
    return;
  }
  init_impl();
  initialized_ = true;
}

void Layer::set_seed(unsigned long long seed) {
  use_seed_ = true;
  srand_seed_ = seed;
  on_set_seed(seed);
}

void Layer::forward(const ConstTensor &input, const Tensor &output, size_t mb_id) {
  if (!initialized_) {
    std::cerr << "Warning: Layer " << name_ << " is not initialized. Call init() before forward."
              << std::endl;
    return;
  }
  Clock::time_point start_time = Clock::now();
  if (input == nullptr || output == nullptr) {
    throw std::runtime_error("Layer " + name_ + " received null IO tensor.");
  }
  if (input->data_type() != io_dtype_) {
    throw std::runtime_error("Layer " + name_ +
                             " input tensor dtype does not match layer io_dtype.");
  }
  if (output->data_type() != io_dtype_) {
    throw std::runtime_error("Layer " + name_ +
                             " output tensor dtype does not match layer io_dtype.");
  }
  ConstTensor current = input;
  Tensor device_input;
  if (input->device() != this->device_) {
    device_input = this->get_buffer(input->shape(), input->data_type());
    input->copy_to(device_input);
    current = device_input;
  }
  if (output->device() != this->device_) {
    throw std::runtime_error("Layer " + name_ +
                             " output tensor device does not match layer device.");
  }
  forward_impl(current, output, mb_id);
#ifndef NDEBUG
  this->device_->getFlow("default")->synchronize();
#endif
  Clock::time_point end_time = Clock::now();
  profiler_.add_event(Event{EventType::COMPUTE, start_time, end_time, "forward"});
}

void Layer::backward(const ConstTensor &gradient, const Tensor &grad_input, size_t mb_id) {
  if (!initialized_) {
    std::cerr << "Warning: Layer " << name_ << " is not initialized. Call init() before backward."
              << std::endl;
    return;
  }
  auto start_time = Clock::now();
  if (gradient == nullptr || grad_input == nullptr) {
    throw std::runtime_error("Layer " + name_ + " received null gradient or grad_input tensor.");
  }
  if (gradient->data_type() != io_dtype_) {
    throw std::runtime_error("Layer " + name_ +
                             " gradient tensor dtype does not match layer io_dtype.");
  }
  if (grad_input->data_type() != io_dtype_) {
    throw std::runtime_error("Layer " + name_ +
                             " grad_input tensor dtype does not match layer io_dtype.");
  }
  ConstTensor current_gradient = gradient;
  Tensor device_gradient;
  if (gradient->device() != this->device_) {
    device_gradient = this->get_buffer(gradient->shape(), gradient->data_type());
    gradient->copy_to(device_gradient);
    current_gradient = device_gradient;
  }
  if (grad_input->device() != this->device_) {
    throw std::runtime_error("Layer " + name_ +
                             " grad_input tensor device does not match layer device.");
  }
  backward_impl(current_gradient, grad_input, mb_id);
#ifndef NDEBUG
  this->device_->getFlow("default")->synchronize();
#endif
  clear_cache(mb_id);
  Clock::time_point end_time = Clock::now();
  profiler_.add_event(Event{EventType::COMPUTE, start_time, end_time, "backward"});
}

std::vector<Tensor> Layer::parameters() { return {}; }

std::vector<Tensor> Layer::gradients() { return {}; }

void Layer::save_state(std::ofstream &file) {
  auto config = get_config();
  nlohmann::json j = config.to_json();
  std::string j_str = j.dump();
  size_t j_size = j_str.size();
  file.write(reinterpret_cast<const char *>(&j_size), sizeof(size_t));
  file.write(j_str.c_str(), j_size);
  auto params = parameters();
  for (const auto &param : params) {
    param->save(file);
  }
}

void Layer::set_training(bool training) {
  is_training_ = training;
  on_set_training(training);
}

bool Layer::is_training() const { return is_training_; }

void Layer::enable_profiling(bool enable) { enable_profiling_ = enable; }

bool Layer::is_profiling_enabled() const { return enable_profiling_; }

void Layer::print_profiling_info() const {
  const auto &events = profiler_.get_events();
  std::string header = "Profiling info for Layer: " + name_;
  std::string format_str = "{:=^70}\n";
  std::string output = fmt::format(fmt::runtime(format_str), header);
  output += fmt::format(fmt::runtime("{:<20}{:<25}{:<25}\n"), "Event", "Duration (ms)", "");
  output += fmt::format(fmt::runtime("{:-^70}\n"), "");
  for (const auto &event : events) {
    float duration_ms =
        Time::duration_cast<Time::microseconds>(event.end_time - event.start_time).count() /
        1000.0f;
    output += fmt::format(fmt::runtime("{:<20}{:<25.3f}\n"), event.name, duration_ms);
  }
  output += fmt::format(fmt::runtime("{:=^70}"), "");
  GlobalLogger::info(output);
}

void Layer::set_mem_pool(PoolAllocator *mem_pool) { mem_pool_ = mem_pool; }

const PoolAllocator *Layer::get_mem_pool() const { return mem_pool_; }

size_t Layer::nbytes_params() {
  size_t total = 0;
  auto params = this->parameters();
  for (const auto &param : params) {
    size_t dtype_size = get_dtype_size(param->data_type());
    total += param->capacity() * dtype_size;
  }
  auto grads = this->gradients();
  for (const auto &grad : grads) {
    size_t dtype_size = get_dtype_size(grad->data_type());
    total += grad->capacity() * dtype_size;
  }
  return total;
}

size_t Layer::cached_memory_bytes() const {
  size_t total = 0;
  for (auto &[key, tensor] : cached_tensors_) {
    if (tensor) {
      size_t dtype_size = get_dtype_size(tensor->data_type());
      total += tensor->capacity() * dtype_size;
    }
  }
  return total;
}

void Layer::reset_profiling_info() { profiler_.reset(); }

std::string Layer::name() const { return name_; }

Tensor Layer::make_param_tensor(std::vector<size_t> shape) {
  return make_tensor(param_dtype_, shape, this->device_);
}

Tensor Layer::make_io_tensor(std::vector<size_t> shape) {
  return make_tensor(io_dtype_, shape, this->device_);
}

Tensor Layer::make_compute_tensor(std::vector<size_t> shape) {
  return make_tensor(compute_dtype_, shape, this->device_);
}

ConstTensor &Layer::get_cached_tensor(size_t mb_id, const std::string &key) {
  return cached_tensors_[{mb_id, key}];
}

Tensor &Layer::get_mutable_tensor(size_t mb_id, const std::string &key) {
  return mutable_tensors_[{mb_id, key}];
}

Tensor Layer::get_buffer(const std::vector<size_t> &shape, DType_t dtype) {
  if (!mem_pool_) {
    throw std::runtime_error("PoolAllocator not set for layer: " + name_);
  }
  return make_tensor(*mem_pool_, dtype, shape);
}

void Layer::clear_cache(size_t mb_id) {
  for (auto it = cached_tensors_.begin(); it != cached_tensors_.end();) {
    if (it->first.first == mb_id) {
      it = cached_tensors_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace tnn
