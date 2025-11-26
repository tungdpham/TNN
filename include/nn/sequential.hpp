/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "blocks.hpp"
#include "device/device_manager.hpp"
#include "device/device_type.hpp"
#include "device/task.hpp"
#include "layers.hpp"
#include "loss.hpp"
#include "nn/layers_impl/base_layer.hpp"
#include "optimizers.hpp"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tnn {
struct Partition {
  size_t start_layer;
  size_t end_layer; // exclusive

  Partition(size_t start, size_t end) : start_layer(start), end_layer(end) {}
};

template <typename T = float> class Sequential {
private:
  std::vector<std::unique_ptr<Layer<T>>> layers_;
  std::string name_;
  std::unique_ptr<Optimizer<T>> optimizer_ = nullptr;
  std::unique_ptr<Loss<T>> loss_ = nullptr;
  const Device *device_ = nullptr;
  bool is_training_;

  bool enable_profiling_ = false;

  mutable std::mutex forward_times_mutex_;
  mutable std::mutex backward_times_mutex_;
  std::map<std::string, uint64_t> forward_times_microseconds_;
  std::map<std::string, uint64_t> backward_times_microseconds_;

public:
  explicit Sequential(const std::string &name = "sequential")
      : name_(name), is_training_(true), enable_profiling_(false) {}

  Sequential(const Sequential &other)
      : name_(other.name_), is_training_(other.is_training_),
        enable_profiling_(other.enable_profiling_) {
    for (const auto &layer : other.layers_) {
      layers_.push_back(layer->clone());
    }
    if (other.optimizer_) {
      optimizer_ = other.optimizer_->clone();
    }
    if (other.loss_) {
      loss_ = other.loss_->clone();
    }

    // Thread-safe copy of timing maps
    {
      std::lock_guard<std::mutex> forward_lock(other.forward_times_mutex_);
      forward_times_microseconds_ = other.forward_times_microseconds_;
    }
    {
      std::lock_guard<std::mutex> backward_lock(other.backward_times_mutex_);
      backward_times_microseconds_ = other.backward_times_microseconds_;
    }
  }

  Sequential(Sequential &&other) noexcept
      : layers_(std::move(other.layers_)), name_(std::move(other.name_)),
        optimizer_(std::move(other.optimizer_)), loss_(std::move(other.loss_)),
        is_training_(other.is_training_), enable_profiling_(other.enable_profiling_),
        forward_times_microseconds_(std::move(other.forward_times_microseconds_)),
        backward_times_microseconds_(std::move(other.backward_times_microseconds_)) {}

  Sequential &operator=(const Sequential &other) {
    if (this != &other) {
      layers_.clear();
      for (const auto &layer : other.layers_) {
        layers_.push_back(layer->clone());
      }
      optimizer_ = other.optimizer_ ? other.optimizer_->clone() : nullptr;
      loss_ = other.loss_ ? other.loss_->clone() : nullptr;
      name_ = other.name_;
      is_training_ = other.is_training_;
      enable_profiling_ = other.enable_profiling_;

      // Thread-safe clearing and copying of timing maps
      {
        std::lock_guard<std::mutex> forward_lock(forward_times_mutex_);
        forward_times_microseconds_.clear();
      }
      {
        std::lock_guard<std::mutex> backward_lock(backward_times_mutex_);
        backward_times_microseconds_.clear();
      }

      // Copy timing maps from other with proper locking
      {
        std::lock_guard<std::mutex> other_forward_lock(other.forward_times_mutex_);
        std::lock_guard<std::mutex> this_forward_lock(forward_times_mutex_);
        forward_times_microseconds_ = other.forward_times_microseconds_;
      }
      {
        std::lock_guard<std::mutex> other_backward_lock(other.backward_times_mutex_);
        std::lock_guard<std::mutex> this_backward_lock(backward_times_mutex_);
        backward_times_microseconds_ = other.backward_times_microseconds_;
      }
    }
    return *this;
  }

  Sequential &operator=(Sequential &&other) noexcept {
    if (this != &other) {
      layers_ = std::move(other.layers_);
      optimizer_ = std::move(other.optimizer_);
      loss_ = std::move(other.loss_);
      name_ = std::move(other.name_);
      is_training_ = other.is_training_;

      enable_profiling_ = other.enable_profiling_;
      forward_times_microseconds_ = std::move(other.forward_times_microseconds_);
      backward_times_microseconds_ = std::move(other.backward_times_microseconds_);
    }
    return *this;
  }

  void set_seed(unsigned long long seed) {
    for (auto &layer : layers_) {
      layer->set_seed(seed);
    }
  }

  void initialize() {
    for (auto &layer : layers_) {
      layer->initialize();
    }
  }

  void add(std::unique_ptr<Layer<T>> layer) {
    if (!layer) {
      throw std::invalid_argument("Cannot add null layer");
    }
    layer->set_training(is_training_);
    layers_.push_back(std::move(layer));
  }

  void insert(size_t index, std::unique_ptr<Layer<T>> layer) {
    if (!layer) {
      throw std::invalid_argument("Cannot insert null layer");
    }
    if (index > layers_.size()) {
      throw std::out_of_range("Insert index out of range");
    }
    layer->set_training(is_training_);
    layers_.insert(layers_.begin() + index, std::move(layer));
  }

  void remove(size_t index) {
    if (index >= layers_.size()) {
      throw std::out_of_range("Remove index out of range");
    }
    layers_.erase(layers_.begin() + index);
  }

  void clear() {
    layers_.clear();
    {
      std::lock_guard<std::mutex> forward_lock(forward_times_mutex_);
      forward_times_microseconds_.clear();
    }
    {
      std::lock_guard<std::mutex> backward_lock(backward_times_mutex_);
      backward_times_microseconds_.clear();
    }
  }

  size_t layer_size() const { return layers_.size(); }

  /**
   * @brief Sets training mode for the model and all its layers.
   * @param training Set to true for training mode, false for evaluation mode.
   */
  void set_training(bool training) {
    is_training_ = training;
    for (auto &layer : layers_) {
      layer->set_training(training);
    }
  }

  /**
   * @brief Returns true if the model is in training mode, false if in evaluation mode.
   */
  bool is_training() const { return is_training_; }

  /**
   * @brief Enables or disables profiling of forward and backward passes.
   * @param enable Set to true to enable profiling, false to disable.
   */
  void enable_profiling(bool enable = true) {
    enable_profiling_ = enable;
    if (enable) {
      {
        std::lock_guard<std::mutex> forward_lock(forward_times_mutex_);
        forward_times_microseconds_.clear();
      }
      {
        std::lock_guard<std::mutex> backward_lock(backward_times_mutex_);
        backward_times_microseconds_.clear();
      }
    }
    for (auto &layer : layers_) {
      layer->enable_profiling(enable);
    }
  }

  /**
   * @brief Returns true if profiling is enabled, false otherwise.
   */
  bool is_profiling_enabled() const { return enable_profiling_; }

  /**
   * @brief Clears all recorded profiling data.
   */
  void clear_profiling_data() {
    {
      std::lock_guard<std::mutex> forward_lock(forward_times_mutex_);
      forward_times_microseconds_.clear();
    }
    {
      std::lock_guard<std::mutex> backward_lock(backward_times_mutex_);
      backward_times_microseconds_.clear();
    }
    for (auto &layer : layers_) {
      layer->reset_profiling_info();
    }
  }

  /**
   * @brief Clears only the recorded forward times.
   */
  void clear_forward_times() {
    std::lock_guard<std::mutex> lock(forward_times_mutex_);
    forward_times_microseconds_.clear();
  }

  /**
   * @brief Clears only the recorded backward times.
   */
  void clear_backward_times() {
    std::lock_guard<std::mutex> lock(backward_times_mutex_);
    backward_times_microseconds_.clear();
  }

  /**
   * @brief Returns the recorded forward times for each layer in milliseconds.
   */
  std::map<std::string, uint64_t> get_forward_times() const {
    std::lock_guard<std::mutex> lock(forward_times_mutex_);
    return forward_times_microseconds_;
  }

  /**
   * @brief Returns the recorded backward times for each layer in milliseconds.
   */
  std::map<std::string, uint64_t> get_backward_times() const {
    std::lock_guard<std::mutex> lock(backward_times_mutex_);
    return backward_times_microseconds_;
  }

  void print_cache_memory_summary() const {
    size_t total_cache_bytes = 0;
    std::cout << "Cache Memory Usage Summary for Sequential Model: " << name_ << "\n";
    std::cout << std::string(50, '=') << "\n";
    std::cout << std::left << std::setw(30) << "Layer" << std::setw(20) << "Cache Memory (bytes)"
              << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (const auto &layer : layers_) {
      size_t layer_cache_bytes = layer->cached_memory_bytes();
      total_cache_bytes += layer_cache_bytes;

      std::string layer_name = layer->type();
      auto config = layer->get_config();
      if (!config.name.empty()) {
        layer_name = config.name;
      }

      std::cout << std::left << std::setw(30) << layer_name << std::setw(20) << layer_cache_bytes
                << "\n";
    }

    std::cout << std::string(50, '-') << "\n";
    std::cout << std::left << std::setw(30) << "TOTAL" << std::setw(20) << total_cache_bytes
              << "\n";
    std::cout << std::string(50, '=') << "\n\n";
  }

  /**
   * @brief Prints a summary of the profiling data to the console if profiling is enabled, otherwise
   * prints a warning.
   */
  void print_profiling_summary() const {
    if (!enable_profiling_) {
      std::cout << "Profiling disabled. Enable profiling with "
                   "enable_profiling(true)\n";
      return;
    }

    // Create thread-safe copies of the timing maps
    std::map<std::string, uint64_t> forward_times_copy;
    std::map<std::string, uint64_t> backward_times_copy;

    {
      std::lock_guard<std::mutex> forward_lock(forward_times_mutex_);
      forward_times_copy = forward_times_microseconds_;
    }
    {
      std::lock_guard<std::mutex> backward_lock(backward_times_mutex_);
      backward_times_copy = backward_times_microseconds_;
    }

    if (forward_times_copy.empty()) {
      std::cout << "Profiling disabled. Enable profiling with "
                   "enable_profiling(true)\n";
      return;
    }

    std::cout << std::string(70, '=') << "\n";
    std::cout << "Performance Profile: " << name_ << "\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << std::left << std::setw(20) << "Layer" << std::setw(15) << "Forward (ms)"
              << std::setw(15) << "Backward (ms)" << std::setw(15) << "Total (ms)" << "\n";
    std::cout << std::string(70, '-') << "\n";

    uint64_t total_forward = 0, total_backward = 0;
    for (size_t i = 0; i < layers_.size(); ++i) {

      std::string layer_name = layers_[i]->type();
      auto config = layers_[i]->get_config();
      if (!config.name.empty()) {
        layer_name = config.name;
      }

      uint64_t forward_time = 0;
      auto forward_it = forward_times_copy.find(layer_name);
      if (forward_it != forward_times_copy.end()) {
        forward_time = forward_it->second;
      }

      uint64_t backward_time = 0;
      auto backward_it = backward_times_copy.find(layer_name);
      if (backward_it != backward_times_copy.end()) {
        backward_time = backward_it->second;
      }

      uint64_t total_time = forward_time + backward_time;

      total_forward += forward_time;
      total_backward += backward_time;

      std::cout << std::left << std::setw(20) << layer_name << std::setw(15) << std::fixed
                << std::setprecision(3) << static_cast<double>(forward_time) / 1000.0
                << std::setw(15) << std::fixed << std::setprecision(3)
                << static_cast<double>(backward_time) / 1000.0 << std::setw(15) << std::fixed
                << std::setprecision(3) << static_cast<double>(total_time) / 1000.0 << "\n";
    }

    std::string sync_layer_name = "synchronization";
    uint64_t forward_sync_time = 0;
    auto forward_sync_it = forward_times_copy.find(sync_layer_name);
    if (forward_sync_it != forward_times_copy.end()) {
      forward_sync_time = forward_sync_it->second;
    }
    uint64_t backward_sync_time = 0;
    auto backward_sync_it = backward_times_copy.find(sync_layer_name);
    if (backward_sync_it != backward_times_copy.end()) {
      backward_sync_time = backward_sync_it->second;
    }
    uint64_t total_sync_time = forward_sync_time + backward_sync_time;
    total_forward += forward_sync_time;
    total_backward += backward_sync_time;

    std::cout << std::left << std::setw(20) << sync_layer_name << std::setw(15) << std::fixed
              << std::setprecision(3) << static_cast<double>(forward_sync_time) / 1000.0
              << std::setw(15) << std::fixed << std::setprecision(3)
              << static_cast<double>(backward_sync_time) / 1000.0 << std::setw(15) << std::fixed
              << std::setprecision(3) << static_cast<double>(total_sync_time) / 1000.0 << "\n";

    std::cout << std::string(70, '-') << "\n";
    std::cout << std::left << std::setw(20) << "TOTAL" << std::setw(15) << std::fixed
              << std::setprecision(3) << static_cast<double>(total_forward / 1000.0)
              << std::setw(15) << std::fixed << std::setprecision(3)
              << static_cast<double>(total_backward / 1000.0) << std::setw(15) << std::fixed
              << std::setprecision(3)
              << static_cast<double>(total_forward + total_backward) / 1000.0 << "\n"
              << std::string(70, '=') << "\n\n";
  }

  const Device *get_device() const { return device_; }

  void set_device(const Device *device) {
    device_ = device;
    for (auto &layer : layers_) {
      layer->set_device(device);
    }
  }

  /**
   * @brief Set device for all layers in the model.
   * @param device_id The target device ID (string).
   */
  void set_device(const std::string &device_id) {
    const Device *target_device = &DeviceManager::getInstance().getDevice(device_id);
    set_device(target_device);
  }

  /**
   */
  void set_device(DeviceType device_type) {
    const Device *target_device = &DeviceManager::getInstance().getDevice(device_type);
    set_device(target_device);
  }

  /**
   * @brief Performs a forward pass through the model.
   * @param input The input tensor.
   * @param micro_batch_id The ID of the microbatch, defaulting to 0 for normal training.
   */
  Tensor<T> forward(const Tensor<T> &input, size_t micro_batch_id = 0) {
    if (layers_.empty()) {
      throw std::runtime_error("Cannot forward through empty sequential model");
    }
    const Device *input_device = input.device();
    const Tensor<T> *current = &input;
    for (size_t i = 0; i < layers_.size(); ++i) {
      try {
        // just profile since it's not expensive
        auto start_time = std::chrono::high_resolution_clock::now();
        current = &layers_[i]->forward(*current, micro_batch_id);
        // cudaDeviceSynchronize(); // DEBUG
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        std::string layer_name = layers_[i]->type();
        auto config = layers_[i]->get_config();
        if (!config.name.empty()) {
          layer_name = config.name;
        }

        {
          std::lock_guard<std::mutex> lock(forward_times_mutex_);
          forward_times_microseconds_[layer_name] += duration.count();
        }
      } catch (const std::exception &e) {
        throw std::runtime_error("Error while forward in layer " + std::to_string(i) + " (" +
                                 layers_[i]->name() + "): " + e.what());
      }
    }

    auto sync_start = std::chrono::high_resolution_clock::now();
    Flow *def_flow = input_device->getFlow("default");
    def_flow->synchronize();
    auto sync_end = std::chrono::high_resolution_clock::now();
    auto sync_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(sync_end - sync_start);
    {
      std::lock_guard<std::mutex> lock(forward_times_mutex_);
      forward_times_microseconds_["synchronization"] += sync_duration.count();
    }

    return current->to_device(input_device);
  }

  /**
   * @brief Prints layers' profiling info for specific operations if profiling is enabled.
   */
  void print_layers_profiling_info() const {
    if (!enable_profiling_) {
      std::cout << "Profiling is not enabled. Enable it with enable_profiling(true)\n";
      return;
    }

    std::cout << "Layers' Profiling Info:\n";
    std::cout << std::string(40, '=') << "\n";
    for (const auto &layer : layers_) {
      layer->print_profiling_info();
    }
  }

  /**
   * @brief Performs a backward pass through the model.
   * @param gradient The gradient tensor from the subsequent layer or loss function.
   * @param micro_batch_id The ID of the microbatch, defaulting to 0 for normal training.
   */
  Tensor<T> backward(const Tensor<T> &gradient, size_t micro_batch_id = 0) {
    if (layers_.empty()) {
      throw std::runtime_error("Cannot backward through empty sequential model");
    }

    const Device *grad_device = gradient.device();
    const Tensor<T> *current_gradient = &gradient;
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
      try {
        auto start_time = std::chrono::high_resolution_clock::now();
        current_gradient = &layers_[i]->backward(*current_gradient, micro_batch_id);
        // cudaDeviceSynchronize(); // DEBUG
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        std::string layer_name = layers_[i]->type();
        auto config = layers_[i]->get_config();
        if (!config.name.empty()) {
          layer_name = config.name;
        }

        {
          std::lock_guard<std::mutex> lock(backward_times_mutex_);
          backward_times_microseconds_[layer_name] += duration.count();
        }

      } catch (const std::exception &e) {
        throw std::runtime_error("Error in backward pass of layer " + std::to_string(i) + " (" +
                                 layers_[i]->type() + "): " + e.what());
      }
    }

    auto sync_start = std::chrono::high_resolution_clock::now();
    Flow *def_flow = grad_device->getFlow("default");
    def_flow->synchronize();
    auto sync_end = std::chrono::high_resolution_clock::now();
    auto sync_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(sync_end - sync_start);
    {
      std::lock_guard<std::mutex> lock(backward_times_mutex_);
      backward_times_microseconds_["synchronization"] += sync_duration.count();
    }

    return current_gradient->to_device(gradient.device());
  }

  /**
   * @brief Returns a vector of pointers to all params in the model
   */
  std::vector<Tensor<T> *> parameters() const {
    std::vector<Tensor<T> *> all_params;
    for (auto &layer : layers_) {
      auto layer_params = layer->parameters();
      all_params.insert(all_params.end(), layer_params.begin(), layer_params.end());
    }
    return all_params;
  }

  /**
   * @brief Returns a vector of pointers to params in the specified partition
   * @param part The partition specifying the range of layers.
   */
  std::vector<Tensor<T> *> parameters(const Partition &part) const {
    if (part.start_layer >= layers_.size() || part.end_layer > layers_.size() ||
        part.start_layer >= part.end_layer) {
      throw std::out_of_range("Partition indices out of range");
    }

    std::vector<Tensor<T> *> part_params;
    for (size_t i = part.start_layer; i < part.end_layer; ++i) {
      auto layer_params = layers_[i]->parameters();
      part_params.insert(part_params.end(), layer_params.begin(), layer_params.end());
    }
    return part_params;
  }

  /**
   * @brief Returns a vector of pointers to all gradients in the model
   */
  std::vector<Tensor<T> *> gradients() const {
    std::vector<Tensor<T> *> all_grads;
    for (auto &layer : layers_) {
      auto layer_grads = layer->gradients();
      all_grads.insert(all_grads.end(), layer_grads.begin(), layer_grads.end());
    }
    return all_grads;
  }

  /**
   * @brief Returns a vector of pointers to gradients in the specified partition
   * @param part The partition specifying the range of layers.
   */
  std::vector<Tensor<T> *> gradients(const Partition &part) const {
    if (part.start_layer >= layers_.size() || part.end_layer > layers_.size() ||
        part.start_layer >= part.end_layer) {
      throw std::out_of_range("Partition indices out of range");
    }

    std::vector<Tensor<T> *> part_grads;
    for (size_t i = part.start_layer; i < part.end_layer; ++i) {
      auto layer_grads = layers_[i]->gradients();
      part_grads.insert(part_grads.end(), layer_grads.begin(), layer_grads.end());
    }
    return part_grads;
  }

  /**
   * @brief Returns the output shape for given input shape
   * @param input_shape The shape of the input tensor as a vector of sizes.
   */
  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const {
    if (layers_.empty()) {
      return input_shape;
    }

    std::vector<size_t> current_shape = input_shape;
    for (const auto &layer : layers_) {
      current_shape = layer->compute_output_shape(current_shape);
    }

    return current_shape;
  }

  /**
   * @brief Returns the output shape for given input shape and partition
   * @param input_shape The shape of the input tensor as a vector of sizes.
   * @param part The partition specifying the range of layers.
   */
  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape,
                                           const Partition &part) const {
    if (layers_.empty()) {
      return input_shape;
    }
    if (part.start_layer >= layers_.size() || part.end_layer > layers_.size() ||
        part.start_layer >= part.end_layer) {
      throw std::out_of_range("Partition indices out of range");
    }

    std::vector<size_t> current_shape = input_shape;
    for (size_t i = part.start_layer; i < part.end_layer; ++i) {
      current_shape = layers_[i]->compute_output_shape(current_shape);
    }

    return current_shape;
  }

  /**
   * @brief Returns the relative forward complexity (in FLOPs) for each layer given an input
   * shape.
   * @param input_shape The shape of the input tensor as a vector of sizes.
   */
  std::vector<uint64_t> forward_complexity(const std::vector<size_t> &input_shape) const {
    if (layers_.empty()) {
      return {};
    }

    std::vector<uint64_t> layer_complexities;
    std::vector<size_t> current_shape = input_shape;

    for (const auto &layer : layers_) {
      uint64_t layer_complexity = layer->forward_complexity(current_shape);
      layer_complexities.push_back(layer_complexity);
      current_shape = layer->compute_output_shape(current_shape);
    }

    return layer_complexities;
  }

  /**
   * @brief Returns the relative forward complexity (in FLOPs) for each layer in the specified
   * partition
   * @param input_shape The shape of the input tensor as a vector of sizes.
   */
  std::vector<uint64_t> forward_complexity(const std::vector<size_t> &input_shape,
                                           const Partition &part) const {
    if (layers_.empty()) {
      return {};
    }
    if (part.start_layer >= layers_.size() || part.end_layer > layers_.size() ||
        part.start_layer >= part.end_layer) {
      throw std::out_of_range("Partition indices out of range");
    }

    std::vector<uint64_t> layer_complexities;
    std::vector<size_t> current_shape = input_shape;

    for (size_t i = part.start_layer; i < part.end_layer; ++i) {
      uint64_t layer_complexity = layers_[i]->forward_complexity(current_shape);
      layer_complexities.push_back(layer_complexity);
      current_shape = layers_[i]->compute_output_shape(current_shape);
    }

    return layer_complexities;
  }

  /**
   * @brief Returns the relative backward complexity (in FLOPs) for each layer given a gradient
   * shape.
   * @param input_shape The shape of the gradient tensor as a vector of sizes.
   */
  std::vector<uint64_t> backward_complexity(const std::vector<size_t> &input_shape) const {
    if (layers_.empty()) {
      return {};
    }

    std::vector<uint64_t> layer_complexities;
    std::vector<size_t> current_shape = input_shape;

    for (auto &layer : layers_) {
      uint64_t layer_complexity = layer->backward_complexity(current_shape);
      layer_complexities.push_back(layer_complexity);
      current_shape = layer->compute_output_shape(current_shape);
    }

    return layer_complexities;
  }

  /**
   * @brief Returns the relative backward complexity (in FLOPs) for each layer in the specified
   * partition
   * @param input_shape The shape of the gradient tensor as a vector of sizes.
   * @param part The partition specifying the range of layers.
   */
  std::vector<uint64_t> backward_complexity(const std::vector<size_t> &input_shape,
                                            const Partition &part) const {
    if (layers_.empty()) {
      return {};
    }
    if (part.start_layer >= layers_.size() || part.end_layer > layers_.size() ||
        part.start_layer >= part.end_layer) {
      throw std::out_of_range("Partition indices out of range");
    }

    std::vector<uint64_t> layer_complexities;
    std::vector<size_t> current_shape = input_shape;

    for (size_t i = part.start_layer; i < part.end_layer; ++i) {
      uint64_t layer_complexity = layers_[i]->backward_complexity(current_shape);
      layer_complexities.push_back(layer_complexity);
      current_shape = layers_[i]->compute_output_shape(current_shape);
    }

    return layer_complexities;
  }

  /**
   * @brief Prints the model's configuration in JSON format to the console.
   */
  void print_config() const { std::cout << get_config().dump(2) << std::endl; }

  void print_summary(const std::vector<size_t> &input_shape) const {
    if (layers_.empty()) {
      std::cout << "Empty model.\n";
      return;
    }

    std::cout << std::string(75, '=') << "\n";
    std::cout << "Model Summary: " << name_ << "\n";
    std::cout << std::string(75, '=') << "\n";
    std::cout << std::left << std::setw(15) << "Layer (Type)" << std::setw(15) << "Input Shape"
              << std::setw(15) << "Output Shape" << std::setw(20) << "Forward Complexity"
              << std::setw(20) << "Backward Complexity" << "\n";

    std::vector<size_t> current_shape = input_shape;
    for (size_t i = 0; i < layers_.size(); ++i) {
      const auto &layer = layers_[i];
      std::cout << std::left << std::setw(15)
                << (layer->get_config().name.empty() ? layer->type() : layer->get_config().name);

      std::string input_shape_str = "(";
      for (size_t j = 0; j < current_shape.size(); ++j) {
        if (j > 0)
          input_shape_str += ",";
        input_shape_str += std::to_string(current_shape[j]);
      }
      input_shape_str += ")";
      std::cout << std::setw(15) << input_shape_str;

      auto output_shape = layer->compute_output_shape(current_shape);
      std::string output_shape_str = "(";
      for (size_t j = 0; j < output_shape.size(); ++j) {
        if (j > 0)
          output_shape_str += ",";
        output_shape_str += std::to_string(output_shape[j]);
      }
      output_shape_str += ")";
      std::cout << std::setw(15) << output_shape_str;

      std::cout << std::setw(20) << layer->forward_complexity(current_shape) << std::setw(20)
                << layer->backward_complexity(current_shape) << "\n";
      current_shape = layer->compute_output_shape(current_shape);
    }
    std::cout << std::string(75, '-') << "\n";
  }

  /**
   * @brief Save the model to specified path.
   * The model's config will be save to json for readability, and the weights will be saved in a
   * binary format.
   * @param path The base path to save the model (without file extension).
   */
  void save_to_file(const std::string &path) const {
    // Create directory if it doesn't exist
    auto dir_path = std::filesystem::path(path).parent_path();
    if (!dir_path.empty() && !std::filesystem::exists(dir_path)) {
      std::filesystem::create_directories(dir_path);
    }

    nlohmann::json config_json = get_config();

    std::ofstream config_file(path + ".json");
    if (!config_file.is_open()) {
      throw std::runtime_error("Could not create config file: " + path + ".json");
    }
    config_file << config_json.dump(4);
    config_file.close();

    std::ofstream weights_file(path + ".bin", std::ios::binary);
    if (!weights_file.is_open()) {
      throw std::runtime_error("Could not create weights file: " + path + ".bin");
    }
    for (const auto &layer : layers_) {
      if (layer->has_parameters()) {
        auto params = const_cast<Layer<T> *>(layer.get())->parameters();
        for (const auto &param : params) {
          param->save(weights_file);
        }
      }
    }
    weights_file.close();
  }

  void load_weights_file(const std::string &path) {
    std::ifstream weights_file(path, std::ios::binary);
    if (!weights_file.is_open()) {
      throw std::runtime_error("Could not open weights file: " + path);
    }
    for (auto &layer : layers_) {
      layer->initialize();
      if (layer->has_parameters()) {
        auto params = layer->parameters();
        for (auto &param : params) {
          *param = Tensor<T>::load(weights_file);
        }
      }
    }
    weights_file.close();
  }

  /**
   * @brief Load a model from specified path.
   * The model's config will be loaded from a json file, and the weights will be loaded from a
   * binary file.
   * @param path The base path to load the model (without file extension).
   * @return The loaded Sequential model.
   */
  static Sequential<T> from_file(const std::string &path) {

    std::ifstream config_file(path + ".json");
    if (!config_file.is_open()) {
      throw std::runtime_error("Could not open config file: " + path + ".json");
    }
    nlohmann::json config_json;
    config_file >> config_json;
    config_file.close();

    Sequential<T> model = load_from_config(config_json);

    std::ifstream weights_file(path + ".bin", std::ios::binary);
    if (!weights_file.is_open()) {
      throw std::runtime_error("Could not open weights file: " + path + ".bin");
    }
    for (auto &layer : model.layers_) {
      if (layer->has_parameters()) {
        layer->initialize();
        auto params = layer->parameters();
        for (auto &param : params) {
          *param = Tensor<T>::load(weights_file);
        }
      }
    }
    weights_file.close();

    return model;
  }

  std::unique_ptr<Sequential> clone() const {
    auto cloned = std::make_unique<Sequential>(name_);
    cloned->set_training(is_training_);

    for (const auto &layer : layers_) {
      cloned->add(layer->clone());
    }

    return cloned;
  }

  const std::string &name() const { return name_; }

  void set_name(const std::string &name) { name_ = name; }

  void update_parameters() const {
    auto params = parameters();
    auto grads = gradients();
    if (params.size() != grads.size()) {
      std::cout << "Params size: " << params.size() << ", Grads size: " << grads.size() << "\n";
      throw std::runtime_error("Parameter and gradient count mismatch during update");
    }
    if (!optimizer_) {
      throw std::runtime_error("No optimizer set for model");
    }
    optimizer_->update(params, grads);
    clear_gradients();
  }

  void clear_gradients() const {
    std::vector<Tensor<T> *> grads = gradients();
    std::vector<std::unique_ptr<Task>> tasks;
    for (auto &grad : grads) {
      tasks.emplace_back(grad->fill(T(0)));
    }
    for (auto &task : tasks) {
      auto err = task->sync();
      if (err != ErrorStatus{}) {
        throw std::runtime_error("Error while clearing gradients: " + err.message());
      }
    }
  }

  void load_parameters(std::vector<Tensor<T>> &&parameters) {
    size_t param_index = 0;
    for (auto &layer : layers_) {
      if (layer->has_parameters()) {
        auto params = layer->parameters();
        for (auto &param : params) {
          if (param_index >= parameters.size()) {
            throw std::runtime_error("Not enough parameters provided to load into model");
          }

          if (param->shape() != parameters[param_index].shape()) {
            throw std::runtime_error("Parameter shape mismatch at index " +
                                     std::to_string(param_index) + ": expected " +
                                     std::to_string(param->shape().size()) + " dimensions");
          }

          *param = std::move(parameters[param_index]);
          param_index++;
        }
      }
    }

    if (param_index != parameters.size()) {
      throw std::runtime_error("Parameter count mismatch: expected " + std::to_string(param_index) +
                               " but got " + std::to_string(parameters.size()));
    }
  }

  void set_optimizer(std::unique_ptr<Optimizer<T>> optimizer) {
    this->optimizer_ = std::move(optimizer);
  }

  void set_loss_function(std::unique_ptr<Loss<T>> loss) { this->loss_ = std::move(loss); }

  Optimizer<T> *optimizer() const { return optimizer_.get(); }

  Loss<T> *loss_function() const { return loss_.get(); }

  std::vector<Sequential<T>> split(std::vector<Partition> &partitions) const {
    if (partitions.empty()) {
      throw std::invalid_argument("Partitions vector is empty");
    }
    std::vector<Sequential<T>> stages;
    stages.reserve(partitions.size());
    for (const auto &part : partitions) {
      if (part.start_layer >= layers_.size() || part.end_layer > layers_.size() ||
          part.start_layer >= part.end_layer) {
        throw std::out_of_range("Invalid partition range");
      }

      Sequential<T> stage(name_ + "_part_" + std::to_string(stages.size()));
      for (size_t i = part.start_layer; i < part.end_layer; ++i) {
        stage.add(layers_[i]->clone());
      }
      if (this->optimizer_) {
        stage.set_optimizer(this->optimizer_->clone());
      }
      if (this->loss_) {
        stage.set_loss_function(this->loss_->clone());
      }
      stages.push_back(std::move(stage));
    }
    return stages;
  }

  const std::vector<Layer<T> *> &get_layers() const {
    static std::vector<Layer<T> *> layer_ptrs;
    layer_ptrs.clear();
    for (const auto &layer : layers_) {
      layer_ptrs.push_back(layer.get());
    }
    return layer_ptrs;
  }

  /**
   * @brief Returns the model configuration as a JSON object.
   * This includes the model name, training mode, layers, optimizer, and loss function.
   */
  nlohmann::json get_config() const {
    nlohmann::json config;
    config["name"] = name_;
    config["is_training"] = is_training_;

    nlohmann::json layers_config = nlohmann::json::array();
    for (const auto &layer : layers_) {
      LayerConfig layer_config = layer->get_config();
      nlohmann::json layer_json;
      layer_json["type"] = layer->type();
      layer_json["name"] = layer_config.name;
      layer_json["parameters"] = nlohmann::json::object();

      for (const auto &[key, value] : layer_config.parameters) {
        try {
          if (auto *int_ptr = std::any_cast<int>(&value)) {
            layer_json["parameters"][key] = *int_ptr;
          } else if (auto *size_ptr = std::any_cast<size_t>(&value)) {
            layer_json["parameters"][key] = *size_ptr;
          } else if (auto *float_ptr = std::any_cast<float>(&value)) {
            layer_json["parameters"][key] = *float_ptr;
          } else if (auto *double_ptr = std::any_cast<double>(&value)) {
            layer_json["parameters"][key] = *double_ptr;
          } else if (auto *bool_ptr = std::any_cast<bool>(&value)) {
            layer_json["parameters"][key] = *bool_ptr;
          } else if (auto *string_ptr = std::any_cast<std::string>(&value)) {
            layer_json["parameters"][key] = *string_ptr;
          }
        } catch (const std::bad_any_cast &) {
        }
      }
      layers_config.push_back(layer_json);
    }
    config["layers"] = layers_config;

    if (optimizer_) {
      OptimizerConfig opt_config = optimizer_->get_config();
      nlohmann::json opt_json;
      opt_json["type"] = opt_config.type;
      opt_json["name"] = opt_config.name;
      opt_json["parameters"] = nlohmann::json::object();

      for (const auto &[key, value] : opt_config.parameters) {
        try {
          if (auto *float_ptr = std::any_cast<float>(&value)) {
            opt_json["parameters"][key] = *float_ptr;
          } else if (auto *double_ptr = std::any_cast<double>(&value)) {
            opt_json["parameters"][key] = *double_ptr;
          }
        } catch (const std::bad_any_cast &) {
        }
      }
      config["optimizer"] = opt_json;
    }

    if (loss_) {
      LossConfig loss_config = loss_->get_config();
      nlohmann::json loss_json;
      loss_json["type"] = loss_config.type;
      loss_json["name"] = loss_config.name;
      loss_json["parameters"] = nlohmann::json::object();

      for (const auto &[key, value] : loss_config.parameters) {
        try {
          if (auto *float_ptr = std::any_cast<float>(&value)) {
            loss_json["parameters"][key] = *float_ptr;
          } else if (auto *double_ptr = std::any_cast<double>(&value)) {
            loss_json["parameters"][key] = *double_ptr;
          }
        } catch (const std::bad_any_cast &) {
        }
      }
      config["loss"] = loss_json;
    }

    return config;
  }

  /**
   * @brief Gets the config json from partition params
   */
  nlohmann::json get_config(const Partition &partition) const {
    if (partition.start_layer >= layers_.size() || partition.end_layer > layers_.size() ||
        partition.start_layer >= partition.end_layer) {
      throw std::out_of_range("Partition indices out of range");
    }

    nlohmann::json config;
    config["name"] = name_ + "_part_" + std::to_string(partition.start_layer) + "_" +
                     std::to_string(partition.end_layer);
    config["is_training"] = is_training_;

    nlohmann::json layers_config = nlohmann::json::array();
    for (size_t i = partition.start_layer; i < partition.end_layer; ++i) {
      const auto &layer = layers_[i];
      LayerConfig layer_config = layer->get_config();
      nlohmann::json layer_json;
      layer_json["type"] = layer->type();
      layer_json["name"] = layer_config.name;
      layer_json["parameters"] = nlohmann::json::object();

      for (const auto &[key, value] : layer_config.parameters) {
        try {
          if (auto *int_ptr = std::any_cast<int>(&value)) {
            layer_json["parameters"][key] = *int_ptr;
          } else if (auto *size_ptr = std::any_cast<size_t>(&value)) {
            layer_json["parameters"][key] = *size_ptr;
          } else if (auto *float_ptr = std::any_cast<float>(&value)) {
            layer_json["parameters"][key] = *float_ptr;
          } else if (auto *double_ptr = std::any_cast<double>(&value)) {
            layer_json["parameters"][key] = *double_ptr;
          } else if (auto *bool_ptr = std::any_cast<bool>(&value)) {
            layer_json["parameters"][key] = *bool_ptr;
          } else if (auto *string_ptr = std::any_cast<std::string>(&value)) {
            layer_json["parameters"][key] = *string_ptr;
          }
        } catch (const std::bad_any_cast &) {
        }
      }
      layers_config.push_back(layer_json);
    }
    config["layers"] = layers_config;

    if (optimizer_) {
      OptimizerConfig opt_config = optimizer_->get_config();
      nlohmann::json opt_json;
      opt_json["type"] = opt_config.type;
      opt_json["name"] = opt_config.name;
      opt_json["parameters"] = nlohmann::json::object();

      for (const auto &[key, value] : opt_config.parameters) {
        try {
          if (auto *float_ptr = std::any_cast<float>(&value)) {
            opt_json["parameters"][key] = *float_ptr;
          } else if (auto *double_ptr = std::any_cast<double>(&value)) {
            opt_json["parameters"][key] = *double_ptr;
          }
        } catch (const std::bad_any_cast &) {
        }
      }
      config["optimizer"] = opt_json;
    }

    if (loss_) {
      LossConfig loss_config = loss_->get_config();
      nlohmann::json loss_json;
      loss_json["type"] = loss_config.type;
      loss_json["name"] = loss_config.name;
      loss_json["parameters"] = nlohmann::json::object();

      for (const auto &[key, value] : loss_config.parameters) {
        try {
          if (auto *float_ptr = std::any_cast<float>(&value)) {
            loss_json["parameters"][key] = *float_ptr;
          } else if (auto *double_ptr = std::any_cast<double>(&value)) {
            loss_json["parameters"][key] = *double_ptr;
          }
        } catch (const std::bad_any_cast &) {
        }
      }
      config["loss"] = loss_json;
    }

    return config;
  }

  /**
   * @brief Loads a Sequential model from a JSON configuration object.
   * @param config The JSON object containing the model configuration.
   * @return The constructed Sequential model.
   */
  static Sequential<T> load_from_config(const nlohmann::json &config) {
    Sequential<T> model(config.value("name", "sequential"));
    model.is_training_ = config.value("is_training", true);

    if (config.contains("optimizer")) {
      OptimizerConfig opt_config;
      opt_config.type = config["optimizer"]["type"];
      opt_config.name = config["optimizer"]["name"];

      if (config["optimizer"].contains("parameters")) {
        for (const auto &[key, value] : config["optimizer"]["parameters"].items()) {
          if (value.is_number_float()) {
            opt_config.parameters[key] = value.template get<float>();
          } else if (value.is_number_integer()) {
            opt_config.parameters[key] = value.template get<int>();
          }
        }
      }
      std::unique_ptr<Optimizer<T>> optimizer = OptimizerFactory<T>::create_from_config(opt_config);
      model.set_optimizer(std::move(optimizer));
    }

    if (config.contains("loss")) {
      LossConfig loss_config;
      loss_config.type = config["loss"]["type"];
      loss_config.name = config["loss"]["name"];

      if (config["loss"].contains("parameters")) {
        for (const auto &[key, value] : config["loss"]["parameters"].items()) {
          if (value.is_number_float()) {
            loss_config.parameters[key] = value.template get<float>();
          } else if (value.is_number_integer()) {
            loss_config.parameters[key] = value.template get<int>();
          }
        }
      }

      model.set_loss_function(LossFactory<T>::create_from_config(loss_config));
    }

    if (config.contains("layers")) {
      auto factory = LayerFactory<T>();
      factory.register_defaults();

      for (const auto &layer_json : config["layers"]) {
        LayerConfig layer_config;
        layer_config.name = layer_json.value("name", "");

        if (layer_json.contains("parameters")) {
          for (const auto &[key, value] : layer_json["parameters"].items()) {
            if (value.is_number_integer()) {
              layer_config.parameters[key] = value.template get<size_t>();
            } else if (value.is_number_float()) {
              layer_config.parameters[key] = value.template get<float>();
            } else if (value.is_boolean()) {
              layer_config.parameters[key] = value.template get<bool>();
            } else if (value.is_string()) {
              layer_config.parameters[key] = value.template get<std::string>();
            }
          }
        }

        std::string layer_type = layer_json.value("type", "");
        auto layer = factory.create(layer_type, layer_config);
        model.add(std::move(layer));
      }
    }

    return model;
  }

  /**
   * @brief Saves the model configuration to a JSON file.
   * @param filepath The path to the file where the configuration will be saved.
   */
  void save_config(const std::string &filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file for writing: " + filepath);
    }
    file << get_config().dump(2);
    file.close();
  }

  static Sequential<T> load_from_config_file(const std::string &filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file for reading: " + filepath);
    }

    nlohmann::json config;
    file >> config;
    file.close();

    return load_from_config(config);
  }
};

template <typename T = float> class SequentialBuilder {
private:
  Sequential<T> model_;
  LayerBuilder<T> layer_builder_;
  std::string model_name_;

public:
  explicit SequentialBuilder(const std::string &name = "sequential")
      : model_(name), model_name_(name) {}

  std::vector<size_t> get_current_shape() const { return layer_builder_.get_current_shape(); }

  SequentialBuilder &input(const std::vector<size_t> &shape) {
    layer_builder_.input(shape);
    return *this;
  }

  SequentialBuilder &dense(size_t output_features, bool use_bias = true,
                           const std::string &name = "") {
    layer_builder_.dense(output_features, use_bias,
                         name.empty() ? "dense_" + std::to_string(model_.layer_size()) : name);
    return *this;
  }

  SequentialBuilder &conv2d(size_t out_channels, size_t kernel_h, size_t kernel_w,
                            size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0,
                            size_t pad_w = 0, bool use_bias = true, const std::string &name = "") {
    layer_builder_.conv2d(out_channels, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                          use_bias,
                          name.empty() ? "conv2d_" + std::to_string(model_.layer_size()) : name);
    return *this;
  }

  SequentialBuilder &batchnorm(T epsilon = T(1e-5), T momentum = T(0.1), bool affine = true,
                               const std::string &name = "") {
    layer_builder_.batchnorm(epsilon, momentum, affine,
                             name.empty() ? "batchnorm_" + std::to_string(model_.layer_size())
                                          : name);
    return *this;
  }

  SequentialBuilder &groupnorm(size_t num_groups, T epsilon = T(1e-5), bool affine = true,
                               const std::string &name = "") {
    layer_builder_.groupnorm(num_groups, epsilon, affine,
                             name.empty() ? "groupnorm_" + std::to_string(model_.layer_size())
                                          : name);
    return *this;
  }

  SequentialBuilder &activation(const std::string &activation_name, const std::string &name = "") {
    layer_builder_.activation(
        activation_name, name.empty() ? "activation_" + std::to_string(model_.layer_size()) : name);
    return *this;
  }

  SequentialBuilder &maxpool2d(size_t pool_h, size_t pool_w, size_t stride_h = 1,
                               size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
                               const std::string &name = "") {
    layer_builder_.maxpool2d(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                             name.empty() ? "maxpool2d_" + std::to_string(model_.layer_size())
                                          : name);
    return *this;
  }

  SequentialBuilder &avgpool2d(size_t pool_h, size_t pool_w, size_t stride_h = 1,
                               size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
                               const std::string &name = "") {
    layer_builder_.avgpool2d(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                             name.empty() ? "avgpool2d_" + std::to_string(model_.layer_size())
                                          : name);
    return *this;
  }

  SequentialBuilder &dropout(T dropout_rate, const std::string &name = "") {
    layer_builder_.dropout(dropout_rate,
                           name.empty() ? "dropout_" + std::to_string(model_.layer_size()) : name);
    return *this;
  }

  SequentialBuilder &flatten(const std::string &name = "") {
    layer_builder_.flatten(name.empty() ? "flatten_" + std::to_string(model_.layer_size()) : name);
    return *this;
  }

  SequentialBuilder &residual(std::vector<std::unique_ptr<Layer<T>>> main_path,
                              std::vector<std::unique_ptr<Layer<T>>> shortcut,
                              const std::string &activation_name = "relu",
                              const std::string &name = "") {
    std::unique_ptr<ResidualBlock<T>> res_block = residual_block<T>(
        std::move(main_path), std::move(shortcut), activation_name,
        name.empty() ? "residual_block_" + std::to_string(model_.layer_size()) : name);
    layer_builder_.add_layer(std::move(res_block));
    return *this;
  }

  /**
   * @brief Helper function to create a basic residual block (e.g., for ResNet-18/34)
   * Two 3x3 convolutions with batch normalization
   */
  SequentialBuilder &basic_residual_block(size_t in_channels, size_t out_channels,
                                          size_t stride = 1,
                                          const std::string &name = "basic_residual_block") {
    auto current_shape = layer_builder_.get_current_shape();
    auto input_shape = std::vector<size_t>{in_channels, current_shape[2], current_shape[3]};
    auto main_path = LayerBuilder<T>()
                         .input(input_shape)
                         .conv2d(out_channels, 3, 3, stride, stride, 1, 1, false)
                         .batchnorm(1e-5f, 0.1f, true, "bn0")
                         //  .groupnorm(32, 1e-5f, true, "gn0")
                         .activation("relu")
                         .conv2d(out_channels, 3, 3, 1, 1, 1, 1, false)
                         .batchnorm(1e-5f, 0.1f, true, "bn0")
                         //  .groupnorm(32, 1e-5f, true, "gn1")
                         .build();

    std::vector<std::unique_ptr<Layer<T>>> shortcut;
    if (stride != 1 || in_channels != out_channels) {
      shortcut = LayerBuilder<T>()
                     .input(input_shape)
                     .conv2d(out_channels, 1, 1, stride, stride, 0, 0, false)
                     .batchnorm(1e-5f, 0.1f, true, "bn0")
                     //  .groupnorm(32, 1e-5f, true, "gn0")
                     .build();
    }

    auto res_block = residual_block<T>(
        std::move(main_path), std::move(shortcut), "relu",
        name.empty() ? "basic_residual_block_" + std::to_string(model_.layer_size()) : name);
    layer_builder_.add_layer(std::move(res_block));
    return *this;
  }

  /**
   * @brief Helper function to create a bottleneck residual block (for ResNet-50/101/152)
   * 1x1 conv -> 3x3 conv -> 1x1 conv with batch normalization
   */
  SequentialBuilder &
  bottleneck_residual_block(size_t in_channels, size_t mid_channels, size_t out_channels,
                            size_t stride = 1,
                            const std::string &name = "bottleneck_residual_block") {
    auto current_shape = layer_builder_.get_current_shape();
    auto input_shape = std::vector<size_t>{in_channels, current_shape[2], current_shape[3]};
    // Build main path using LayerBuilder
    auto main_path = LayerBuilder<T>()
                         .input(input_shape)
                         .conv2d(mid_channels, 1, 1, 1, 1, 0, 0, false)
                         .batchnorm(1e-3f, 0.1f, true, "bn0")
                         .activation("relu")
                         .conv2d(mid_channels, 3, 3, stride, stride, 1, 1, false)
                         .batchnorm(1e-3f, 0.1f, true, "bn0")
                         .activation("relu")
                         // 1x1 conv to expand dimensions (no activation, added after residual)
                         .conv2d(out_channels, 1, 1, 1, 1, 0, 0, false)
                         .batchnorm(1e-3f, 0.1f, true, "bn0")
                         .build();

    // Build projection shortcut if dimensions change
    std::vector<std::unique_ptr<Layer<T>>> shortcut;
    if (stride != 1 || in_channels != out_channels) {
      shortcut = LayerBuilder<T>()
                     .input(input_shape)
                     .conv2d(out_channels, 1, 1, stride, stride, 0, 0, false)
                     .batchnorm(1e-3f, 0.1f, true, "bn0")
                     .build();
    }

    auto res_block = residual_block<T>(std::move(main_path), std::move(shortcut), "relu", name);
    layer_builder_.add_layer(std::move(res_block));
    return *this;
  }

  SequentialBuilder &add_layer(std::unique_ptr<Layer<T>> layer) {
    layer_builder_.add_layer(std::move(layer));
    return *this;
  }

  Sequential<T> build() {
    if (!layer_builder_.is_input_shape_set()) {
      throw std::runtime_error("Input shape must be set before building model. "
                               "Use .input() method.");
    }

    auto layers = layer_builder_.build();
    for (auto &layer : layers) {
      model_.add(std::move(layer));
    }

    return std::move(model_);
  }
};

} // namespace tnn