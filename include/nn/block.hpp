#pragma once

#include "nn/layer.hpp"

namespace tnn {
class Block : virtual public Layer {
public:
  Block(const std::string &name = "block") { this->name_ = name; }

  Vec<ParamDescriptor> param_descriptors() override {
    Vec<ParamDescriptor> descriptors;
    for (const auto &layer : layers()) {
      auto layer_descriptors = layer->param_descriptors();
      descriptors.insert(descriptors.end(), layer_descriptors.begin(), layer_descriptors.end());
    }
    return descriptors;
  }

protected:
  void on_set_engine_type(EngineType engine_type) override {
    auto layers = this->layers();
    for (auto &layer : layers) {
      layer->set_engine_type(engine_type);
    }
  }

  void init_impl() override {
    auto layers = this->layers();
    for (auto &layer : layers) {
      layer->init();
    }
  }

  void on_set_allocator(DELAllocatorV2 &allocator) override {
    auto layers = this->layers();
    for (auto &layer : layers) {
      layer->set_allocator(allocator);
    }
  }
  void on_set_flow_handle(flowHandle_t handle) override {
    auto layers = this->layers();
    for (auto &layer : layers) {
      layer->set_flow_handle(handle);
    }
  }
  void on_set_seed(unsigned long long seed) override {
    auto layers = this->layers();
    for (auto &layer : layers) {
      layer->set_seed(seed);
    }
  }
  void on_set_training(bool training) override {
    auto layers = this->layers();
    for (auto &layer : layers) {
      layer->set_training(training);
    }
  }
  void on_set_io_dtype(DType_t dtype) override {
    auto layers = this->layers();
    for (auto &layer : layers) {
      layer->set_io_dtype(dtype);
    }
  }
  void on_set_param_dtype(DType_t dtype) override {
    auto layers = this->layers();
    for (auto &layer : layers) {
      layer->set_param_dtype(dtype);
    }
  }
  void on_set_compute_dtype(DType_t dtype) override {
    auto layers = this->layers();
    for (auto &layer : layers) {
      layer->set_compute_dtype(dtype);
    }
  }

  virtual Vec<Layer *> layers() = 0;
};
}  // namespace tnn