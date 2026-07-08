#pragma once

#include "nn/layer.hpp"

namespace tunx {
class Block : public LayerImpl {
public:
  Block(const std::string &name = "block")
      : LayerImpl(name) {}

  Vec<ParamDescriptor> param_descriptors() override {
    Vec<ParamDescriptor> descriptors;
    for (const Layer &layer : layers()) {
      const auto &layer_descriptors = layer.param_descriptors();
      descriptors.insert(descriptors.end(), layer_descriptors.begin(), layer_descriptors.end());
    }
    return descriptors;
  }

  virtual Vec<Layer> layers() = 0;

protected:
  void on_set_engine_type(EngineType engine_type) override {
    Vec<Layer> layers = this->layers();
    for (Layer &layer : layers) {
      layer.set_engine_type(engine_type);
    }
  }

  void init_impl() override {
    Vec<Layer> layers = this->layers();
    for (Layer &layer : layers) {
      layer.init();
    }
  }

  void on_set_allocator(DELAllocatorV2 &allocator) override {
    Vec<Layer> layers = this->layers();
    for (Layer &layer : layers) {
      layer.set_allocator(allocator);
    }
  }

  void on_set_flow_handle(flowHandle_t handle) override {
    Vec<Layer> layers = this->layers();
    for (Layer &layer : layers) {
      layer.set_flow_handle(handle);
    }
  }

  void on_set_seed(unsigned long long seed) override {
    Vec<Layer> layers = this->layers();
    for (Layer &layer : layers) {
      layer.set_seed(seed);
    }
  }

  void on_set_training(bool training) override {
    Vec<Layer> layers = this->layers();
    for (Layer &layer : layers) {
      layer.set_training(training);
    }
  }

  void on_set_io_dtype(DType_t dtype) override {
    Vec<Layer> layers = this->layers();
    for (Layer &layer : layers) {
      layer.set_io_dtype(dtype);
    }
  }

  void on_set_param_dtype(DType_t dtype) override {
    Vec<Layer> layers = this->layers();
    for (Layer &layer : layers) {
      layer.set_param_dtype(dtype);
    }
  }

  void on_set_compute_dtype(DType_t dtype) override {
    Vec<Layer> layers = this->layers();
    for (Layer &layer : layers) {
      layer.set_compute_dtype(dtype);
    }
  }
};
}  // namespace tunx