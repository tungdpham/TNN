#pragma once

#include "nn/layer.hpp"

namespace tnn {
class Block : public Layer {
public:
  Block(const std::string &name = "block") { this->name_ = name; }

protected:
  void init_impl() override = 0;
  void on_set_device(const Device &device) override = 0;
  void on_set_flow_handle(flowHandle_t handle) override = 0;
  void on_set_seed(unsigned long long seed) override {}
  void on_set_training(bool training) override = 0;
  void on_set_io_dtype(DType_t dtype) override = 0;
  void on_set_param_dtype(DType_t dtype) override = 0;
  void on_set_compute_dtype(DType_t dtype) override = 0;
  virtual std::vector<Tensor> parameters() override = 0;
  virtual std::vector<Tensor> gradients() override = 0;
};
}  // namespace tnn