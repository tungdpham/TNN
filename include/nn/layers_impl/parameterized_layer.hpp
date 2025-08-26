#pragma once

#include "../../tensor/tensor.hpp"
#include "base_layer.hpp"
#include "../optimizers.hpp"

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace tnn {
// Base class for layers with parameters (dense, conv, etc.)
template <typename T = float> class ParameterizedLayer : public Layer<T> {
public:
  explicit ParameterizedLayer(const std::string &name = "") {
    this->name_ = name;
  }

  std::vector<Tensor<T> *> parameters() override;
  std::vector<Tensor<T> *> gradients() override;
  bool has_parameters() const override { return true; }
  void update_parameters() override; 
  void set_optimizer(std::unique_ptr<Optimizer<T>> optimizer);

protected:
  std::unique_ptr<Optimizer<T>> layer_optimizer_ = nullptr;  // Layer's own optimizer
  virtual void collect_parameters(std::vector<Tensor<T> *> &params) = 0;
  virtual void collect_gradients(std::vector<Tensor<T> *> &grads) = 0;
};
} // namespace tnn

#include "parameterized_layer.tpp"