#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../tensor/tensor.hpp"
#include "../activations.hpp"
#include "stateless_layer.hpp"

namespace tnn {

// Activation Layer (stateless)
template <typename T = float> class ActivationLayer : public StatelessLayer<T> {
private:
  std::unique_ptr<ActivationFunction<T>> activation_;
  std::unordered_map<int, Tensor<T>> micro_batch_inputs_;

public:
  explicit ActivationLayer(std::unique_ptr<ActivationFunction<T>> activation,
                           const std::string &name = "activation");

  Tensor<T> forward(const Tensor<T> &input, int micro_batch_id = 0) override;
  Tensor<T> backward(const Tensor<T> &grad_output,
                     int micro_batch_id = 0) override;

  std::string type() const override;
  LayerConfig get_config() const override;
  std::unique_ptr<Layer<T>> clone() const override;
  std::vector<size_t>
  compute_output_shape(const std::vector<size_t> &input_shape) const override;
};

} // namespace tnn