#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../tensor/tensor.hpp"
#include "../layers.hpp"

namespace tnn {

// Flatten Layer for converting from 4D to 2D tensors (for compatibility with dense layers)
template <typename T = float> class FlattenLayer : public StatelessLayer<T> {
private:
  std::unordered_map<int, std::vector<size_t>> micro_batch_original_shapes_;

public:
  explicit FlattenLayer(const std::string &name = "flatten");

  Tensor<T> forward(const Tensor<T> &input, int micro_batch_id = 0) override;
  Tensor<T> backward(const Tensor<T> &grad_output,
                     int micro_batch_id = 0) override;

  std::string type() const override;
  LayerConfig get_config() const override;
  std::unique_ptr<Layer<T>> clone() const override;

  std::vector<size_t>
  compute_output_shape(const std::vector<size_t> &input_shape) const override;

public:
  static std::unique_ptr<Layer<T>>
  create_from_config(const LayerConfig &config);
};

} // namespace tnn
