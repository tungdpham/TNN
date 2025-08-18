#pragma once

#include "../layers.hpp"

namespace tnn {

// Flatten Layer for converting from 4D to 2D tensors (for compatibility with dense layers)
template <typename T = float> class FlattenLayer : public StatelessLayer<T> {
private:
  std::unordered_map<int, std::vector<size_t>> micro_batch_original_shapes_;

public:
  explicit FlattenLayer(const std::string &name = "flatten")
      : StatelessLayer<T>(name) {}

  Tensor<T> forward(const Tensor<T> &input, int micro_batch_id = 0) override {
    micro_batch_original_shapes_[micro_batch_id] = input.shape();

    size_t batch_size = input.batch_size();
    size_t features = input.channels() * input.height() * input.width();

    Tensor<T> output = input.reshape({batch_size, features, 1, 1});

    return output;
  }

  Tensor<T> backward(const Tensor<T> &grad_output,
                     int micro_batch_id = 0) override {
    auto it = micro_batch_original_shapes_.find(micro_batch_id);
    if (it == micro_batch_original_shapes_.end()) {
      throw std::runtime_error(
          "No cached shape found for micro-batch ID in FlattenLayer: " +
          std::to_string(micro_batch_id));
    }
    const std::vector<size_t> &original_shape = it->second;

    // Reshape back to original shape
    Tensor<T> grad_input = grad_output.reshape(original_shape);

    // micro_batch_original_shapes_.erase(it);
    return grad_input;
  }

  std::string type() const override { return "flatten"; }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = this->name_;
    return config;
  }

  std::unique_ptr<Layer<T>> clone() const override {
    return std::make_unique<FlattenLayer<T>>(this->name_);
  }

  std::vector<size_t>
  compute_output_shape(const std::vector<size_t> &input_shape) const override {
    if (input_shape.size() != 4) {
      throw std::invalid_argument("FlattenLayer expects 4D input");
    }

    size_t features = input_shape[1] * input_shape[2] * input_shape[3];
    return {input_shape[0], features, 1, 1};
  }
};

} // namespace tnn