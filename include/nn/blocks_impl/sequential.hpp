/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <fmt/core.h>

#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "nn/block.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {
class Sequential : public Block {
private:
  std::vector<std::unique_ptr<Layer>> layers_;
  std::unordered_map<size_t, Vec<Vec<size_t>>> input_shapes_cache_;

  Vec<size_t> fwd_workspace_sizes(const std::vector<size_t> &shape);
  Vec<size_t> bwd_workspace_sizes(const std::vector<size_t> &shape);

protected:
  std::vector<Layer *> layers() override {
    std::vector<Layer *> layers;
    for (auto &layer : layers_) {
      layers.push_back(layer.get());
    }
    return layers;
  }

public:
  explicit Sequential(std::vector<std::unique_ptr<Layer>> layers = {},
                      const std::string &name = "sequential");

  static constexpr const char *TYPE_NAME = "sequential";

  std::string type() const override { return TYPE_NAME; }

  void forward(const Vec<ConstTensor> &inputs, const Vec<Tensor> &outputs, size_t mb_id) override;
  void backward(const Vec<ConstTensor> &grad_outputs, const Vec<Tensor> &grad_inputs,
                size_t mb_id) override;

  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t inf_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  void print_summary(const std::vector<size_t> &input_shape) const;
  std::vector<Layer *> get_layers();
  LayerConfig get_config() const override;
  static std::unique_ptr<Sequential> create_from_config(const LayerConfig &config);
};

}  // namespace tnn