/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "nn/activations_impl/base_activation.hpp"
#include "nn/block.hpp"
#include "nn/layer.hpp"

namespace tnn {

/**
 * @brief Residual block implementing skip connections: output = F(x) + x
 *
 * Supports both identity shortcuts (when input/output dimensions match)
 * and projection shortcuts (1x1 conv when dimensions differ).
 */
class ResidualBlock : public Block {
private:
  std::vector<std::unique_ptr<Layer>> main_path_;
  std::vector<std::unique_ptr<Layer>> shortcut_path_;
  std::unique_ptr<ActivationFunction> final_activation_;
  std::unordered_map<size_t, Tensor> pre_activation_cache_;
  std::unordered_map<size_t, Vec<Vec<size_t>>> input_shape_cache_;
  std::string activation_type_;

  std::vector<Layer *> layers() override {
    std::vector<Layer *> layers;
    for (auto &layer : main_path_) {
      layers.push_back(layer.get());
    }
    for (auto &layer : shortcut_path_) {
      layers.push_back(layer.get());
    }
    return layers;
  }

  void forward_impl(const Vec<ConstTensor> &inputs, const Vec<Tensor> &outputs,
                    size_t mb_id = 0) override;
  void backward_impl(const Vec<ConstTensor> &grad_outputs, const Vec<Tensor> &grad_inputs,
                     size_t mb_id = 0) override;

public:
  /**
   * @brief Constructs a residual block
   * @param main_path The main transformation path F(x) as a vector of layers
   * @param shortcut_path Optional projection path for dimension matching (empty for identity)
   * @param final_activation Activation applied after addition (e.g., "relu")
   * @param name Layer name
   */
  ResidualBlock(std::vector<std::unique_ptr<Layer>> main_path,
                std::vector<std::unique_ptr<Layer>> shortcut_path,
                const std::string &final_activation = "relu",
                const std::string &name = "residual_block");

  ResidualBlock(const ResidualBlock &other);

  static constexpr const char *TYPE_NAME = "residual_block";

  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t inf_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  static std::unique_ptr<ResidualBlock> create_from_config(const LayerConfig &config);
};
}  // namespace tnn
