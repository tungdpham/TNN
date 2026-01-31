/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "nn/activations_impl/base_activation.hpp"
#include "nn/layer.hpp"
#include "nn/layers_impl/parameterized_layer.hpp"

namespace tnn {

/**
 * @brief Residual block implementing skip connections: output = F(x) + x
 *
 * Supports both identity shortcuts (when input/output dimensions match)
 * and projection shortcuts (1x1 conv when dimensions differ).
 */
class ResidualBlock : public ParameterizedLayer {
private:
  std::vector<std::unique_ptr<Layer>> main_path_;
  std::vector<std::unique_ptr<Layer>> shortcut_path_;
  std::unique_ptr<ActivationFunction> final_activation_;
  std::unordered_map<size_t, Tensor> pre_activation_cache_;
  std::unordered_map<size_t, std::vector<size_t>> input_shape_cache_;
  std::string activation_type_;

  void init_params() override;
  void on_set_io_dtype(DType_t dtype) override;
  void on_set_param_dtype(DType_t dtype) override;
  void on_set_compute_dtype(DType_t dtype) override;
  void on_set_training(bool training) override;
  void on_set_device(const Device &device) override;
  void forward_impl(const Tensor &input, Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const Tensor &gradient, Tensor &grad_input, size_t mb_id = 0) override;
  void collect_parameters(std::vector<Tensor> &params) override;
  void collect_gradients(std::vector<Tensor> &grads) override;

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

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;
  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;
  std::string type() const override { return TYPE_NAME; }

  LayerConfig get_config() const override;
  static std::unique_ptr<ResidualBlock> create_from_config(const LayerConfig &config);

  std::unique_ptr<Layer> clone() const override;

  const std::vector<std::unique_ptr<Layer>> &get_main_path() const;
  const std::vector<std::unique_ptr<Layer>> &get_shortcut_path() const;

  size_t cached_memory_bytes() const override;
};
}  // namespace tnn
