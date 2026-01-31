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
#include <vector>

#include "nn/layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {
struct Partition {
  size_t start_layer;
  size_t end_layer;  // exclusive

  Partition(size_t start, size_t end) : start_layer(start), end_layer(end) {}
};

class Sequential : public Layer {
private:
  std::vector<std::unique_ptr<Layer>> layers_;
  size_t max_size_ = 0;

  void compute_max_size(const std::vector<size_t> &input_shape, DType_t dtype);

protected:
  void init_impl() override;
  void on_set_io_dtype(DType_t dtype) override;
  void on_set_param_dtype(DType_t dtype) override;
  void on_set_compute_dtype(DType_t dtype) override;
  void on_set_device(const Device &device) override;
  void on_set_training(bool training) override;
  void forward_impl(const Tensor &input, Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const Tensor &gradient, Tensor &grad_input, size_t mb_id = 0) override;

public:
  explicit Sequential(const std::string &name = "seq",
                      std::vector<std::unique_ptr<Layer>> layers = {});

  Sequential();

  static constexpr const char *TYPE_NAME = "sequential";

  Sequential(const Sequential &) = delete;
  Sequential &operator=(const Sequential &) = delete;

  Sequential(Sequential &&) = default;
  Sequential &operator=(Sequential &&) = default;

  /**
   * @brief Returns a vector of pointers to all params in the model
   */
  std::vector<Tensor> parameters() override;

  /**
   * @brief Returns a vector of pointers to all gradients in the model
   */
  std::vector<Tensor> gradients() override;

  /**
   * @brief Returns the output shape for given input shape
   * @param input_shape The shape of the input tensor as a vector of sizes.
   */
  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;

  void print_summary(const std::vector<size_t> &input_shape) const;

  std::vector<Sequential> split(std::vector<Partition> &partitions) const;

  const std::vector<Layer *> &get_layers() const;

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;

  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;

  bool has_parameters() const override;

  std::string type() const override { return TYPE_NAME; }

  LayerConfig get_config() const override;

  static std::unique_ptr<Sequential> create_from_config(const LayerConfig &config);

  std::unique_ptr<Layer> clone() const override;

  size_t cached_memory_bytes() const override;
};

}  // namespace tnn