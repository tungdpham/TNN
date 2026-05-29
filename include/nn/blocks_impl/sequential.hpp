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
#include <type_traits>
#include <utility>

#include "nn/block.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {
class Sequential : public Block {
private:
  Vec<std::unique_ptr<Layer>> layers_;

protected:
  Vec<Layer *> layers() override {
    Vec<Layer *> layers;
    for (auto &layer : layers_) {
      layers.push_back(layer.get());
    }
    return layers;
  }

  Vec<Tensor> forward_impl(const Vec<ConstTensor> &inputs, size_t mb_id) override;
  Vec<Tensor> backward_impl(const Vec<ConstTensor> &grad_outputs, size_t mb_id) override;

public:
  explicit Sequential(Vec<std::unique_ptr<Layer>> layers = {},
                      const std::string &name = "sequential");

  template <typename... LayerArgs,
            typename = std::enable_if_t<
                (sizeof...(LayerArgs) > 0) &&
                (std::conjunction_v<std::is_base_of<Layer, std::decay_t<LayerArgs>>...>)>>
  explicit Sequential(LayerArgs &&...layers)
      : Block("sequential") {
    layers_.reserve(sizeof...(LayerArgs));
    (layers_.emplace_back(
         std::make_unique<std::decay_t<LayerArgs>>(std::forward<LayerArgs>(layers))),
     ...);
  }

  template <typename... LayerArgs,
            typename = std::enable_if_t<
                (sizeof...(LayerArgs) > 0) &&
                (std::conjunction_v<std::is_base_of<Layer, std::decay_t<LayerArgs>>...>)>>
  explicit Sequential(const std::string &name, LayerArgs &&...layers)
      : Block(name) {
    layers_.reserve(sizeof...(LayerArgs));
    (layers_.emplace_back(
         std::make_unique<std::decay_t<LayerArgs>>(std::forward<LayerArgs>(layers))),
     ...);
  }

  static constexpr const char *TYPE_NAME = "sequential";

  std::string type() const override { return TYPE_NAME; }

  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;
  void print_summary(const Vec<size_t> &input_shape) const;
  Vec<Layer *> get_layers();
  LayerConfig get_config() const override;
  static std::unique_ptr<Sequential> create_from_config(const LayerConfig &config);
};

}  // namespace tnn