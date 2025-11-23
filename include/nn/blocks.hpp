#pragma once

#include "blocks_impl/residual_block.hpp"
#include "nn/layers_impl/base_layer.hpp"
#include <memory>
#include <vector>

namespace tnn {

template <typename T = float>
std::unique_ptr<ResidualBlock<T>> residual_block(std::vector<std::unique_ptr<Layer<T>>> main_path,
                                                 std::vector<std::unique_ptr<Layer<T>>> shortcut,
                                                 const std::string &activation_name = "relu",
                                                 const std::string &name = "") {
  return std::make_unique<ResidualBlock<T>>(
      std::move(main_path), std::move(shortcut), activation_name,
      name.empty() ? "residual_block_" + std::to_string(main_path.size()) : name);
}

} // namespace tnn