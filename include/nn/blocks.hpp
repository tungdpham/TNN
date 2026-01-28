#pragma once

#include "blocks_impl/residual_block.hpp"
#include "nn/layer.hpp"
#include <memory>
#include <vector>

namespace tnn {

inline std::unique_ptr<ResidualBlock> residual_block(std::vector<std::unique_ptr<Layer>> main_path,
                                                     std::vector<std::unique_ptr<Layer>> shortcut,
                                                     const std::string &activation_name = "relu",
                                                     const std::string &name = "") {
  return std::make_unique<ResidualBlock>(
      std::move(main_path), std::move(shortcut), activation_name,
      name.empty() ? "residual_block_" + std::to_string(main_path.size()) : name);
}

} // namespace tnn