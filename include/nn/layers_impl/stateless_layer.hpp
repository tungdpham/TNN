/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#pragma once

#include <string>

#include "nn/siso_layer.hpp"

namespace tnn {

class StatelessLayer : public SISOLayer {
public:
  explicit StatelessLayer(const std::string &name = "") { this->name_ = name; }

protected:
  std::vector<ParamDescriptor> param_descriptors() override { return {}; }
};

}  // namespace tnn