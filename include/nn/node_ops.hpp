#pragma once

#include "nn/graph_api.hpp"
#include "nn/layers_impl/add_layer.hpp"
#include "nn/layers_impl/div_layer.hpp"
#include "nn/layers_impl/mul_layer.hpp"
#include "nn/layers_impl/sub_layer.hpp"

namespace tnn {
namespace graph_api_v2 {
inline Node operator+(const Node &a, const Node &b) {
  auto add_layer = make_layer<AddLayerImpl>();
  return add_layer(a, b);
}
inline Node operator-(const Node &a, const Node &b) {
  auto sub_layer = make_layer<SubLayerImpl>();
  return sub_layer(a, b);
}
inline Node operator*(const Node &a, const Node &b) {
  auto mul_layer = make_layer<MulLayerImpl>();
  return mul_layer(a, b);
}
inline Node operator/(const Node &a, const Node &b) {
  auto div_layer = make_layer<DivLayerImpl>();
  return div_layer(a, b);
}
}  // namespace graph_api_v2
}  // namespace tnn