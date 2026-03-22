#include "nn/siso_layer.hpp"

namespace tnn {

Vec<Vec<size_t>> SISOLayer::output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
  if (input_shapes.size() != 1) {
    throw std::runtime_error("Only single input supported in output_shape for SISO layers.");
  }
  return {compute_output_shape(input_shapes[0])};
}

}  // namespace tnn