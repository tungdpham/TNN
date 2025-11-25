#pragma once

#include "tensor/tensor.hpp"
#include "utils/accuracy_impl/cpu/accuracy.hpp"
#ifdef USE_CUDA
#include "utils/accuracy_impl/cuda/accuracy.hpp"
#endif

namespace tnn {

inline float compute_class_accuracy(const Tensor<float> &predictions,
                                    const Tensor<float> &targets) {
  const size_t batch_size = predictions.shape()[0];
  const size_t num_classes = predictions.shape()[1];

  if (predictions.device_type() == DeviceType::CPU) {
    return cpu::accuracy::compute_class_accuracy(predictions.data(), targets.data(), batch_size,
                                                 num_classes);
  }
#ifdef USE_CUDA
  return cuda::accuracy::compute_class_accuracy(predictions.data(), targets.data(), batch_size,
                                                num_classes);
#endif
  throw std::runtime_error("Unsupported device type for compute_class_accuracy.");
}

inline int compute_class_corrects(const Tensor<float> &predictions, const Tensor<float> &targets,
                                  float threshold = 0.5f) {
  const size_t batch_size = predictions.shape()[0];
  const size_t num_classes = predictions.shape()[1];

  if (predictions.device_type() == DeviceType::CPU) {
    return cpu::accuracy::compute_class_corrects(predictions.data(), targets.data(), batch_size,
                                                 num_classes, threshold);
  }
#ifdef USE_CUDA
  return cuda::accuracy::compute_class_corrects(predictions.data(), targets.data(), batch_size,
                                                num_classes, threshold);
#endif
  throw std::runtime_error("Unsupported device type for compute_class_corrects.");
}

} // namespace tnn