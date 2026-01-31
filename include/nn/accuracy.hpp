#pragma once

#include "accuracy_impl/cpu/accuracy.hpp"
#include "tensor/tensor.hpp"
#ifdef USE_CUDA
#include "accuracy_impl/cuda/accuracy.hpp"
#endif

namespace tnn {

namespace detail {

template <typename T>
inline float compute_class_accuracy_impl(const Tensor &predictions, const Tensor &targets,
                                         size_t batch_size, size_t num_classes) {
  if (predictions->device_type() == DeviceType::CPU) {
    return cpu::accuracy::compute_class_accuracy(predictions->data_as<T>(), targets->data_as<T>(),
                                                 batch_size, num_classes);
  }
#ifdef USE_CUDA
  else {
    return cuda::accuracy::compute_class_accuracy(predictions->data_as<T>(), targets->data_as<T>(),
                                                  batch_size, num_classes);
  }
#endif
  throw std::runtime_error("Unsupported device type for compute_class_accuracy.");
}

template <typename T>
inline int compute_class_corrects_impl(const Tensor &predictions, const Tensor &targets,
                                       size_t batch_size, size_t num_classes, float threshold) {
  if (predictions->device_type() == DeviceType::CPU) {
    return cpu::accuracy::compute_class_corrects(predictions->data_as<T>(), targets->data_as<T>(),
                                                 batch_size, num_classes, threshold);
  }
#ifdef USE_CUDA
  else {
    return cuda::accuracy::compute_class_corrects(predictions->data_as<T>(), targets->data_as<T>(),
                                                  batch_size, num_classes, threshold);
  }
#endif
  throw std::runtime_error("Unsupported device type for compute_class_corrects.");
}

}  // namespace detail

inline float compute_class_accuracy(const Tensor &predictions, const Tensor &targets) {
  const size_t batch_size = predictions->shape()[0];
  const size_t num_classes = predictions->shape()[1];

  DISPATCH_ON_DTYPE(
      predictions->data_type(), T,
      return detail::compute_class_accuracy_impl<T>(predictions, targets, batch_size, num_classes));
}

inline int compute_class_corrects(const Tensor &predictions, const Tensor &targets,
                                  float threshold = 0.0f) {
  const size_t batch_size = predictions->shape()[0];
  const size_t num_classes = predictions->shape()[1];

  DISPATCH_ON_DTYPE(predictions->data_type(), T,
                    return detail::compute_class_corrects_impl<T>(predictions, targets, batch_size,
                                                                  num_classes, threshold));
}

}  // namespace tnn