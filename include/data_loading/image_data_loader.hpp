#pragma once

#include "data_loader.hpp"

namespace tnn {
/**
 * Specialized base class for image classification datasets
 * Provides common functionality for image-based datasets like MNIST, CIFAR,
 * etc.
 */
template <typename T = float> class ImageDataLoader : public BaseDataLoader<T> {
public:
  virtual ~ImageDataLoader() = default;

  /**
   * Get number of classes
   */
  virtual int get_num_classes() const = 0;

  /**
   * Get class names (optional)
   */
  virtual std::vector<std::string> get_class_names() const {
    std::vector<std::string> names;
    int num_classes = get_num_classes();
    names.reserve(num_classes);
    for (int i = 0; i < num_classes; ++i) {
      names.push_back("class_" + std::to_string(i));
    }
    return names;
  }

protected:
  using BaseDataLoader<T>::current_index_;
  using BaseDataLoader<T>::batch_size_;
  using BaseDataLoader<T>::rng_;
};

} // namespace tnn