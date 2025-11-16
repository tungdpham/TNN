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
   * Get image dimensions
   */
  virtual std::vector<size_t> get_image_shape() const = 0;

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

  /**
   * Utility to copy image data to tensor with proper channel ordering
   */
  void copy_image_to_tensor(Tensor<T> &tensor, int batch_idx, const std::vector<T> &image_data,
                            const std::vector<size_t> &shape) {
    size_t channels = shape[0];
    size_t height = shape[1];
    size_t width = shape[2];

    for (size_t c = 0; c < channels; ++c) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          size_t idx = c * height * width + h * width + w;
          tensor(batch_idx, c, h, w) = image_data[idx];
        }
      }
    }
  }
};

} // namespace tnn