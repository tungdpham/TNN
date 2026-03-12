#pragma once

#include <random>

#include "augmentation.hpp"

namespace tnn {

/**
 * Horizontal flip augmentation
 */
class HorizontalFlipAugmentation : public Augmentation {
public:
  explicit HorizontalFlipAugmentation(float probability = 0.5f)
      : probability_(probability) {
    this->name_ = "HorizontalFlip";
  }

  void apply(const Tensor &data, const Tensor &labels) override {
    DISPATCH_DTYPE(data->data_type(), T, apply_impl<T>(data, labels));
  }

  std::unique_ptr<Augmentation> clone() const override {
    return std::make_unique<HorizontalFlipAugmentation>(probability_);
  }

private:
  float probability_;

  template <typename T>
  void apply_impl(const Tensor &data, const Tensor &labels) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    const auto shape = data->shape();
    if (shape.size() != 4) return;  // Expected: [batch, height, width, channels]

    const size_t batch_size = shape[0];
    const size_t height = shape[1];
    const size_t width = shape[2];
    const size_t channels = shape[3];

    for (size_t b = 0; b < batch_size; ++b) {
      if (dist(this->rng_) < probability_) {
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width / 2; ++w) {
            for (size_t c = 0; c < channels; ++c) {
              std::swap(data->at<T>({b, h, w, c}), data->at<T>({b, h, width - 1 - w, c}));
            }
          }
        }
      }
    }
  }
};

}  // namespace tnn
