#pragma once

#include "augmentation.hpp"
#include <random>

namespace data_augmentation {

/**
 * Horizontal flip augmentation
 */
template <typename T = float> class HorizontalFlipAugmentation : public Augmentation<T> {
public:
  explicit HorizontalFlipAugmentation(float probability = 0.5f) : probability_(probability) {
    this->name_ = "HorizontalFlip";
  }

  void apply(Tensor<T> &data, Tensor<T> &labels) override {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    const auto shape = data.shape();
    if (shape.size() != 4)
      return; // Expected: [batch, channels, height, width]

    const size_t batch_size = shape[0];
    const size_t channels = shape[1];
    const size_t height = shape[2];
    const size_t width = shape[3];

    for (size_t b = 0; b < batch_size; ++b) {
      if (dist(this->rng_) < probability_) {
        for (size_t c = 0; c < channels; ++c) {
          for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width / 2; ++w) {
              std::swap(data(b, c, h, w), data(b, c, h, width - 1 - w));
            }
          }
        }
      }
    }
  }

  std::unique_ptr<Augmentation<T>> clone() const override {
    return std::make_unique<HorizontalFlipAugmentation<T>>(probability_);
  }

private:
  float probability_;
};

} // namespace data_augmentation
