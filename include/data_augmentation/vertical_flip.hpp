#pragma once

#include "augmentation.hpp"
#include <random>

namespace data_augmentation {

/**
 * Vertical flip augmentation
 */
template <typename T = float> class VerticalFlipAugmentation : public Augmentation<T> {
public:
  explicit VerticalFlipAugmentation(float probability = 0.5f) : probability_(probability) {
    this->name_ = "VerticalFlip";
  }

  void apply(Tensor<T> &data, Tensor<T> &labels) override {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    const auto shape = data.shape();
    if (shape.size() != 4)
      return;

    const size_t batch_size = shape[0];
    const size_t channels = shape[1];
    const size_t height = shape[2];
    const size_t width = shape[3];

    for (size_t b = 0; b < batch_size; ++b) {
      if (dist(this->rng_) < probability_) {
        for (size_t c = 0; c < channels; ++c) {
          for (size_t h = 0; h < height / 2; ++h) {
            for (size_t w = 0; w < width; ++w) {
              std::swap(data(b, c, h, w), data(b, c, height - 1 - h, w));
            }
          }
        }
      }
    }
  }

  std::unique_ptr<Augmentation<T>> clone() const override {
    return std::make_unique<VerticalFlipAugmentation<T>>(probability_);
  }

private:
  float probability_;
};

} // namespace data_augmentation
