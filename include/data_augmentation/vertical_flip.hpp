#pragma once

#include <random>

#include "augmentation.hpp"

namespace tnn {

/**
 * Vertical flip augmentation
 */
class VerticalFlipAugmentation : public Augmentation {
public:
  explicit VerticalFlipAugmentation(float probability = 0.5f) : probability_(probability) {
    this->name_ = "VerticalFlip";
  }

  void apply(const Tensor &data, const Tensor &labels) override {
    DISPATCH_ON_DTYPE(data->data_type(), T, apply_impl<T>(data, labels));
  }

  std::unique_ptr<Augmentation> clone() const override {
    return std::make_unique<VerticalFlipAugmentation>(probability_);
  }

private:
  float probability_;

  template <typename T>
  void apply_impl(const Tensor &data, const Tensor &labels) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    const auto shape = data->shape();
    if (shape.size() != 4) return;

    const size_t batch_size = shape[0];
    const size_t height = shape[1];
    const size_t width = shape[2];
    const size_t channels = shape[3];

    for (size_t b = 0; b < batch_size; ++b) {
      if (dist(this->rng_) < probability_) {
        for (size_t h = 0; h < height / 2; ++h) {
          for (size_t w = 0; w < width; ++w) {
            for (size_t c = 0; c < channels; ++c) {
              std::swap(data->at<T>({b, h, w, c}), data->at<T>({b, height - 1 - h, w, c}));
            }
          }
        }
      }
    }
  }
};

}  // namespace tnn
