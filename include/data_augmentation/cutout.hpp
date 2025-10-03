#pragma once

#include "augmentation.hpp"
#include <random>

namespace data_augmentation {

/**
 * Cutout augmentation (random erasing)
 */
template <typename T = float> class CutoutAugmentation : public Augmentation<T> {
public:
  CutoutAugmentation(float probability = 0.5f, int cutout_size = 8)
      : probability_(probability), cutout_size_(cutout_size) {
    this->name_ = "Cutout";
  }

  void apply(Tensor<T> &data, Tensor<T> &labels) override {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    const auto shape = data.shape();
    if (shape.size() != 4)
      return;

    const size_t batch_size = shape[0];
    const size_t channels = shape[1];
    const size_t height = shape[2];
    const size_t width = shape[3];

    for (size_t b = 0; b < batch_size; ++b) {
      if (prob_dist(this->rng_) < probability_) {
        std::uniform_int_distribution<int> x_dist(0, width - cutout_size_);
        std::uniform_int_distribution<int> y_dist(0, height - cutout_size_);

        int x = x_dist(this->rng_);
        int y = y_dist(this->rng_);

        for (size_t c = 0; c < channels; ++c) {
          for (int h = y; h < y + cutout_size_ && h < static_cast<int>(height); ++h) {
            for (int w = x; w < x + cutout_size_ && w < static_cast<int>(width); ++w) {
              data(b, c, h, w) = static_cast<T>(0);
            }
          }
        }
      }
    }
  }

  std::unique_ptr<Augmentation<T>> clone() const override {
    return std::make_unique<CutoutAugmentation<T>>(probability_, cutout_size_);
  }

private:
  float probability_;
  int cutout_size_;
};

} // namespace data_augmentation
