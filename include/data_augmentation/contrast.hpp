#pragma once

#include "augmentation.hpp"
#include <algorithm>
#include <random>

namespace data_augmentation {

/**
 * Contrast adjustment augmentation
 */
template <typename T = float> class ContrastAugmentation : public Augmentation<T> {
public:
  ContrastAugmentation(float probability = 0.5f, float contrast_range = 0.2f)
      : probability_(probability), contrast_range_(contrast_range) {
    this->name_ = "Contrast";
  }

  void apply(Tensor<T> &data, Tensor<T> &labels) override {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> contrast_dist(1.0f - contrast_range_,
                                                        1.0f + contrast_range_);

    const auto shape = data.shape();
    if (shape.size() != 4)
      return;

    const size_t batch_size = shape[0];

    for (size_t b = 0; b < batch_size; ++b) {
      if (prob_dist(this->rng_) < probability_) {
        float contrast_factor = contrast_dist(this->rng_);

        for (size_t i = 0; i < data.size() / batch_size; ++i) {
          size_t idx = b * (data.size() / batch_size) + i;
          T *ptr = data.data() + idx;
          *ptr = std::clamp(*ptr * static_cast<T>(contrast_factor), static_cast<T>(0),
                            static_cast<T>(1));
        }
      }
    }
  }

  std::unique_ptr<Augmentation<T>> clone() const override {
    return std::make_unique<ContrastAugmentation<T>>(probability_, contrast_range_);
  }

private:
  float probability_;
  float contrast_range_;
};

} // namespace data_augmentation
