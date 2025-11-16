#pragma once

#include "augmentation.hpp"
#include <algorithm>
#include <random>

namespace tnn {

/**
 * Gaussian noise augmentation
 */
template <typename T = float> class GaussianNoiseAugmentation : public Augmentation<T> {
public:
  GaussianNoiseAugmentation(float probability = 0.3f, float noise_std = 0.05f)
      : probability_(probability), noise_std_(noise_std) {
    this->name_ = "GaussianNoise";
  }

  void apply(Tensor<T> &data, Tensor<T> &labels) override {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::normal_distribution<float> noise_dist(0.0f, noise_std_);

    const auto shape = data.shape();
    if (shape.size() != 4)
      return;

    const size_t batch_size = shape[0];

    for (size_t b = 0; b < batch_size; ++b) {
      if (prob_dist(this->rng_) < probability_) {
        for (size_t i = 0; i < data.size() / batch_size; ++i) {
          size_t idx = b * (data.size() / batch_size) + i;
          T *ptr = data.data() + idx;
          float noise = noise_dist(this->rng_);
          *ptr = std::clamp(*ptr + static_cast<T>(noise), static_cast<T>(0), static_cast<T>(1));
        }
      }
    }
  }

  std::unique_ptr<Augmentation<T>> clone() const override {
    return std::make_unique<GaussianNoiseAugmentation<T>>(probability_, noise_std_);
  }

private:
  float probability_;
  float noise_std_;
};

} // namespace tnn
