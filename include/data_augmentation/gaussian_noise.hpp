#pragma once

#include <algorithm>
#include <random>
#include <vector>

#include "augmentation.hpp"
#include "threading/thread_handler.hpp"

namespace synet {

/**
 * Gaussian noise augmentation
 */
class GaussianNoiseAugmentation : public Augmentation {
public:
  GaussianNoiseAugmentation(float probability = 0.3f, float noise_std = 0.05f)
      : probability_(probability),
        noise_std_(noise_std) {
    this->name_ = "GaussianNoise";
  }

  void apply(Tensor &data, Tensor &labels) override {
    DISPATCH_DTYPE(data.dtype(), T, apply_impl<T>(data, labels));
  }

  std::unique_ptr<Augmentation> clone() const override {
    return std::make_unique<GaussianNoiseAugmentation>(probability_, noise_std_);
  }

private:
  float probability_;
  float noise_std_;

  template <typename T>
  void apply_impl(Tensor &data, Tensor &labels) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    if (data.dims() != 4) return;

    size_t batch_size = data.dimension(0);
    size_t spatial_size = data.stride(0);
    T *ptr = data.data_as<T>();

    // Pre-compute per-batch apply flags sequentially to avoid data races
    std::vector<bool> apply_flags(batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
      apply_flags[b] = prob_dist(this->rng_) < probability_;
    }

    const float noise_std = noise_std_;
    parallel_for<size_t>(0, batch_size, [&](size_t b) {
      if (apply_flags[b]) {
        // Thread-local RNG for per-element noise generation
        thread_local std::mt19937 local_rng{std::random_device{}()};
        std::normal_distribution<float> noise_dist(0.0f, noise_std);
        for (size_t i = 0; i < spatial_size; ++i) {
          size_t idx = b * spatial_size + i;
          ptr[idx] = std::clamp(static_cast<float>(ptr[idx]) + noise_dist(local_rng), 0.0f, 1.0f);
        }
      }
    });
  }
};

}  // namespace synet
