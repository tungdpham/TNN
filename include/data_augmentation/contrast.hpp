#pragma once

#include <algorithm>
#include <random>
#include <utility>
#include <vector>

#include "augmentation.hpp"
#include "threading/thread_handler.hpp"

namespace synet {

/**
 * Contrast adjustment augmentation
 */
class ContrastAugmentation : public Augmentation {
public:
  ContrastAugmentation(float probability = 0.5f, float contrast_range = 0.2f)
      : probability_(probability),
        contrast_range_(contrast_range) {
    this->name_ = "Contrast";
  }

  void apply(Tensor &data, Tensor &labels) override {
    DISPATCH_DTYPE(data.data_type(), T, apply_impl<T>(data, labels));
  }

  std::unique_ptr<Augmentation> clone() const override {
    return std::make_unique<ContrastAugmentation>(probability_, contrast_range_);
  }

private:
  float probability_;
  float contrast_range_;

  template <typename T>
  void apply_impl(Tensor &data, Tensor &labels) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> contrast_dist(1.0f - contrast_range_,
                                                        1.0f + contrast_range_);

    if (data.dims() != 4) return;

    size_t batch_size = data.dimension(0);
    size_t spatial_size = data.stride(0);
    T *ptr = data.data_as<T>();

    // Pre-compute per-batch random decisions sequentially to avoid data races
    std::vector<std::pair<bool, float>> decisions(batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
      bool apply = prob_dist(this->rng_) < probability_;
      decisions[b] = {apply, apply ? contrast_dist(this->rng_) : 1.0f};
    }

    parallel_for<size_t>(0, batch_size, [&](size_t b) {
      if (decisions[b].first) {
        const float contrast_factor = decisions[b].second;
        for (size_t i = 0; i < spatial_size; ++i) {
          size_t idx = b * spatial_size + i;
          ptr[idx] = std::clamp(static_cast<float>(ptr[idx]) * contrast_factor, 0.0f, 1.0f);
        }
      }
    });
  }
};

}  // namespace synet
