#pragma once

#include <algorithm>
#include <random>
#include <utility>
#include <vector>

#include "augmentation.hpp"
#include "threading/thread_handler.hpp"

namespace tnn {

class SaturationAugmentation : public Augmentation {
public:
  SaturationAugmentation(float probability = 0.5f, float saturation_range = 0.1f)
      : probability_(probability),
        saturation_range_(saturation_range) {
    this->name_ = "Saturation";
  }

  void apply(const Tensor &data, const Tensor &labels) override {
    DISPATCH_DTYPE(data->data_type(), T, apply_impl<T>(data, labels));
  }

  std::unique_ptr<Augmentation> clone() const override {
    return std::make_unique<SaturationAugmentation>(probability_, saturation_range_);
  }

private:
  float probability_;
  float saturation_range_;

  template <typename T>
  void apply_impl(const Tensor &data, const Tensor &labels) {
    if (data->dims() != 4 || data->dimension(3) != 3) return;

    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> factor_dist(1.0f - saturation_range_,
                                                      1.0f + saturation_range_);

    const size_t batch_size = data->dimension(0);
    const size_t height = data->dimension(1);
    const size_t width = data->dimension(2);
    const size_t channels = data->dimension(3);

    T *ptr = data->data_as<T>();

    // Pre-compute per-batch random decisions sequentially to avoid data races
    std::vector<std::pair<bool, float>> decisions(batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
      bool apply = prob_dist(this->rng_) < probability_;
      decisions[b] = {apply, apply ? factor_dist(this->rng_) : 1.0f};
    }

    parallel_for<size_t>(0, batch_size, [&](size_t b) {
      if (!decisions[b].first) return;

      const float factor = decisions[b].second;

      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          const size_t idx = b * height * width * channels + h * width * channels + w * channels;
          const float r = static_cast<float>(ptr[idx + 0]);
          const float g = static_cast<float>(ptr[idx + 1]);
          const float bval = static_cast<float>(ptr[idx + 2]);

          const float gray = 0.299f * r + 0.587f * g + 0.114f * bval;

          ptr[idx + 0] = static_cast<T>(std::clamp(gray + (r - gray) * factor, 0.0f, 1.0f));
          ptr[idx + 1] = static_cast<T>(std::clamp(gray + (g - gray) * factor, 0.0f, 1.0f));
          ptr[idx + 2] = static_cast<T>(std::clamp(gray + (bval - gray) * factor, 0.0f, 1.0f));
        }
      }
    });
  }
};

}  // namespace tnn
