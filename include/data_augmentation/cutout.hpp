#pragma once

#include <random>
#include <tuple>
#include <vector>

#include "augmentation.hpp"
#include "threading/thread_handler.hpp"

namespace synet {

/**
 * Cutout augmentation (random erasing)
 */
class CutoutAugmentation : public Augmentation {
public:
  CutoutAugmentation(float probability = 0.5f, int cutout_size = 8)
      : probability_(probability),
        cutout_size_(cutout_size) {
    this->name_ = "Cutout";
  }

  void apply(Tensor &data, Tensor &labels) override {
    DISPATCH_DTYPE(data->data_type(), T, apply_impl<T>(data, labels));
  }

  std::unique_ptr<Augmentation> clone() const override {
    return std::make_unique<CutoutAugmentation>(probability_, cutout_size_);
  }

private:
  float probability_;
  int cutout_size_;

  template <typename T>
  void apply_impl(Tensor &data, Tensor &labels) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    const auto shape = data->shape();
    if (shape.size() != 4) return;

    const size_t batch_size = shape[0];
    const size_t height = shape[1];
    const size_t width = shape[2];
    const size_t channels = shape[3];

    std::uniform_int_distribution<size_t> x_dist(0, width - cutout_size_);
    std::uniform_int_distribution<size_t> y_dist(0, height - cutout_size_);

    // Pre-compute per-batch random decisions sequentially to avoid data races
    std::vector<std::tuple<bool, size_t, size_t>> decisions(batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
      if (prob_dist(this->rng_) < probability_) {
        decisions[b] = {true, x_dist(this->rng_), y_dist(this->rng_)};
      } else {
        decisions[b] = {false, 0, 0};
      }
    }

    parallel_for<size_t>(0, batch_size, [&](size_t b) {
      if (std::get<0>(decisions[b])) {
        const size_t x = std::get<1>(decisions[b]);
        const size_t y = std::get<2>(decisions[b]);
        for (size_t h = y; h < y + cutout_size_ && h < height; ++h) {
          for (size_t w = x; w < x + cutout_size_ && w < width; ++w) {
            for (size_t c = 0; c < channels; ++c) {
              data->at<T>({b, h, w, c}) = static_cast<T>(0);
            }
          }
        }
      }
    });
  }
};

}  // namespace synet
