#pragma once

#include <random>
#include <tuple>
#include <vector>

#include "augmentation.hpp"
#include "threading/thread_handler.hpp"

namespace tnn {

/**
 * Random crop augmentation with padding
 */
class RandomCropAugmentation : public Augmentation {
public:
  RandomCropAugmentation(float probability = 0.5f, int padding = 4)
      : probability_(probability),
        padding_(padding) {
    this->name_ = "RandomCrop";
  }

  void apply(Tensor &data, Tensor &labels) override {
    DISPATCH_DTYPE(data->data_type(), T, apply_impl<T>(data, labels));
  }

  std::unique_ptr<Augmentation> clone() const override {
    return std::make_unique<RandomCropAugmentation>(probability_, padding_);
  }

private:
  float probability_;
  int padding_;

  template <typename T>
  void apply_impl(Tensor &data, Tensor &labels) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    const auto shape = data->shape();
    if (shape.size() != 4) return;

    const size_t batch_size = shape[0];
    const size_t height = shape[1];
    const size_t width = shape[2];
    const size_t channels = shape[3];

    std::uniform_int_distribution<int> crop_dist(0, 2 * padding_);

    // Pre-compute per-batch random decisions sequentially to avoid data races
    std::vector<std::tuple<bool, int, int>> decisions(batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
      if (prob_dist(this->rng_) < probability_) {
        decisions[b] = {true, crop_dist(this->rng_), crop_dist(this->rng_)};
      } else {
        decisions[b] = {false, 0, 0};
      }
    }

    parallel_for<size_t>(0, batch_size, [&](size_t b) {
      if (std::get<0>(decisions[b])) {
        apply_crop<T>(data, b, height, width, channels, std::get<1>(decisions[b]),
                      std::get<2>(decisions[b]));
      }
    });
  }

  template <typename T>
  void apply_crop(const Tensor &data, size_t batch_idx, size_t height, size_t width,
                  size_t channels, int start_x, int start_y) {
    const size_t padded_size = width + 2 * padding_;
    Tensor padded = make_tensor<T>({1, padded_size, padded_size, channels});

    padded->fill(0.0);

    // Copy original image to center of padded image
    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        for (size_t c = 0; c < channels; ++c) {
          padded->at<T>({0, h + padding_, w + padding_, c}) = data->at<T>({batch_idx, h, w, c});
        }
      }
    }

    // Crop from padded image
    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        for (size_t c = 0; c < channels; ++c) {
          data->at<T>({batch_idx, h, w, c}) = padded->at<T>({0, start_y + h, start_x + w, c});
        }
      }
    }
  }
};

}  // namespace tnn
