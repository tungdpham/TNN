#pragma once

#include "augmentation.hpp"
#include <random>

namespace data_augmentation {

/**
 * Random crop augmentation with padding
 */
template <typename T = float> class RandomCropAugmentation : public Augmentation<T> {
public:
  RandomCropAugmentation(float probability = 0.5f, int padding = 4)
      : probability_(probability), padding_(padding) {
    this->name_ = "RandomCrop";
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

    std::uniform_int_distribution<int> crop_dist(0, 2 * padding_);

    for (size_t b = 0; b < batch_size; ++b) {
      if (prob_dist(this->rng_) < probability_) {
        int start_x = crop_dist(this->rng_);
        int start_y = crop_dist(this->rng_);
        apply_crop(data, b, channels, height, width, start_x, start_y);
      }
    }
  }

  std::unique_ptr<Augmentation<T>> clone() const override {
    return std::make_unique<RandomCropAugmentation<T>>(probability_, padding_);
  }

private:
  float probability_;
  int padding_;

  void apply_crop(Tensor<T> &data, size_t batch_idx, size_t channels, size_t height, size_t width,
                  int start_x, int start_y) {
    const int padded_size = width + 2 * padding_;
    Tensor<T> padded(1, channels, padded_size, padded_size);
    padded.fill(static_cast<T>(0));

    // Copy original image to center of padded image
    for (size_t c = 0; c < channels; ++c) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          padded(0, c, h + padding_, w + padding_) = data(batch_idx, c, h, w);
        }
      }
    }

    // Crop from padded image
    for (size_t c = 0; c < channels; ++c) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          data(batch_idx, c, h, w) = padded(0, c, start_y + h, start_x + w);
        }
      }
    }
  }
};

} // namespace data_augmentation
