/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "augmentation.hpp"
#include <array>

namespace tnn {

/**
 * Normalization augmentation
 * Normalizes tensor values using per-channel mean and standard deviation
 * Formula: output = (input - mean) / std
 *
 * Note: This should typically be applied AFTER all other augmentations
 * and after converting pixel values to [0, 1] range.
 */
template <typename T = float> class NormalizationAugmentation : public Augmentation<T> {
private:
  std::array<T, 3> mean_;
  std::array<T, 3> std_;

public:
  /**
   * Constructor
   * @param mean Per-channel mean values [R, G, B]
   * @param std Per-channel standard deviation values [R, G, B]
   */
  NormalizationAugmentation(const std::array<T, 3> &mean = {0.485f, 0.456f, 0.406f},
                            const std::array<T, 3> &std = {0.229f, 0.224f, 0.225f})
      : mean_(mean), std_(std) {
    this->name_ = "Normalization";
  }

  void apply(Tensor<T> &data, Tensor<T> &labels) override {

    const auto shape = data.shape();
    if (shape.size() != 4)
      return;

    const size_t batch_size = shape[0];
    const size_t channels = shape[1];
    const size_t height = shape[2];
    const size_t width = shape[3];

    if (channels != 3 && channels != 1) {
      throw std::invalid_argument("NormalizationAugmentation: unsupported number of channels");
    }

    // Apply normalization to each image in the batch
    for (size_t b = 0; b < batch_size; ++b) {
      // Normalize each channel
      for (size_t c = 0; c < channels; ++c) {
        T channel_mean = (channels == 3) ? mean_[c] : mean_[0];
        T channel_std = (channels == 3) ? std_[c] : std_[0];

        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            size_t idx = b * channels * height * width + c * height * width + h * width + w;
            data.data()[idx] = (data.data()[idx] - channel_mean) / channel_std;
          }
        }
      }
    }
  }

  std::unique_ptr<Augmentation<T>> clone() const override {
    return std::make_unique<NormalizationAugmentation<T>>(mean_, std_);
  }

  /**
   * Get mean values
   */
  const std::array<T, 3> &get_mean() const { return mean_; }

  /**
   * Get std values
   */
  const std::array<T, 3> &get_std() const { return std_; }
};

} // namespace tnn
