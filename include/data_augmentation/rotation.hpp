#pragma once

#include "augmentation.hpp"
#include <cmath>
#include <random>

namespace data_augmentation {

/**
 * Rotation augmentation
 */
template <typename T = float> class RotationAugmentation : public Augmentation<T> {
public:
  RotationAugmentation(float probability = 0.5f, float max_angle_degrees = 15.0f)
      : probability_(probability), max_angle_degrees_(max_angle_degrees) {
    this->name_ = "Rotation";
  }

  void apply(Tensor<T> &data, Tensor<T> &labels) override {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> angle_dist(-max_angle_degrees_, max_angle_degrees_);

    const auto shape = data.shape();
    if (shape.size() != 4)
      return;

    const size_t batch_size = shape[0];
    const size_t channels = shape[1];
    const size_t height = shape[2];
    const size_t width = shape[3];

    for (size_t b = 0; b < batch_size; ++b) {
      if (prob_dist(this->rng_) < probability_) {
        float angle_degrees = angle_dist(this->rng_);
        rotate_image(data, b, channels, height, width, angle_degrees);
      }
    }
  }

  std::unique_ptr<Augmentation<T>> clone() const override {
    return std::make_unique<RotationAugmentation<T>>(probability_, max_angle_degrees_);
  }

private:
  float probability_;
  float max_angle_degrees_;

  void rotate_image(Tensor<T> &data, size_t batch_idx, size_t channels, size_t height, size_t width,
                    float angle_degrees) {
    const float angle_rad = angle_degrees * M_PI / 180.0f;
    const float cos_angle = std::cos(angle_rad);
    const float sin_angle = std::sin(angle_rad);
    const float center_x = width / 2.0f;
    const float center_y = height / 2.0f;

    Tensor<T> rotated(1, channels, height, width);
    rotated.fill(static_cast<T>(0));

    for (size_t c = 0; c < channels; ++c) {
      for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
          float src_x = (x - center_x) * cos_angle - (y - center_y) * sin_angle + center_x;
          float src_y = (x - center_x) * sin_angle + (y - center_y) * cos_angle + center_y;

          int x1 = static_cast<int>(std::floor(src_x));
          int y1 = static_cast<int>(std::floor(src_y));
          int x2 = x1 + 1;
          int y2 = y1 + 1;

          if (x1 >= 0 && x2 < static_cast<int>(width) && y1 >= 0 && y2 < static_cast<int>(height)) {
            float wx = src_x - x1;
            float wy = src_y - y1;

            T val1 = data(batch_idx, c, y1, x1);
            T val2 = data(batch_idx, c, y1, x2);
            T val3 = data(batch_idx, c, y2, x1);
            T val4 = data(batch_idx, c, y2, x2);

            rotated(0, c, y, x) = val1 * (1 - wx) * (1 - wy) + val2 * wx * (1 - wy) +
                                  val3 * (1 - wx) * wy + val4 * wx * wy;
          }
        }
      }
    }

    // Copy back
    for (size_t c = 0; c < channels; ++c) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          data(batch_idx, c, h, w) = rotated(0, c, h, w);
        }
      }
    }
  }
};

} // namespace data_augmentation
