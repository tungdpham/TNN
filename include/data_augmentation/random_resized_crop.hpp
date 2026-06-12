#pragma once

#include <cmath>
#include <random>
#include <vector>

#include "augmentation.hpp"
#include "device/flow.hpp"
#include "device/pool_allocator.hpp"
#include "tensor/tensor.hpp"
#include "threading/thread_handler.hpp"

namespace synet {

/**
 * Torchvision-like RandomResizedCrop augmentation.
 *
 * For each image in the batch:
 *   1. Attempts up to max_attempts times to find a crop whose area falls
 *      within [scale_min, scale_max] of the image area and whose aspect
 *      ratio (w/h) falls within [ratio_min, ratio_max].
 *   2. Falls back to a centre-crop that satisfies the ratio bounds if no
 *      valid crop is found.
 *   3. Bilinearly resizes the crop to (out_h, out_w).
 *
 * Default parameters match torchvision.transforms.RandomResizedCrop(224):
 *   scale  = (0.08, 1.0)
 *   ratio  = (3/4, 4/3)
 */
class RandomResizedCropAugmentation : public Augmentation {
public:
  RandomResizedCropAugmentation(size_t out_h = 224, size_t out_w = 224, float scale_min = 0.08f,
                                float scale_max = 1.0f, float ratio_min = 3.0f / 4.0f,
                                float ratio_max = 4.0f / 3.0f, int max_attempts = 10)
      : out_h_(out_h),
        out_w_(out_w),
        scale_min_(scale_min),
        scale_max_(scale_max),
        log_ratio_min_(std::log(ratio_min)),
        log_ratio_max_(std::log(ratio_max)),
        max_attempts_(max_attempts) {
    this->name_ = "RandomResizedCrop";
  }

  void apply(Tensor &data, Tensor &labels) override {
    DISPATCH_DTYPE(data->data_type(), T, apply_impl<T>(data, labels));
  }

  std::unique_ptr<Augmentation> clone() const override {
    return std::make_unique<RandomResizedCropAugmentation>(out_h_, out_w_, scale_min_, scale_max_,
                                                           std::exp(log_ratio_min_),
                                                           std::exp(log_ratio_max_), max_attempts_);
  }

private:
  size_t out_h_;
  size_t out_w_;
  float scale_min_;
  float scale_max_;
  float log_ratio_min_;
  float log_ratio_max_;
  int max_attempts_;

  // Per-image crop parameters: {crop_x, crop_y, crop_w, crop_h}
  struct CropParams {
    int x, y, w, h;
  };

  CropParams sample_crop(int img_w, int img_h) {
    std::uniform_real_distribution<float> scale_dist(scale_min_, scale_max_);
    std::uniform_real_distribution<float> ratio_dist(log_ratio_min_, log_ratio_max_);

    const int area = img_w * img_h;

    for (int attempt = 0; attempt < max_attempts_; ++attempt) {
      const float target_area = scale_dist(this->rng_) * static_cast<float>(area);
      const float aspect = std::exp(ratio_dist(this->rng_));

      const int w = static_cast<int>(std::round(std::sqrt(target_area * aspect)));
      const int h = static_cast<int>(std::round(std::sqrt(target_area / aspect)));

      if (w > 0 && h > 0 && w <= img_w && h <= img_h) {
        std::uniform_int_distribution<int> x_dist(0, img_w - w);
        std::uniform_int_distribution<int> y_dist(0, img_h - h);
        return {x_dist(this->rng_), y_dist(this->rng_), w, h};
      }
    }

    // Torchvision fallback: centre-crop respecting ratio bounds.
    const float in_ratio = static_cast<float>(img_w) / static_cast<float>(img_h);
    int crop_w, crop_h;
    if (in_ratio < std::exp(log_ratio_min_)) {
      crop_w = img_w;
      crop_h = static_cast<int>(std::round(crop_w / std::exp(log_ratio_min_)));
    } else if (in_ratio > std::exp(log_ratio_max_)) {
      crop_h = img_h;
      crop_w = static_cast<int>(std::round(crop_h * std::exp(log_ratio_max_)));
    } else {
      crop_w = img_w;
      crop_h = img_h;
    }
    crop_w = std::min(crop_w, img_w);
    crop_h = std::min(crop_h, img_h);
    return {std::max(0, (img_w - crop_w) / 2), std::max(0, (img_h - crop_h) / 2), crop_w, crop_h};
  }

  // Bilinear resize of a single crop region into a (out_h_, out_w_) destination.
  template <typename T>
  void bilinear_resize_crop(const Tensor &src, size_t batch_idx, const CropParams &crop,
                            const Tensor &dst) {
    const size_t channels = src->shape()[3];

    for (size_t dy = 0; dy < out_h_; ++dy) {
      for (size_t dx = 0; dx < out_w_; ++dx) {
        // Map destination pixel to source crop coordinates.
        const float sx = (static_cast<float>(dx) + 0.5f) * static_cast<float>(crop.w) /
                             static_cast<float>(out_w_) -
                         0.5f;
        const float sy = (static_cast<float>(dy) + 0.5f) * static_cast<float>(crop.h) /
                             static_cast<float>(out_h_) -
                         0.5f;

        const int x0 = static_cast<int>(std::floor(sx));
        const int y0 = static_cast<int>(std::floor(sy));
        const int x1 = x0 + 1;
        const int y1 = y0 + 1;

        // Clamp to crop bounds.
        const int cx0 = std::max(0, std::min(x0, crop.w - 1));
        const int cy0 = std::max(0, std::min(y0, crop.h - 1));
        const int cx1 = std::max(0, std::min(x1, crop.w - 1));
        const int cy1 = std::max(0, std::min(y1, crop.h - 1));

        const float wx = sx - static_cast<float>(x0);
        const float wy = sy - static_cast<float>(y0);

        for (size_t c = 0; c < channels; ++c) {
          const T v00 = src->at<T>(
              {batch_idx, static_cast<size_t>(crop.y + cy0), static_cast<size_t>(crop.x + cx0), c});
          const T v10 = src->at<T>(
              {batch_idx, static_cast<size_t>(crop.y + cy0), static_cast<size_t>(crop.x + cx1), c});
          const T v01 = src->at<T>(
              {batch_idx, static_cast<size_t>(crop.y + cy1), static_cast<size_t>(crop.x + cx0), c});
          const T v11 = src->at<T>(
              {batch_idx, static_cast<size_t>(crop.y + cy1), static_cast<size_t>(crop.x + cx1), c});

          dst->at<T>({0, dy, dx, c}) = v00 * static_cast<T>((1.0f - wx) * (1.0f - wy)) +
                                       v10 * static_cast<T>(wx * (1.0f - wy)) +
                                       v01 * static_cast<T>((1.0f - wx) * wy) +
                                       v11 * static_cast<T>(wx * wy);
        }
      }
    }
  }

  template <typename T>
  void apply_impl(Tensor &data, Tensor & /*labels*/) {
    const auto shape = data->shape();
    if (shape.size() != 4) return;

    const size_t batch_size = shape[0];
    const size_t in_h = shape[1];
    const size_t in_w = shape[2];
    const size_t channels = shape[3];

    std::vector<CropParams> params(batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
      params[b] = sample_crop(static_cast<int>(in_w), static_cast<int>(in_h));
    }

    PoolAllocator &allocator = PoolAllocator::instance(data->device(), defaultFlowHandle);
    Tensor output =
        make_tensor(allocator, data->data_type(), {batch_size, out_h_, out_w_, channels});

    parallel_for<size_t>(0, batch_size, [&](size_t b) {
      Tensor buf = make_tensor(allocator, data->data_type(), {1, out_h_, out_w_, channels});
      bilinear_resize_crop<T>(data, b, params[b], buf);

      for (size_t h = 0; h < out_h_; ++h) {
        for (size_t w = 0; w < out_w_; ++w) {
          for (size_t c = 0; c < channels; ++c) {
            output->at<T>({b, h, w, c}) = buf->at<T>({0, h, w, c});
          }
        }
      }
    });

    data = std::move(output);
  }
};

}  // namespace synet
