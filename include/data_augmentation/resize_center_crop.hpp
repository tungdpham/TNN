#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "augmentation.hpp"
#include "device/flow.hpp"
#include "device/pool_allocator.hpp"
#include "tensor/tensor.hpp"
#include "threading/thread_handler.hpp"

namespace synet {

/**
 * Torchvision-like Resize(resize_short_side) + CenterCrop(crop_h, crop_w).
 *
 * For each image in the batch:
 *   1. Bilinearly resizes so that the shorter spatial dimension equals
 *      resize_short_side, preserving aspect ratio.
 *   2. Takes a centre-crop of (crop_h, crop_w).
 *
 * Default parameters match the torchvision ImageNet validation pipeline:
 *   resize_short_side = 256
 *   crop_h = crop_w = 224
 *
 * NOTE: The input tensor must have spatial dimensions >= crop_h x crop_w
 * after the resize step.  If the input is already crop_h x crop_w the
 * resize step is skipped and only the centre-crop is applied (which is a
 * no-op in that case too).
 */
class ResizeCenterCropAugmentation : public Augmentation {
public:
  ResizeCenterCropAugmentation(int resize_short_side = 256, size_t crop_h = 224,
                               size_t crop_w = 224)
      : resize_short_side_(resize_short_side),
        crop_h_(crop_h),
        crop_w_(crop_w) {
    this->name_ = "ResizeCenterCrop";
  }

  void apply(Tensor &data, Tensor &labels) override {
    DISPATCH_DTYPE(data.dtype(), T, apply_impl<T>(data, labels));
  }

  std::unique_ptr<Augmentation> clone() const override {
    return std::make_unique<ResizeCenterCropAugmentation>(resize_short_side_, crop_h_, crop_w_);
  }

private:
  int resize_short_side_;
  size_t crop_h_;
  size_t crop_w_;

  // Bilinear resize from (src_h x src_w) into an already-allocated dst tensor.
  // src and dst are both single-image tensors with shape {1, H, W, C}.
  template <typename T>
  void bilinear_resize(const Tensor &src, size_t src_h, size_t src_w, Tensor &dst, size_t dst_h,
                       size_t dst_w, size_t channels) {
    for (size_t dy = 0; dy < dst_h; ++dy) {
      for (size_t dx = 0; dx < dst_w; ++dx) {
        const float sx = (static_cast<float>(dx) + 0.5f) * static_cast<float>(src_w) /
                             static_cast<float>(dst_w) -
                         0.5f;
        const float sy = (static_cast<float>(dy) + 0.5f) * static_cast<float>(src_h) /
                             static_cast<float>(dst_h) -
                         0.5f;

        const int x0 = static_cast<int>(std::floor(sx));
        const int y0 = static_cast<int>(std::floor(sy));

        const int cx0 = std::max(0, std::min(x0, static_cast<int>(src_w) - 1));
        const int cy0 = std::max(0, std::min(y0, static_cast<int>(src_h) - 1));
        const int cx1 = std::max(0, std::min(x0 + 1, static_cast<int>(src_w) - 1));
        const int cy1 = std::max(0, std::min(y0 + 1, static_cast<int>(src_h) - 1));

        const float wx = sx - static_cast<float>(x0);
        const float wy = sy - static_cast<float>(y0);

        for (size_t c = 0; c < channels; ++c) {
          const T v00 = src.at<T>({0, static_cast<size_t>(cy0), static_cast<size_t>(cx0), c});
          const T v10 = src.at<T>({0, static_cast<size_t>(cy0), static_cast<size_t>(cx1), c});
          const T v01 = src.at<T>({0, static_cast<size_t>(cy1), static_cast<size_t>(cx0), c});
          const T v11 = src.at<T>({0, static_cast<size_t>(cy1), static_cast<size_t>(cx1), c});

          dst.at<T>({0, dy, dx, c}) = v00 * static_cast<T>((1.0f - wx) * (1.0f - wy)) +
                                      v10 * static_cast<T>(wx * (1.0f - wy)) +
                                      v01 * static_cast<T>((1.0f - wx) * wy) +
                                      v11 * static_cast<T>(wx * wy);
        }
      }
    }
  }

  template <typename T>
  void apply_impl(Tensor &data, Tensor & /*labels*/) {
    const auto shape = data.shape();
    if (shape.size() != 4) return;

    size_t batch_size = shape[0];
    size_t in_h = shape[1];
    size_t in_w = shape[2];
    size_t channels = shape[3];

    PoolAllocator &allocator = PoolAllocator::instance(data.device(), defaultFlowHandle);
    Tensor output(Vec<size_t>{batch_size, crop_h_, crop_w_, channels}, data.dtype(), allocator);

    parallel_for<size_t>(0, batch_size, [&](size_t b) {
      size_t resized_h, resized_w;
      if (in_h <= in_w) {
        resized_h = static_cast<size_t>(resize_short_side_);
        resized_w = static_cast<size_t>(
            std::round(static_cast<float>(in_w) * resize_short_side_ / static_cast<float>(in_h)));
      } else {
        resized_w = static_cast<size_t>(resize_short_side_);
        resized_h = static_cast<size_t>(
            std::round(static_cast<float>(in_h) * resize_short_side_ / static_cast<float>(in_w)));
      }
      resized_w = std::max(resized_w, crop_w_);
      resized_h = std::max(resized_h, crop_h_);

      Tensor src_buf(Vec<size_t>{1, in_h, in_w, channels}, data.dtype(), allocator);
      for (size_t h = 0; h < in_h; ++h)
        for (size_t w = 0; w < in_w; ++w)
          for (size_t c = 0; c < channels; ++c)
            src_buf.at<T>({0, h, w, c}) = data.at<T>({b, h, w, c});

      Tensor resized_buf(Vec<size_t>{1, resized_h, resized_w, channels}, data.dtype(), allocator);
      bilinear_resize<T>(src_buf, in_h, in_w, resized_buf, resized_h, resized_w, channels);

      size_t cx = std::max<size_t>(0, (resized_w - crop_w_) / 2);
      size_t cy = std::max<size_t>(0, (resized_h - crop_h_) / 2);

      for (size_t h = 0; h < crop_h_; ++h)
        for (size_t w = 0; w < crop_w_; ++w)
          for (size_t c = 0; c < channels; ++c)
            output.at<T>({b, h, w, c}) = resized_buf.at<T>({0, cy + h, cx + w, c});
    });

    data = std::move(output);
  }
};

}  // namespace synet
