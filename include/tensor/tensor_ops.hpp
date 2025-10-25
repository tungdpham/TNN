#pragma once
#include "tensor.hpp"

template <typename T, Layout L>
Tensor<T, L> pad(const Tensor<T, L> &input, size_t pad_h, size_t pad_w, T value = T(0)) {
  throw std::runtime_error("Unsupported tensor layout for padding");
}

template <typename T>
Tensor<T, NCHW> pad(const Tensor<T, NCHW> &input, size_t pad_h, size_t pad_w, T value = T(0)) {
  const size_t batch_size_ = input.batch_size();
  const size_t channels_ = input.channels();
  const size_t height_ = input.height();
  const size_t width_ = input.width();

  Tensor<T, NCHW> result(batch_size_, channels_, height_ + 2 * pad_h, width_ + 2 * pad_w, nullptr);

  const T *input_data = input.data();
  T *result_data = result.data();

  tthreads::parallel_for_2d(batch_size_, channels_, [&](size_t n, size_t c) {
    const size_t padded_height = height_ + 2 * pad_h;
    const size_t padded_width = width_ + 2 * pad_w;
    // fill top padding rows
    for (size_t h = 0; h < pad_h; ++h) {
      std::fill(&result_data[((n * channels_ + c) * padded_height + h) * padded_width],
                &result_data[((n * channels_ + c) * padded_height + h) * padded_width] +
                    padded_width,
                value);
    }

    // Copy middle rows with left and right padding
    for (size_t h = 0; h < height_; ++h) {
      const size_t new_h = h + pad_h;
      // copy the row over
      std::copy(&input_data[((n * channels_ + c) * height_ + h) * width_],
                &input_data[((n * channels_ + c) * height_ + h) * width_] + width_,
                &result_data[((n * channels_ + c) * padded_height + new_h) * padded_width + pad_w]);

      // set values on left and right
      std::fill(&result_data[((n * channels_ + c) * padded_height + new_h) * padded_width],
                &result_data[((n * channels_ + c) * padded_height + new_h) * padded_width + pad_w],
                value);

      // right side
      std::fill(
          &result_data[((n * channels_ + c) * padded_height + new_h) * padded_width + pad_w +
                       width_],
          &result_data[((n * channels_ + c) * padded_height + new_h) * padded_width + padded_width],
          value);
    }

    // fill bottom padding rows
    for (size_t h = height_ + pad_h; h < padded_height; ++h) {
      std::fill(&result_data[((n * channels_ + c) * padded_height + h) * padded_width],
                &result_data[((n * channels_ + c) * padded_height + h) * padded_width] +
                    padded_width,
                value);
    }
  });

  return result;
}

template <typename T, Layout L>
Tensor<T, NCHW> unpad(const Tensor<T, L> &input, size_t pad_h, size_t pad_w) {
  throw std::runtime_error("Unsupported tensor layout for unpadding");
}

template <typename T>
Tensor<T, NCHW> unpad(const Tensor<T, NCHW> &input, size_t pad_h, size_t pad_w) {
  const size_t batch_size_ = input.batch_size();
  const size_t channels_ = input.channels();
  const size_t height_ = input.height();
  const size_t width_ = input.width();

  if (height_ <= 2 * pad_h || width_ <= 2 * pad_w) {
    throw std::invalid_argument("Padding size too large for unpadding");
  }

  Tensor<T, NCHW> result(batch_size_, channels_, height_ - 2 * pad_h, width_ - 2 * pad_w, nullptr);

  const T *input_data = input.data();
  T *result_data = result.data();

  tthreads::parallel_for_2d(batch_size_, channels_, [&](size_t n, size_t c) {
    for (size_t h = 0; h < height_ - 2 * pad_h; ++h) {
      const size_t src_h = h + pad_h;
      std::copy(
          &input_data[((n * channels_ + c) * height_ + src_h) * width_ + pad_w],
          &input_data[((n * channels_ + c) * height_ + src_h) * width_ + pad_w] +
              (width_ - 2 * pad_w),
          &result_data[((n * channels_ + c) * (height_ - 2 * pad_h) + h) * (width_ - 2 * pad_w)]);
    }
  });

  return result;
}

template <typename T, Layout L>
Tensor<T, L> crop(const Tensor<T, L> &input, const size_t start_h, const size_t start_w,
                  const size_t end_h, const size_t end_w) {
  throw std::runtime_error("Unsupported tensor layout for cropping");
}

template <typename T>
Tensor<T, NCHW> crop(const Tensor<T, NCHW> &input, const size_t start_h, const size_t start_w,
                     const size_t end_h, const size_t end_w) {
  if (end_h >= input.height() || end_w >= input.width() || start_h > end_h || start_w > end_w) {
    throw std::invalid_argument("Invalid crop dimensions");
  }
  const size_t new_height = end_h - start_h + 1;
  const size_t new_width = end_w - start_w + 1;

  const size_t batch_size = input.batch_size();
  const size_t channels = input.channels();
  const size_t height_ = input.height();
  const size_t width_ = input.width();
  Tensor<T, NCHW> result(batch_size, channels, new_height, new_width, nullptr);

  T *input_data = input.data();
  T *result_data = result.data();
  for (size_t n = 0; n < batch_size; ++n) {
    for (size_t c = 0; c < channels; ++c) {
      for (size_t h = 0; h < new_height; ++h) {
        std::copy(&input_data[((n * channels + c) * height_ + (h + start_h)) * width_ + start_w],
                  &input_data[((n * channels + c) * height_ + (h + start_h)) * width_ + start_w] +
                      new_width,
                  &result_data[((n * channels + c) * new_height + h) * new_width]);
      }
    }
  }
  return result;
}

/**
 * @brief Slice the tensor along the batch dimension.
 * @param start_batch Starting batch index (inclusive)
 * @param end_batch Ending batch index (exclusive)
 * @return A new tensor containing the sliced batches
 */
template <typename T, Layout L>
Tensor<T, L> slice_batch(const Tensor<T, L> &input, size_t start_batch, size_t end_batch) {
  if (end_batch > input.batch_size() || start_batch > end_batch) {
    throw std::invalid_argument("Invalid batch slice range");
  }

  size_t new_batch_size = end_batch - start_batch;
  std::vector<size_t> new_shape(input.shape());
  new_shape[0] = new_batch_size;
  Tensor<T, L> result(new_shape);

  const T *input_data = input.data();
  const std::vector<size_t> strides = input.strides();
  T *result_data = result.data();

  std::copy(&input_data[start_batch * strides[0]], &input_data[end_batch * strides[0]],
            result_data);
  return result;
}

/*
 * @brief Slice the tensor along the channel dimension.
 */
template <typename T, Layout L>
Tensor<T, L> slice_channels(const Tensor<T, L> &input, size_t start_ch, size_t end_ch) {
  throw std::runtime_error("Unsupported tensor layout for channel slicing");
}

/**
 * @brief Slice the tensor along the channel dimension.
 * @param start_ch Starting channel index (inclusive)
 * @param end_ch Ending channel index (inclusive)
 * @return A new tensor containing the sliced channels
 */
template <typename T>
Tensor<T, NCHW> slice_channels(const Tensor<T, NCHW> &input, size_t start_ch, size_t end_ch) {
  if (end_ch >= input.channels() || start_ch > end_ch) {
    throw std::invalid_argument("Invalid channel slice range");
  }

  size_t new_channels = end_ch - start_ch + 1;

  Tensor<T, NCHW> result(input.batch_size(), new_channels, input.height(), input.width());

  for (size_t n = 0; n < input.batch_size(); ++n) {
    for (size_t c = 0; c < new_channels; ++c) {
      for (size_t h = 0; h < input.height(); ++h) {
        for (size_t w = 0; w < input.width(); ++w) {
          result(n, c, h, w) = input(n, start_ch + c, h, w);
        }
      }
    }
  }
  return result;
}

template <typename T, Layout L>
std::vector<Tensor<T, L>> split(const Tensor<T, L> &input, size_t num_splits) {
  if (num_splits == 0 || num_splits > input.batch_size()) {
    throw std::invalid_argument("Invalid number of splits");
  }

  std::vector<Tensor<T, L>> splits;
  size_t split_size = input.batch_size() / num_splits;

  for (size_t i = 0; i < num_splits; ++i) {
    size_t start = i * split_size;
    size_t end = (i == num_splits - 1) ? input.batch_size() : start + split_size;

    splits.emplace_back(slice_batch(input, start, end));
  }
  return splits;
}

template <typename T>
std::vector<Tensor<T, NCHW>> split(const Tensor<T, NCHW> &input, size_t num_splits) {
  if (num_splits == 0 || num_splits > input.batch_size()) {
    throw std::invalid_argument("Invalid number of splits");
  }

  std::vector<Tensor<T, NCHW>> splits;
  size_t split_size = input.batch_size() / num_splits;

  for (size_t i = 0; i < num_splits; ++i) {
    size_t start = i * split_size;
    size_t end = (i == num_splits - 1) ? input.batch_size() : start + split_size;

    splits.emplace_back(slice_batch(input, start, end));
  }
  return splits;
}

template <typename T, Layout L> void apply_softmax(Tensor<T, L> &input) {
  throw std::runtime_error("Unsupported tensor layout for softmax");
}

template <typename T> void apply_softmax(Tensor<T, NCHW> &input) {
  auto shape = input.shape();
  const size_t batch_size = shape[0];
  const size_t num_classes = shape[1];
  const size_t height = shape[2];
  const size_t width = shape[3];

  // Apply softmax across channels at each spatial location
  for (size_t batch = 0; batch < batch_size; ++batch) {
    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        // Find max value for numerical stability
        T max_val = input(batch, 0, h, w);
        for (size_t c = 1; c < num_classes; ++c) {
          max_val = std::max(max_val, input(batch, c, h, w));
        }

        // Apply exp and sum
        T sum = T(0);
        for (size_t c = 0; c < num_classes; ++c) {
          const T exp_val = std::exp(input(batch, c, h, w) - max_val);
          input(batch, c, h, w) = exp_val;
          sum += exp_val;
        }

        // Normalize with numerical stability protection
        const T inv_sum = T(1) / std::max(sum, static_cast<T>(1e-8));
        for (size_t c = 0; c < num_classes; ++c) {
          input(batch, c, h, w) *= inv_sum;
        }
      }
    }
  }
}
