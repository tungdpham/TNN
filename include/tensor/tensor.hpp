/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "layout_trait.hpp"
#include "utils/ops.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <malloc.h>
#include <windows.h>
#endif

#include "utils/parallel_for.hpp"

#ifdef __AVX2__
#include <immintrin.h>
#endif

enum ALIGNMENT_TYPE { MKL = 64, AVX2 = 32, DEFAULT = 16 };

/**
 * @brief A tensor class dedicated for ML and DL applications.
 * @tparam T Data type (e.g., float, double, int)
 * @tparam L Memory layout (NCHW, NHWC, NCDHW, NDHWC)
 * For now only NCHW is supported. A lot of changes are needed to support other
 * layouts.
 */
template <typename T = float, Layout L = NCHW> struct Tensor {
  static_assert(std::is_arithmetic<T>::value, "Tensor type must be arithmetic");
  static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
                "Tensor type must be floating point or integral");

private:
  LayoutTrait<L> layout_trait_;

  T *data_;

  static constexpr size_t dims_ = LayoutTrait<L>::dims;
  size_t (&shape_)[LayoutTrait<L>::dims] = layout_trait_.shape;
  size_t (&strides_)[LayoutTrait<L>::dims] = layout_trait_.strides;

  size_t data_size_;

  template <typename... Indices> inline size_t compute_index(Indices... indices) const {
    static_assert(sizeof...(indices) == dims_, "Incorrect number of dimensions");
    size_t index = 0;
    short count = 0;
    ((index += indices * strides_[count++]), ...);
    return index;
  }

  static T *allocate_aligned(size_t count) {
    if (count == 0)
      return nullptr;

    constexpr size_t alignment = ALIGNMENT_TYPE::MKL;
    size_t byte_size = count * sizeof(T);
    size_t aligned_size = ((byte_size + alignment - 1) / alignment) * alignment;

    void *ptr = nullptr;

#ifdef _WIN32
    ptr = _aligned_malloc(aligned_size, alignment);
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }
#elif defined(__linux__) || defined(__unix__)
    ptr = aligned_alloc(alignment, aligned_size);
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }
#else
    // Fall back to POSIX memalign
    if (posix_memalign(&ptr, alignment, aligned_size) != 0) {
      throw std::bad_alloc();
    }
#endif
    return static_cast<T *>(ptr);
  }

  static void deallocate_aligned(T *ptr) {
    if (ptr != nullptr) {
#ifdef _WIN32
      _aligned_free(ptr);
#else
      free(ptr);
#endif
    }
  }

public:
  // Constructors and Destructor
  Tensor() : data_(nullptr), data_size_(0) { data_ = allocate_aligned(0); }

  Tensor(size_t batch_size, size_t channels, size_t height, size_t width) : data_(nullptr) {
    static_assert(dims_ == 4, "This constructor is only for 4D tensors");
    layout_trait_.assign_shape(batch_size, channels, height, width);
    data_size_ = std::accumulate(shape_, shape_ + dims_, size_t(1), std::multiplies<size_t>());
    data_ = allocate_aligned(data_size_);
    utils::avx2_set_scalar(data_, T(0), data_size_);
  }

  Tensor(size_t batch_size, size_t channels, size_t height, size_t width, T *data)
      : data_(nullptr) {
    static_assert(dims_ == 4, "This constructor is only for 4D tensors");
    layout_trait_.assign_shape(batch_size, channels, height, width);
    data_size_ = std::accumulate(shape_, shape_ + dims_, size_t(1), std::multiplies<size_t>());
    data_ = allocate_aligned(data_size_);
    if (data != nullptr)
      utils::avx2_copy(data, data_, data_size_);
  }

  Tensor(std::vector<size_t> shape) : data_(nullptr) {
    assert(shape.size() == dims_ && "Shape vector size must match tensor dimensions");
    std::copy(shape.begin(), shape.end(), shape_);
    layout_trait_.compute_strides();
    data_size_ = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    data_ = allocate_aligned(data_size_);
    utils::avx2_set_scalar(data_, T(0), data_size_);
  }

  Tensor(std::vector<size_t> shape, const T *data) : data_(nullptr) {
    assert(shape.size() == dims_ && "Shape vector size must match dimensions");
    std::copy(shape.begin(), shape.end(), shape_);
    layout_trait_.compute_strides();
    data_size_ = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    data_ = allocate_aligned(data_size_);
    if (data != nullptr)
      utils::avx2_copy(data, data_, data_size_);
  }

  ~Tensor() { deallocate_aligned(data_); }

  Tensor(const Tensor &other) : data_size_(other.data_size_) {
    layout_trait_ = other.layout_trait_;
    if (data_size_ > 0) {
      data_ = allocate_aligned(data_size_);
      utils::avx2_copy(other.data_, data_, data_size_);
    }
  }

  Tensor(Tensor &&other) noexcept : data_(other.data_), data_size_(other.data_size_) {
    layout_trait_ = other.layout_trait_;
    other.data_ = nullptr;
    other.data_size_ = 0;
  }

  // Operators
  Tensor<T, L> &operator=(const Tensor<T, L> &other) = delete;

  Tensor<T, L> &operator=(Tensor<T, L> &&other) noexcept {
    if (this != &other) {
      deallocate_aligned(data_);

      layout_trait_ = other.layout_trait_;
      data_ = other.data_;
      data_size_ = other.data_size_;

      other.data_ = nullptr;
      other.data_size_ = 0;
    }
    return *this;
  }

  template <typename... Indices> T &operator()(Indices... indices) {
    static_assert(sizeof...(indices) == dims_, "Incorrect number of dimensions");
    return data_[compute_index(indices...)];
  }

  template <typename... Indices> const T &operator()(Indices... indices) const {
    static_assert(sizeof...(indices) == dims_, "Incorrect number of dimensions");
    return data_[compute_index(indices...)];
  }

  bool same_shape(const Tensor<T, L> &other) const {
    for (size_t i = 0; i < dims_; ++i) {
      if (shape_[i] != other.shape_[i]) {
        return false;
      }
    }
    return true;
  }

  Tensor<T, L> operator+(const Tensor<T, L> &other) const {
    if (!same_shape(other)) {
      throw std::invalid_argument("Tensor shapes must match for addition");
    }

    std::vector<size_t> shape_vec(shape_, shape_ + dims_);
    Tensor<T, L> result(shape_vec);

    utils::avx2_add(data_, other.data_, result.data_, data_size_);

    return result;
  }

  Tensor<T, L> operator-(const Tensor<T, L> &other) const {
    if (!same_shape(other)) {
      throw std::invalid_argument("Tensor shapes must match for subtraction");
    }

    std::vector<size_t> shape_vec(shape_, shape_ + dims_);
    Tensor<T, L> result(shape_vec);

    utils::avx2_sub(data_, other.data_, result.data_, data_size_);

    return result;
  }

  Tensor<T, L> operator*(const Tensor<T, L> &other) const {
    if (!same_shape(other)) {
      throw std::invalid_argument("Tensor shapes must match for element-wise multiplication");
    }

    std::vector<size_t> shape_vec(shape_, shape_ + dims_);
    Tensor<T, L> result(shape_vec);

    utils::avx2_mul(data_, other.data_, result.data_, data_size_);

    return result;
  }

  Tensor<T, L> operator/(const Tensor<T, L> &other) const {
    if (!same_shape(other)) {
      throw std::invalid_argument("Tensor shapes must match for element-wise division");
    }

    std::vector<size_t> shape_vec(shape_, shape_ + dims_);
    Tensor<T, L> result(shape_vec);

    utils::avx2_div(data_, other.data_, result.data_, data_size_);

    return result;
  }

  Tensor<T, L> operator*(T scalar) const {
    std::vector<size_t> shape_vec(shape_, shape_ + dims_);
    Tensor<T, L> result(shape_vec);

    utils::avx2_mul_scalar(data_, scalar, result.data_, data_size_);

    return result;
  }

  Tensor<T, L> operator/(T scalar) const {
    if (scalar == T(0)) {
      throw std::invalid_argument("Division by zero");
    }

    std::vector<size_t> shape_vec(shape_, shape_ + dims_);
    Tensor<T, L> result(shape_vec);

    utils::avx2_div_scalar(data_, scalar, result.data_, data_size_);

    return result;
  }

  Tensor<T, L> &operator+=(const Tensor<T, L> &other) {
    if (!same_shape(other)) {
      throw std::invalid_argument("Tensor shapes must match for addition");
    }

    utils::avx2_add(data_, other.data_, data_, data_size_);

    return *this;
  }

  Tensor<T, L> &operator-=(const Tensor<T, L> &other) {
    if (!same_shape(other)) {
      throw std::invalid_argument("Tensor shapes must match for subtraction");
    }

    utils::avx2_sub(data_, other.data_, data_, data_size_);

    return *this;
  }

  Tensor<T, L> &operator*=(const Tensor<T, L> &other) {
    if (!same_shape(other)) {
      throw std::invalid_argument("Tensor shapes must match for element-wise multiplication");
    }

    utils::avx2_mul(data_, other.data_, data_, data_size_);

    return *this;
  }

  Tensor<T, L> &operator*=(T scalar) {

    utils::avx2_mul_scalar(data_, scalar, data_, data_size_);

    return *this;
  }

  Tensor<T, L> &operator/=(T scalar) {
    if (scalar == T(0)) {
      throw std::invalid_argument("Division by zero");
    }

    utils::avx2_div_scalar(data_, scalar, data_, data_size_);

    return *this;
  }

  std::vector<size_t> shape() const { return std::vector<size_t>(shape_, shape_ + dims_); }

  std::string shape_str() const {
    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0; i < dims_; ++i) {
      oss << shape_[i];
      if (i < dims_ - 1) {
        oss << ", ";
      }
    }
    oss << "}";
    return oss.str();
  }

  const size_t batch_size() const { return layout_trait_.batch_size(); }

  const size_t channels() const { return layout_trait_.channels(); }

  const size_t height() const { return layout_trait_.height(); }

  const size_t width() const { return layout_trait_.width(); }

  const size_t depth() const {
    if constexpr (dims_ == 5) {
      return layout_trait_.depth();
    } else {
      return 1;
    }
  }

  const size_t dimension(const size_t index) const { return shape_[index]; }

  const size_t stride(const size_t index) const { return strides_[index]; }

  const size_t size() const { return data_size_; }

  T *data() { return data_; }

  const T *data() const { return data_; }

  bool is_aligned(size_t alignment = 32) const {
    return (reinterpret_cast<uintptr_t>(data_) % alignment) == 0;
  }

  Tensor<T, L> clone() const {
    return Tensor<T, L>(std::vector<size_t>(shape_, shape_ + dims_), data_);
  }

  void fill(T value) { std::fill(data_, data_ + data_size_, value); }

  void fill_random_uniform(T range) {
    std::mt19937 gen(std::random_device{}());
    if constexpr (std::is_floating_point<T>::value) {
      std::uniform_real_distribution<T> dis(-range, range);
      for (size_t i = 0; i < data_size_; ++i) {
        data_[i] = dis(gen);
      }
    } else {
      auto int_range = static_cast<
          typename std::conditional<std::is_signed<T>::value, std::uint64_t, std::uint64_t>::type>(
          range);
      std::uniform_int_distribution<decltype(int_range)> dis(-int_range, int_range);
      for (size_t i = 0; i < data_size_; ++i) {
        data_[i] = static_cast<T>(dis(gen));
      }
    }
  }

  void fill_random_normal(T mean, T stddev) {
    static_assert(std::is_floating_point<T>::value,
                  "Normal distribution requires floating point type");
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<T> dis(mean, stddev);
    for (size_t i = 0; i < data_size_; ++i) {
      data_[i] = dis(gen);
    }
  }

  Tensor<T, L> reshape(const std::vector<size_t> &new_shape) const {
    size_t new_size =
        std::accumulate(new_shape.begin(), new_shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_size != size()) {
      throw std::invalid_argument("New shape must have same total size");
    }
    return Tensor<T, L>(new_shape, data_);
  }

  Tensor<T, L> pad(size_t pad_h, size_t pad_w, T value = T(0)) const {
    assert(dims_ == 4 && "Padding only supported for 4D tensors");
    if (pad_h == 0 && pad_w == 0) {
      return *this;
    }

    const size_t batch_size_ = batch_size();
    const size_t channels_ = channels();
    const size_t height_ = height();
    const size_t width_ = width();

    Tensor<T, L> result(batch_size_, channels_, height_ + 2 * pad_h, width_ + 2 * pad_w, nullptr);

    const T *input_data = this->data();
    T *result_data = result.data();

    utils::parallel_for_2d(batch_size_, channels_, [&](size_t n, size_t c) {
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
        std::copy(
            &input_data[((n * channels_ + c) * height_ + h) * width_],
            &input_data[((n * channels_ + c) * height_ + h) * width_] + width_,
            &result_data[((n * channels_ + c) * padded_height + new_h) * padded_width + pad_w]);

        // set values on left and right
        std::fill(
            &result_data[((n * channels_ + c) * padded_height + new_h) * padded_width],
            &result_data[((n * channels_ + c) * padded_height + new_h) * padded_width + pad_w],
            value);
        std::fill(&result_data[((n * channels_ + c) * padded_height + new_h) * padded_width +
                               pad_w + width_],
                  &result_data[((n * channels_ + c) * padded_height + new_h) * padded_width +
                               padded_width],
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

  Tensor<T, L> unpad(size_t pad_h, size_t pad_w) const {
    assert(dims_ == 4 && "Unpadding only supported for 4D tensors");
    if (pad_h == 0 && pad_w == 0) {
      return this->clone();
    }

    const size_t batch_size_ = batch_size();
    const size_t channels_ = channels();
    const size_t height_ = height();
    const size_t width_ = width();

    if (height_ <= 2 * pad_h || width_ <= 2 * pad_w) {
      throw std::invalid_argument("Padding size too large for unpadding");
    }

    Tensor<T, L> result(batch_size_, channels_, height_ - 2 * pad_h, width_ - 2 * pad_w);

    const T *input_data = this->data();
    T *result_data = result.data();

    utils::parallel_for_2d(batch_size_, channels_, [&](size_t n, size_t c) {
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

  Tensor<T, L> crop(const size_t start_h, const size_t start_w, const size_t end_h,
                    const size_t end_w) const {
    if constexpr (dims_ != 4) {
      throw std::runtime_error("2D cropping only supported for 4D tensors");
    }

    if (end_h >= height() || end_w >= width() || start_h > end_h || start_w > end_w) {
      throw std::invalid_argument("Invalid crop dimensions");
    }

    const size_t new_height = end_h - start_h + 1;
    const size_t new_width = end_w - start_w + 1;

    const size_t batch_size_ = batch_size();
    const size_t channels_ = channels();
    const size_t height_ = height();
    const size_t width_ = width();
    Tensor<T, L> result(batch_size_, channels_, new_height, new_width);

    T *result_data = result.data();
    for (size_t n = 0; n < batch_size_; ++n) {
      for (size_t c = 0; c < channels_; ++c) {
        for (size_t h = 0; h < new_height; ++h) {
          std::copy(&data_[((n * channels_ + c) * height_ + (h + start_h)) * width_ + start_w],
                    &data_[((n * channels_ + c) * height_ + (h + start_h)) * width_ + start_w] +
                        new_width,
                    &result_data[((n * channels_ + c) * new_height + h) * new_width]);
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
  Tensor<T, L> slice_batch(size_t start_batch, size_t end_batch) const {
    if (end_batch > batch_size() || start_batch > end_batch) {
      throw std::invalid_argument("Invalid batch slice range");
    }

    size_t new_batch_size = end_batch - start_batch;
    std::vector<size_t> new_shape(shape_, shape_ + dims_);
    new_shape[0] = new_batch_size;
    Tensor<T, L> result(new_shape);

    T *result_data = result.data();
    std::copy(&data_[start_batch * strides_[0]], &data_[end_batch * strides_[0]], result_data);

    return result;
  }

  /**
   * @brief Copy a specific batch from another tensor to this tensor.
   */
  void copy_batch(Tensor<T, L> &other, size_t src_batch_idx, size_t dest_batch_idx) const {
    if (dest_batch_idx >= batch_size() || src_batch_idx >= other.batch_size()) {
      throw std::invalid_argument("Invalid batch index for copy");
    }

    std::copy(&other.data_[src_batch_idx * other.strides_[0]],
              &other.data_[(src_batch_idx + 1) * other.strides_[0]],
              &data_[dest_batch_idx * strides_[0]]);
  }

  /**
   * @brief Slice the tensor along the channel dimension.
   * @param start_ch Starting channel index (inclusive)
   * @param end_ch Ending channel index (inclusive)
   * @return A new tensor containing the sliced channels
   * Note: Only supports 4D tensors (NCHW layout)
   */
  Tensor<T, L> slice_channels(size_t start_ch, size_t end_ch) const {
    if (end_ch >= channels() || start_ch > end_ch) {
      throw std::invalid_argument("Invalid channel slice range");
    }

    size_t new_channels = end_ch - start_ch + 1;

    if constexpr (dims_ == 4) {
      Tensor<T, L> result(batch_size(), new_channels, height(), width());

      for (size_t n = 0; n < batch_size(); ++n) {
        for (size_t c = 0; c < new_channels; ++c) {
          for (size_t h = 0; h < height(); ++h) {
            for (size_t w = 0; w < width(); ++w) {
              result(n, c, h, w) = (*this)(n, start_ch + c, h, w);
            }
          }
        }
      }
      return result;
    } else {
      throw std::runtime_error("Unsupported tensor dimensionality for channel slicing");
    }
  }

  T mean() const {
    T sum = T(0);
    for (size_t i = 0; i < data_size_; ++i) {
      sum += data_[i];
    }
    return sum / static_cast<T>(data_size_);
  }

  T variance() const {
    T m = mean();
    T sum_sq_diff = T(0);
    for (size_t i = 0; i < data_size_; ++i) {
      T diff = data_[i] - m;
      sum_sq_diff += diff * diff;
    }
    return sum_sq_diff / static_cast<T>(data_size_);
  }

  std::vector<T> channel_means() const {
    std::vector<T> means(channels(), T(0));

    if constexpr (dims_ == 4) {
      size_t channel_size = batch_size() * height() * width();

      for (size_t c = 0; c < channels(); ++c) {
        T sum = T(0);
        for (size_t n = 0; n < batch_size(); ++n) {
          for (size_t h = 0; h < height(); ++h) {
            for (size_t w = 0; w < width(); ++w) {
              sum += (*this)(n, c, h, w);
            }
          }
        }
        means[c] = sum / static_cast<T>(channel_size);
      }
    } else {
      throw std::runtime_error("Unsupported tensor dimensionality for channel statistics");
    }

    return means;
  }

  void apply_softmax() {
    // Only supports 4D tensors (NCHW layout)
    if (dims_ != 4) {
      throw std::invalid_argument("apply_softmax() only supports 4D tensors");
    }

    const size_t batch_size = shape_[0];
    const size_t num_classes = shape_[1];
    const size_t height = shape_[2];
    const size_t width = shape_[3];

    // Apply softmax across channels at each spatial location
    for (size_t batch = 0; batch < batch_size; ++batch) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          // Find max value for numerical stability
          T max_val = (*this)(batch, 0, h, w);
          for (size_t c = 1; c < num_classes; ++c) {
            max_val = std::max(max_val, (*this)(batch, c, h, w));
          }

          // Apply exp and sum
          T sum = T(0);
          for (size_t c = 0; c < num_classes; ++c) {
            const T exp_val = std::exp((*this)(batch, c, h, w) - max_val);
            (*this)(batch, c, h, w) = exp_val;
            sum += exp_val;
          }

          // Normalize with numerical stability protection
          const T inv_sum = T(1) / std::max(sum, static_cast<T>(1e-8));
          for (size_t c = 0; c < num_classes; ++c) {
            (*this)(batch, c, h, w) *= inv_sum;
          }
        }
      }
    }
  }

  std::vector<Tensor<T>> split(size_t num_splits) const {
    if (num_splits == 0 || num_splits > batch_size()) {
      throw std::invalid_argument("Invalid number of splits");
    }

    std::vector<Tensor<T>> splits;
    size_t split_size = batch_size() / num_splits;

    for (size_t i = 0; i < num_splits; ++i) {
      size_t start = i * split_size;
      size_t end = (i == num_splits - 1) ? batch_size() : start + split_size;

      if constexpr (dims_ == 4) {
        splits.emplace_back(slice_batch(start, end));
      } else {
        throw std::runtime_error("Unsupported tensor dimensionality for splitting");
      }
    }
    return splits;
  }

  template <Layout New_Layout> Tensor<T, New_Layout> as_layout() const {
    Tensor<T, New_Layout> result(this->shape());

    if constexpr (dims_ == 4) {
      for (size_t n = 0; n < shape_[0]; ++n) {
        for (size_t c = 0; c < shape_[1]; ++c) {
          for (size_t h = 0; h < shape_[2]; ++h) {
            for (size_t w = 0; w < shape_[3]; ++w) {
              result(n, c, h, w) = (*this)(n, c, h, w);
            }
          }
        }
      }
    } else {
      throw std::runtime_error("Conversion for this dimensionality not implemented.");
    }
    return result;
  }

  void print_data() const {
    std::cout << "Tensor data: ";
    if constexpr (dims_ == 4) {
      for (size_t n = 0; n < batch_size(); ++n) {
        for (size_t c = 0; c < channels(); ++c) {
          for (size_t h = 0; h < height(); ++h) {
            for (size_t w = 0; w < width(); ++w) {
              std::cout << operator()(n, c, h, w) << " ";
            }
          }
        }
        std::cout << std::endl;
      }
    }
    std::cout << std::endl;
  }

  void save(std::ofstream &out) const {
    if (!out.is_open()) {
      throw std::runtime_error("File is not open for writing");
    }

    out.write(reinterpret_cast<const char *>(shape_), dims_ * sizeof(size_t));

    out.write(reinterpret_cast<const char *>(data_), data_size_ * sizeof(T));
  }

  static Tensor<T, L> load(std::ifstream &in) {
    if (!in.is_open()) {
      throw std::runtime_error("File is not open for reading");
    }
    std::vector<size_t> shape(dims_);
    in.read(reinterpret_cast<char *>(shape.data()), dims_ * sizeof(size_t));
    if (in.gcount() != dims_ * sizeof(size_t)) {
      throw std::runtime_error("Failed to read tensor shape from file");
    }

    Tensor<T, L> tensor(shape);
    in.read(reinterpret_cast<char *>(tensor.data()), tensor.size() * sizeof(T));
    if (in.gcount() != static_cast<std::streamsize>(tensor.size() * sizeof(T))) {
      throw std::runtime_error("Failed to read tensor data from file");
    }
    return tensor;
  }
};

#include "tensor/tensor_extended.hpp"