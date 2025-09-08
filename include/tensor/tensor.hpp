/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "matrix/matrix.hpp"
#include "tensor_view.hpp"
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

enum ALIGNMENT_TYPE { AVX2 = 32, DEFAULT = 16 };

template <typename T = float, Layout L = NCHW> struct Tensor {
  static_assert(std::is_arithmetic<T>::value, "Tensor type must be arithmetic");
  static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
                "Tensor type must be floating point or integral");

private:
  using View = TensorView<T, L>;

  static constexpr size_t dims_ = View::dims;
  size_t shape_[dims_];
  size_t strides_[dims_];
  T *data_;

  size_t data_size_;

  inline void compute_strides() { View::compute_strides(strides_, shape_); }

  template <typename... Indices>
  inline size_t compute_index(Indices... indices) const {
    static_assert(sizeof...(indices) == dims_,
                  "Incorrect number of dimensions");
    size_t index = 0;
    short count = 0;
    ((index += indices * strides_[count++]), ...);
    return index;
  }

  static T *allocate_aligned(size_t count) {
    if (count == 0)
      return nullptr;

    constexpr size_t alignment = ALIGNMENT_TYPE::AVX2;
    size_t byte_size = count * sizeof(T);
    size_t aligned_size = ((byte_size + alignment - 1) / alignment) * alignment;

    void *ptr = nullptr;

#ifdef _WIN32
    ptr = _aligned_malloc(aligned_size, alignment);
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }
#else
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
  Tensor() : data_(nullptr), data_size_(0) {}

  Tensor(size_t batch_size, size_t channels, size_t height, size_t width) : data_(nullptr) {
    if constexpr (dims_ != 4) {
      throw std::invalid_argument(
          "This constructor is only for 4D tensors (NCHW or NHWC)");
    }
    shape_[0] = batch_size;
    shape_[1] = channels;
    shape_[2] = height;
    shape_[3] = width;
    compute_strides();
    data_size_ = std::accumulate(shape_, shape_ + dims_, size_t(1),
                                 std::multiplies<size_t>());
    data_ = allocate_aligned(data_size_);
    std::fill(data_, data_ + data_size_, T(0));
  }

  Tensor(size_t batch_size, size_t channels, size_t height, size_t width, T* data){
    // Use with precautions
    if constexpr (dims_ != 4) {
      throw std::invalid_argument(
          "This constructor is only for 4D tensors (NCHW or NHWC)");
    }
    shape_[0] = batch_size;
    shape_[1] = channels;
    shape_[2] = height;
    shape_[3] = width;
    compute_strides();
    data_size_ = std::accumulate(shape_, shape_ + dims_, size_t(1),
                                 std::multiplies<size_t>());
    data_ = allocate_aligned(data_size_);
    if(data != nullptr)
      data_ = data;
  }

  Tensor(std::vector<size_t> shape) : data_(nullptr) {
    if (shape.size() != dims_) {
      throw std::invalid_argument(
          "Shape vector size must match tensor dimensions");
    }
    std::copy(shape.begin(), shape.end(), shape_);
    compute_strides();
    data_size_ = std::accumulate(shape_, shape_ + dims_, size_t(1),
                                 std::multiplies<size_t>());
    data_ = allocate_aligned(data_size_);
    std::fill(data_, data_ + data_size_, T(0));
  }

  Tensor(std::vector<size_t> shape, const T *data) : data_(nullptr) {
    if (shape.size() != dims_) {
      throw std::invalid_argument(
          "Shape vector size must match tensor dimensions");
    }
    std::copy(shape.begin(), shape.end(), shape_);
    compute_strides();
    data_size_ = std::accumulate(shape_, shape_ + dims_, size_t(1),
                                 std::multiplies<size_t>());
    data_ = allocate_aligned(data_size_);
    if(data != nullptr)
      std::copy(data, data + data_size_, data_);
  }

  ~Tensor() { deallocate_aligned(data_); }

  Tensor(const Tensor &other) : data_size_(other.data_size_) {
    std::copy(other.shape_, other.shape_ + dims_, shape_);
    compute_strides();
    if (data_size_ > 0) {
      data_ = allocate_aligned(data_size_);
      std::copy(other.data_, other.data_ + data_size_, data_);
    }
  }

  Tensor(Tensor &&other) noexcept
      : data_(other.data_), data_size_(other.data_size_) {
    std::copy(other.shape_, other.shape_ + dims_, shape_);
    compute_strides();
    other.data_ = nullptr;
    other.data_size_ = 0;
  }

  Tensor<T, L> &operator=(const Tensor<T, L> &other) = delete;

  Tensor &operator=(Tensor &&other) noexcept {
    if (this != &other) {

      deallocate_aligned(data_);

      std::copy(other.shape_, other.shape_ + dims_, shape_);
      compute_strides();
      data_ = other.data_;
      data_size_ = other.data_size_;

      other.data_ = nullptr;
      other.data_size_ = 0;
    }
    return *this;
  }

  template <typename... Indices> T &operator()(Indices... indices) {
    static_assert(sizeof...(indices) == dims_,
                  "Incorrect number of dimensions");
    return data_[compute_index(indices...)];
  }

  template <typename... Indices> const T &operator()(Indices... indices) const {
    static_assert(sizeof...(indices) == dims_,
                  "Incorrect number of dimensions");
    return data_[compute_index(indices...)];
  }

  std::vector<size_t> shape() const {
    return std::vector<size_t>(shape_, shape_ + dims_);
  }

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

  const size_t *strides_ptr() const { return strides_; }

  size_t batch_size() const { return shape_[0]; }

  size_t channels() const { return shape_[1]; }

  size_t height() const { return shape_[dims_ - 2]; }

  size_t width() const { return shape_[dims_ - 1]; }

  size_t depth() const {
    if constexpr (dims_ == 5) {
      return shape_[2];
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
    std::cout << "Data pointer address: " << reinterpret_cast<uintptr_t>(data_)
              << std::endl;
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

      auto int_range = static_cast<typename std::conditional<
          std::is_signed<T>::value, std::int64_t, std::uint64_t>::type>(range);
      std::uniform_int_distribution<decltype(int_range)> dis(-int_range,
                                                             int_range);
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

    bool same_shape = (new_shape.size() == dims_);
    if (same_shape) {
      for (size_t i = 0; i < dims_; ++i) {
        if (new_shape[i] != shape_[i]) {
          same_shape = false;
          break;
        }
      }
    }

    if (same_shape) {
      return *this;
    }

    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(),
                                      size_t(1), std::multiplies<size_t>());
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

    Tensor<T, L> result(batch_size_, channels_, height_ + 2 * pad_h,
                        width_ + 2 * pad_w);

    if (value != T(0))
      result.fill(value);

    const T *input_data = this->data();
    T *result_data = result.data();

    utils::parallel_for_2d(batch_size_, channels_, [&](size_t n, size_t c) {
      for (size_t h = 0; h < height_; ++h) {
        const size_t new_h = h + pad_h;
        std::copy(
            &input_data[((n * channels_ + c) * height_ + h) * width_],
            &input_data[((n * channels_ + c) * height_ + h) * width_] + width_,
            &result_data[((n * channels_ + c) * (height_ + 2 * pad_h) + new_h) *
                             (width_ + 2 * pad_w) +
                         pad_w]);
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

    Tensor<T, L> result(batch_size_, channels_, height_ - 2 * pad_h,
                        width_ - 2 * pad_w);

    const T *input_data = this->data();
    T *result_data = result.data();

    utils::parallel_for_2d(batch_size_, channels_, [&](size_t n, size_t c) {
      for (size_t h = 0; h < height_ - 2 * pad_h; ++h) {
        const size_t src_h = h + pad_h;
        std::copy(
            &input_data[((n * channels_ + c) * height_ + src_h) * width_ +
                        pad_w],
            &input_data[((n * channels_ + c) * height_ + src_h) * width_ +
                        pad_w] +
                (width_ - 2 * pad_w),
            &result_data[((n * channels_ + c) * (height_ - 2 * pad_h) + h) *
                         (width_ - 2 * pad_w)]);
      }
    });

    return result;
  }

  Tensor<T, L> crop(const size_t start_h, const size_t start_w,
                    const size_t end_h, const size_t end_w) const {
    if constexpr (dims_ != 4) {
      throw std::runtime_error("2D cropping only supported for 4D tensors");
    }

    if (end_h >= height() || end_w >= width() || start_h > end_h ||
        start_w > end_w) {
      throw std::invalid_argument("Invalid crop dimensions");
    }

    const size_t new_height = end_h - start_h + 1;
    const size_t new_width = end_w - start_w + 1;

    const size_t batch_size_ = batch_size();
    const size_t channels_ = channels();
    Tensor<T, L> result(batch_size_, channels_, new_height, new_width);

    for (size_t n = 0; n < batch_size_; ++n) {
      for (size_t c = 0; c < channels_; ++c) {
        for (size_t h = 0; h < new_height; ++h) {
          for (size_t w = 0; w < new_width; ++w) {
            result(n, c, h, w) = (*this)(n, c, start_h + h, start_w + w);
          }
        }
      }
    }

    return result;
  }

  Tensor<T, L> slice_batch(size_t start_batch, size_t end_batch) const {
    if (end_batch >= batch_size() || start_batch > end_batch) {
      throw std::invalid_argument("Invalid batch slice range");
    }

    size_t new_batch_size = end_batch - start_batch + 1;

    if constexpr (dims_ == 4) {
      Tensor<T, L> result(new_batch_size, channels(), height(), width());
      T *result_data = result.data();
      size_t output_size = channels() * height() * width();
      for (size_t n = 0; n < new_batch_size; ++n) {
        for (size_t idx = 0; idx < output_size; ++idx) {
          size_t batch_idx = start_batch + n;
          result_data[n * output_size + idx] =
              this->data_[batch_idx * strides_[0] + idx];
        }
      }
      return result;
    } else {
      throw std::runtime_error(
          "Unsupported tensor dimensionality for batch slicing");
    }
  }

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
      throw std::runtime_error(
          "Unsupported tensor dimensionality for channel slicing");
    }
  }

  Tensor<T, L> operator+(const Tensor<T, L> &other) const {

    for (size_t i = 0; i < dims_; ++i) {
      if (shape_[i] != other.shape_[i]) {
        std::cerr << "Shape mismatch: " << shape_[i] << " vs "
                  << other.shape_[i] << std::endl;
        throw std::invalid_argument("Tensor shapes must match for addition");
      }
    }

    std::vector<size_t> shape_vec(shape_, shape_ + dims_);
    Tensor<T, L> result(shape_vec);

    for (size_t idx = 0; idx < data_size_; ++idx)
      result.data_[idx] = data_[idx] + other.data_[idx];

    return result;
  }

  Tensor<T, L> operator-(const Tensor<T, L> &other) const {

    for (size_t i = 0; i < dims_; ++i) {
      if (shape_[i] != other.shape_[i]) {
        throw std::invalid_argument("Tensor shapes must match for subtraction");
      }
    }

    std::vector<size_t> shape_vec(shape_, shape_ + dims_);
    Tensor<T, L> result(shape_vec);

    for (size_t idx = 0; idx < data_size_; ++idx) {
      result.data_[idx] = data_[idx] - other.data_[idx];
    }

    return result;
  }

  Tensor<T, L> operator*(T scalar) const {
    std::vector<size_t> shape_vec(shape_, shape_ + dims_);
    Tensor<T, L> result(shape_vec);
    for (size_t i = 0; i < data_size_; ++i) {
      result.data_[i] = data_[i] * scalar;
    }
    return result;
  }

  Tensor<T, L> operator/(T scalar) const {
    if (scalar == T(0)) {
      throw std::invalid_argument("Division by zero");
    }

    std::vector<size_t> shape_vec(shape_, shape_ + dims_);
    Tensor<T, L> result(shape_vec);
    for (size_t i = 0; i < data_size_; ++i) {
      result.data_[i] = data_[i] / scalar;
    }
    return result;
  }

  Tensor<T, L> &operator+=(const Tensor<T, L> &other) {

    for (size_t i = 0; i < dims_; ++i) {
      if (shape_[i] != other.shape_[i]) {
        std::cerr << "Shape mismatch: " << shape_[i] << " vs "
                  << other.shape_[i] << std::endl;
        throw std::invalid_argument("Tensor shapes must match for addition");
      }
    }

    for (size_t idx = 0; idx < data_size_; ++idx) {
      data_[idx] += other.data_[idx];
    }

    return *this;
  }

  Tensor<T, L> &operator-=(const Tensor<T, L> &other) {

    for (size_t i = 0; i < dims_; ++i) {
      if (shape_[i] != other.shape_[i]) {
        throw std::invalid_argument("Tensor shapes must match for subtraction");
      }
    }

    for (size_t idx = 0; idx < data_size_; ++idx) {
      data_[idx] -= other.data_[idx];
    }

    return *this;
  }

  Tensor<T, L> &operator*=(const Tensor<T, L> &other) {

    for (size_t i = 0; i < dims_; ++i) {
      if (shape_[i] != other.shape_[i]) {
        throw std::invalid_argument(
            "Tensor shapes must match for element-wise multiplication");
      }
    }

    for (size_t idx = 0; idx < data_size_; ++idx) {
      data_[idx] *= other.data_[idx];
    }

    return *this;
  }

  Tensor<T, L> &operator*=(T scalar) {
    for (size_t i = 0; i < data_size_; ++i) {
      data_[i] *= scalar;
    }
    return *this;
  }

  Tensor<T, L> &operator/=(T scalar) {
    if (scalar == T(0)) {
      throw std::invalid_argument("Division by zero");
    }

    for (size_t i = 0; i < data_size_; ++i) {
      data_[i] /= scalar;
    }
    return *this;
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
      throw std::runtime_error(
          "Unsupported tensor dimensionality for channel statistics");
    }

    return means;
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
        splits.emplace_back(slice_batch(start, end - 1));
      } else {
        throw std::runtime_error(
            "Unsupported tensor dimensionality for splitting");
      }
    }
    return splits;
  }

  Matrix<T> im2col(size_t kernel_h, size_t kernel_w, size_t stride_h = 1,
                   size_t stride_w = 1, size_t pad_h = 0,
                   size_t pad_w = 0) const {
    static_assert(dims_ == 4,
                  "im2col is only supported for 4D tensors (NCHW/NHWC)");

    const Tensor<T, L> *input_ptr = this;
    std::unique_ptr<Tensor<T, L>> padded_input_storage;

    if (pad_h > 0 || pad_w > 0) {
      padded_input_storage = std::make_unique<Tensor<T, L>>(pad(pad_h, pad_w));
      input_ptr = padded_input_storage.get();
    }
    const Tensor<T, L> &input_tensor = *input_ptr;
    const T *input_data = input_tensor.data();

    const size_t in_h = input_tensor.height();
    const size_t in_w = input_tensor.width();
    const size_t out_h = (in_h - kernel_h) / stride_h + 1;
    const size_t out_w = (in_w - kernel_w) / stride_w + 1;
    const size_t channels = input_tensor.channels();
    const size_t batch_size = input_tensor.batch_size();

    size_t col_height = channels * kernel_h * kernel_w;
    size_t col_width = batch_size * out_h * out_w;
    Matrix<T> col_matrix(col_height, col_width);

    utils::parallel_for_2d<size_t>(
        batch_size, channels, [&](size_t n, size_t c) {
          for (size_t kh = 0; kh < kernel_h; ++kh) {
            for (size_t kw = 0; kw < kernel_w; ++kw) {
              size_t col_row_idx = (c * kernel_h + kh) * kernel_w + kw;
              for (size_t out_h_idx = 0; out_h_idx < out_h; ++out_h_idx) {
                for (size_t out_w_idx = 0; out_w_idx < out_w; ++out_w_idx) {
                  size_t in_h_idx = out_h_idx * stride_h + kh;
                  size_t in_w_idx = out_w_idx * stride_w + kw;
                  size_t col_col_idx = (n * out_h + out_h_idx) * out_w + out_w_idx;

                  col_matrix(col_row_idx, col_col_idx) =
                      input_data[n * strides_[0] + c * strides_[1] +
                                 in_h_idx * strides_[2] +
                                 in_w_idx * strides_[3]];
                }
              }
            }
          }
        });
    return col_matrix;
  }

  static Tensor<T, L> col2im(const Matrix<T> &col_matrix, size_t batch_size,
                             size_t channels, size_t height, size_t width,
                             size_t kernel_h, size_t kernel_w, size_t stride_h,
                             size_t stride_w, size_t pad_h, size_t pad_w) {

    size_t padded_h = height + 2 * pad_h;
    size_t padded_w = width + 2 * pad_w;
    size_t output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    size_t output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    Tensor<T, L> result_padded(batch_size, channels, padded_h, padded_w);

    utils::parallel_for_2d<size_t>(
        batch_size, channels, [&](size_t n, size_t c) {
          size_t base_col_row_idx = c * kernel_h * kernel_w;
          size_t base_col_col_idx = n * output_h * output_w;
          for (size_t kh = 0; kh < kernel_h; ++kh) {
            for (size_t kw = 0; kw < kernel_w; ++kw) {
              size_t col_row_idx = base_col_row_idx + kh * kernel_w + kw;
              for (size_t h_out = 0; h_out < output_h; ++h_out) {
                for (size_t w_out = 0; w_out < output_w; ++w_out) {
                  size_t col_col_idx =
                      base_col_col_idx + h_out * output_w + w_out;

                  size_t h_dest = h_out * stride_h + kh;
                  size_t w_dest = w_out * stride_w + kw;

                  result_padded(n, c, h_dest, w_dest) +=
                      col_matrix(col_row_idx, col_col_idx);
                }
              }
            }
          }
        });

    if (pad_h > 0 || pad_w > 0) {
      return result_padded.unpad(pad_h, pad_w);
    } else {
      return result_padded;
    }
  }

  template <Layout new_layout> Tensor<T, new_layout> as_layout() const {
    Tensor<T, new_layout> result(this->shape());

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
      throw std::runtime_error(
          "Conversion for this dimensionality not implemented.");
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
    if (in.gcount() !=
        static_cast<std::streamsize>(tensor.size() * sizeof(T))) {
      throw std::runtime_error("Failed to read tensor data from file");
    }
    return tensor;
  }
};
