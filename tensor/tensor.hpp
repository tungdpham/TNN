#pragma once

#include "../matrix/matrix.hpp"
#include "tensor_view.hpp"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <fstream>

// 4D/5D Tensor template class for CNN operations, supporting various layouts
// but primarily NCHW and NCDHW.
template <typename T = float, Layout layout = NCHW> class Tensor {
  static_assert(std::is_arithmetic<T>::value, "Tensor type must be arithmetic");
  static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
                "Tensor type must be floating point or integral");

private:
  using View = TensorView<T, layout>;

  static constexpr size_t dims = View::dims;
  size_t shape_[dims];  // Shape of the tensor
  size_t strides[dims]; // Strides for each dimension
  std::unique_ptr<T[]> data_;

  size_t data_size_; // Total number of elements in the tensor

  inline void compute_strides() { View::compute_strides(strides, shape_); }

  inline size_t compute_index(size_t batch, size_t channel, size_t height,
                             size_t width) const {
    static_assert(dims == 4, "compute_index only valid for 4D tensors");
    return batch * strides[0] + channel * strides[1] +
           height * strides[2] + width * strides[3];
  }

  inline size_t compute_index(std::initializer_list<size_t> indices) const {
    size_t index = 0;
    for (size_t i = 0; i < dims; ++i) {
      if (i >= indices.size()) {
        throw std::out_of_range("Index out of range for tensor dimensions");
      }
      index += *(indices.begin() + i) * strides[i];
    }
    return index;
  }

public:
  // Constructors
  Tensor() : data_size_(0) {
    std::fill(shape_, shape_ + dims, 0);
    data_ = nullptr;
    compute_strides();
  }

  Tensor(size_t batch, size_t channels, size_t height, size_t width) {
    static_assert(dims == 4,
                  "4-parameter constructor only valid for 4D tensors");
    if (layout == NCDHW || layout == NDHWC) {
      throw std::invalid_argument("5D layout specified for 4D constructor");
    }
    shape_[0] = batch;
    shape_[1] = channels;
    shape_[2] = height;
    shape_[3] = width;
    compute_strides();
    data_size_ = batch * channels * height * width;
    data_ = std::make_unique<T[]>(data_size_);
    std::fill(data_.get(), data_.get() + data_size_,
              T(0)); 
  }

  // 5D constructor for 3D CNNs
  Tensor(size_t batch, size_t channels, size_t depth, size_t height,
         size_t width) {
    static_assert(dims == 5,
                  "5-parameter constructor only valid for 5D tensors");
    if (layout == NCHW || layout == NHWC) {
      throw std::invalid_argument("4D layout specified for 5D constructor");
    }
    shape_[0] = batch;
    shape_[1] = channels;
    shape_[2] = depth;
    shape_[3] = height;
    shape_[4] = width;
    compute_strides();
    data_size_ = batch * channels * depth * height * width;
    data_ = std::make_unique<T[]>(data_size_);
    std::fill(data_.get(), data_.get() + data_size_,
              T(0));
  }

  Tensor(const std::initializer_list<size_t> &shape) {
    // Initialize from a initializer list of dimensions
    if (shape.size() != dims) {
      throw std::invalid_argument("Shape must have " + std::to_string(dims) +
                                  " dimensions");
    }
    std::copy(shape.begin(), shape.end(), shape_);
    compute_strides();
    data_size_ =
        std::accumulate(shape_, shape_ + dims, 1UL, std::multiplies<size_t>());
    data_ = std::make_unique<T[]>(data_size_);
    std::fill(data_.get(), data_.get() + data_size_,
              T(0));
  }

  Tensor(const std::vector<size_t> &shape) {
    assert(shape.size() == dims && "Shape must match dimensions");
    std::copy(shape.begin(), shape.end(), shape_);
    compute_strides();
    data_size_ =
        std::accumulate(shape_, shape_ + dims, 1UL, std::multiplies<size_t>());
    data_ = std::make_unique<T[]>(data_size_);
    std::fill(data_.get(), data_.get() + data_size_, T(0));
  }

  // Constructor with initial data
  Tensor(const std::vector<size_t> &shape, const std::vector<T> &data) {
    assert(shape.size() == dims && "Shape must match dimensions");
    std::copy(shape.begin(), shape.end(), shape_);
    compute_strides();
    data_size_ =
        std::accumulate(shape_, shape_ + dims, 1UL, std::multiplies<size_t>());
    if (data.size() != data_size_) {
      throw std::invalid_argument("Data size doesn't match tensor shape");
    }
    data_ = std::make_unique<T[]>(data_size_);
    std::copy(data.begin(), data.end(), data_.get());
  }

  ~Tensor() = default;

  // Copy constructor
  Tensor(const Tensor &other) : data_size_(other.data_size_) {
    std::copy(other.shape_, other.shape_ + dims, shape_);
    compute_strides();
    if (data_size_ > 0) {
      data_ = std::make_unique<T[]>(data_size_);
      std::copy(other.data_.get(),
                other.data_.get() + data_size_, data_.get());
    }
  }

  // Move constructor
  Tensor(Tensor &&other) noexcept
      : data_(std::move(other.data_)), data_size_(other.data_size_) {
    std::copy(other.shape_, other.shape_ + dims, shape_);
    compute_strides();
    other.data_size_ = 0;
  }

  // Assignment operators
  Tensor &operator=(const Tensor &other) {
    if (this != &other) {
      std::copy(other.shape_, other.shape_ + dims, shape_);
      compute_strides();
      data_size_ = other.data_size_;
      data_ = std::make_unique<T[]>(data_size_);
      std::copy(other.data_.get(),
                other.data_.get() + data_size_, data_.get());
    }
    return *this;
  }

  Tensor &operator=(Tensor &&other) noexcept {
    if (this != &other) {
      std::copy(other.shape_, other.shape_ + dims, shape_);
      compute_strides();
      data_ = std::move(other.data_);
      data_size_ = other.data_size_;
      other.data_size_ = 0;
    }
    return *this;
  }

  // Accessors for 4D tensors
  T &operator()(size_t n, size_t c, size_t h, size_t w) {
    return data_[compute_index(n, c, h, w)];
  }

  const T &operator()(size_t n, size_t c, size_t h, size_t w) const {
    return data_[compute_index(n, c, h, w)];
  }

  T &operator()(const std::vector<size_t> &indices) {
    return data_[compute_index(indices)];
  }

  const T &operator()(const std::vector<size_t> &indices) const {
    return data_[compute_index(indices)];
  }

  // Shape information
  std::vector<size_t> shape() const {
    return std::vector<size_t>(shape_, shape_ + dims);
  }

  std::string shape_str() const {
    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0; i < dims; ++i) {
      oss << shape_[i];
      if (i < dims - 1) {
        oss << ", ";
      }
    }
    oss << "}";
    return oss.str();
  }

  const size_t *shape_ptr() const { return shape_; }

  const size_t *strides_ptr() const { return strides; }

  size_t batch_size() const { return shape_[0]; }

  size_t channels() const { return shape_[1]; }

  size_t height() const {
    if constexpr (dims == 4) {
      return shape_[2];
    } else if constexpr (dims == 5) {
      return shape_[3]; // For 5D, height is the 4th dimension
    } else {
      throw std::runtime_error(
          "height() called on unsupported tensor dimensionality");
    }
  }

  size_t width() const {
    if constexpr (dims == 4) {
      return shape_[3];
    } else if constexpr (dims == 5) {
      return shape_[4]; // For 5D, width is the 5th dimension
    } else {
      throw std::runtime_error(
          "width() called on unsupported tensor dimensionality");
    }
  }

  size_t depth() const {
    if constexpr (dims == 5) {
      return shape_[2];
    } else {
      return 1;
    }
  }

  // Generic dimension accessors
  size_t dimension(size_t index) const {
    assert(index < dims && "Dimension index out of range");
    return shape_[index];
  }

  size_t stride(size_t index) const {
    assert(index < dims && "Stride index out of range");
    return strides[index];
  }

  size_t num_dimensions() const { return dims; }

  static constexpr size_t expected_dimensions() { return dims; }

  constexpr bool is_4d() const { return dims == 4; }

  static constexpr bool is_expected_4d() { return dims == 4; }

  size_t size() const { return data_size_; }

  // Data access
  T *data() { return data_.get(); }

  const T *data() const { return data_.get(); }

  // Clone
  Tensor<T, layout> clone() const {
    return Tensor<T, layout>(
        std::vector<size_t>(shape_, shape_ + dims),
        std::vector<T>(data_.get(), data_.get() + data_size_));
  }

  // Fill operations
  void fill(T value) {
    std::fill(data_.get(), data_.get() + data_size_,
              value);
  }

  void fill_random_uniform(T range) {
    std::mt19937 gen(std::random_device{}());
    if constexpr (std::is_floating_point<T>::value) {
      std::uniform_real_distribution<T> dis(-range, range);
      for (size_t i = 0; i < data_size_; ++i) {
        data_[i] = dis(gen);
      }
    } else {
      // For integral types, use integer distribution
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

  // Reshape (must preserve total size)
  Tensor<T, layout> reshape(const std::vector<size_t> &new_shape) const {
    // Check if new shape is same as current shape
    bool same_shape = (new_shape.size() == dims);
    if (same_shape) {
      for (size_t i = 0; i < dims; ++i) {
        if (new_shape[i] != shape_[i]) {
          same_shape = false;
          break;
        }
      }
    }

    if (same_shape) {
      return *this;
    }

    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1UL,
                                      std::multiplies<size_t>());
    if (new_size != size()) {
      throw std::invalid_argument("New shape must have same total size");
    }
    std::vector<T> temp_data(data_.get(), data_.get() + data_size_);
    return Tensor<T, layout>(new_shape, temp_data);
  }

  // Padding operations
  Tensor<T, layout> pad(size_t pad_h, size_t pad_w, T value = T(0)) const {
    assert(dims == 4 && "Padding only supported for 4D tensors");
    if (pad_h == 0 && pad_w == 0) {
      return *this; // No padding needed
    }

    Tensor<T, layout> result(batch_size(), channels(), height() + 2 * pad_h,
                             width() + 2 * pad_w);
    result.fill(value);

#ifdef _OPENMP
#pragma omp parallel for collapse(4) schedule(static)
#endif
    for (size_t n = 0; n < batch_size(); ++n) {
      for (size_t c = 0; c < channels(); ++c) {
        for (size_t h = 0; h < height(); ++h) {
          for (size_t w = 0; w < width(); ++w) {
            result(n, c, h + pad_h, w + pad_w) = (*this)(n, c, h, w);
          }
        }
      }
    }

    return result;
  }

  // Cropping operations for 4D tensors
  Tensor<T, layout> crop(size_t start_h, size_t start_w, size_t end_h,
                         size_t end_w) const {
    if constexpr (dims != 4) {
      throw std::runtime_error("2D cropping only supported for 4D tensors");
    }

    if (end_h >= height() || end_w >= width() || start_h > end_h ||
        start_w > end_w) {
      throw std::invalid_argument("Invalid crop dimensions");
    }

    size_t new_height = end_h - start_h + 1;
    size_t new_width = end_w - start_w + 1;

    Tensor<T, layout> result(batch_size(), channels(), new_height, new_width);

#ifdef _OPENMP
#pragma omp parallel for collapse(4) schedule(static)
#endif
    for (size_t n = 0; n < batch_size(); ++n) {
      for (size_t c = 0; c < channels(); ++c) {
        for (size_t h = 0; h < new_height; ++h) {
          for (size_t w = 0; w < new_width; ++w) {
            result(n, c, h, w) = (*this)(n, c, start_h + h, start_w + w);
          }
        }
      }
    }

    return result;
  }

  // Slicing operations
  Tensor<T, layout> slice_batch(size_t start_batch, size_t end_batch) const {
    if (end_batch >= batch_size() || start_batch > end_batch) {
      throw std::invalid_argument("Invalid batch slice range");
    }

    size_t new_batch_size = end_batch - start_batch + 1;

    if constexpr (dims == 4) {
      Tensor<T, layout> result(new_batch_size, channels(), height(), width());
      T* result_data = result.data();
      size_t output_size = channels() * height() * width();
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
      for (size_t n = 0; n < new_batch_size; ++n) {
        for(size_t idx = 0; idx < output_size; ++idx) {
          size_t batch_idx = start_batch + n;
          result_data[n * output_size + idx] = this->data_[batch_idx * strides[0] + idx];
        }
      }
      return result;
    } else {
      throw std::runtime_error(
          "Unsupported tensor dimensionality for batch slicing");
    }
  }

  Tensor<T, layout> slice_channels(size_t start_ch, size_t end_ch) const {
    if (end_ch >= channels() || start_ch > end_ch) {
      throw std::invalid_argument("Invalid channel slice range");
    }

    size_t new_channels = end_ch - start_ch + 1;

    if constexpr (dims == 4) {
      Tensor<T, layout> result(batch_size(), new_channels, height(), width());

#ifdef _OPENMP
#pragma omp parallel for collapse(4) schedule(static)
#endif
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

  // Arithmetic operations
  Tensor<T, layout> operator+(const Tensor<T, layout> &other) const {
    // Compare shapes element by element
    for (size_t i = 0; i < dims; ++i) {
      if (shape_[i] != other.shape_[i]) {
        std::cerr << "Shape mismatch: " << shape_[i] << " vs " << other.shape_[i] << std::endl;
        throw std::invalid_argument("Tensor shapes must match for addition");
      }
    }

    std::vector<size_t> shape_vec(shape_, shape_ + dims);
    Tensor<T, layout> result(shape_vec);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t idx = 0; idx < data_size_; ++idx)
      result.data_[idx] = data_[idx] + other.data_[idx];

    return result;
  }

  Tensor<T, layout> operator-(const Tensor<T, layout> &other) const {
    // Compare shapes element by element
    for (size_t i = 0; i < dims; ++i) {
      if (shape_[i] != other.shape_[i]) {
        throw std::invalid_argument("Tensor shapes must match for subtraction");
      }
    }

    std::vector<size_t> shape_vec(shape_, shape_ + dims);
    Tensor<T, layout> result(shape_vec);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t idx = 0; idx < data_size_; ++idx) {
      result.data_[idx] = data_[idx] - other.data_[idx];
    }

    return result;
  }

  Tensor<T, layout> operator*(T scalar) const {
    std::vector<size_t> shape_vec(shape_, shape_ + dims);
    Tensor<T, layout> result(shape_vec);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < data_size_; ++i) {
      result.data_[i] = data_[i] * scalar;
    }
    return result;
  }

  Tensor<T, layout> operator/(T scalar) const {
    if (scalar == T(0)) {
      throw std::invalid_argument("Division by zero");
    }

    std::vector<size_t> shape_vec(shape_, shape_ + dims);
    Tensor<T, layout> result(shape_vec);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < data_size_; ++i) {
      result.data_[i] = data_[i] / scalar;
    }
    return result;
  }

  Tensor<T, layout> &operator+=(const Tensor<T, layout> &other) {
    // Compare shapes element by element
    for (size_t i = 0; i < dims; ++i) {
      if (shape_[i] != other.shape_[i]) {
        std::cerr << "Shape mismatch: " << shape_[i] << " vs " << other.shape_[i] << std::endl;
        throw std::invalid_argument("Tensor shapes must match for addition");
      }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t idx = 0; idx < data_size_; ++idx) {
      data_[idx] += other.data_[idx];
    }

    return *this;
  }

  Tensor<T, layout> &operator-=(const Tensor<T, layout> &other) {
    // Compare shapes element by element
    for (size_t i = 0; i < dims; ++i) {
      if (shape_[i] != other.shape_[i]) {
        throw std::invalid_argument("Tensor shapes must match for subtraction");
      }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t idx = 0; idx < data_size_; ++idx) {
      data_[idx] -= other.data_[idx];
    }

    return *this;
  }

  Tensor<T, layout> &operator*=(const Tensor<T, layout> &other) {
    // Compare shapes element by element
    for (size_t i = 0; i < dims; ++i) {
      if (shape_[i] != other.shape_[i]) {
        throw std::invalid_argument(
            "Tensor shapes must match for element-wise multiplication");
      }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t idx = 0; idx < data_size_; ++idx) {
      data_[idx] *= other.data_[idx];
    }

    return *this;
  }

  Tensor<T, layout> &operator*=(T scalar) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < data_size_; ++i) {
      data_[i] *= scalar;
    }
    return *this;
  }

  Tensor<T, layout> &operator/=(T scalar) {
    if (scalar == T(0)) {
      throw std::invalid_argument("Division by zero");
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < data_size_; ++i) {
      data_[i] /= scalar;
    }
    return *this;
  }

  // Statistical operations
  T mean() const {
    T sum = T(0);
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum) schedule(static)
#endif
    for (size_t i = 0; i < data_size_; ++i) {
      sum += data_[i];
    }
    return sum / static_cast<T>(data_size_);
  }

  T variance() const {
    T m = mean();
    T sum_sq_diff = T(0);
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum_sq_diff) schedule(static)
#endif
    for (size_t i = 0; i < data_size_; ++i) {
      T diff = data_[i] - m;
      sum_sq_diff += diff * diff;
    }
    return sum_sq_diff / static_cast<T>(data_size_);
  }

  // Channel-wise statistics (useful for batch normalization)
  std::vector<T> channel_means() const {
    std::vector<T> means(channels(), T(0));

    if constexpr (dims == 4) {
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

      if constexpr (dims == 4) {
        splits.emplace_back(slice_batch(start, end - 1));
      } else {
        throw std::runtime_error(
            "Unsupported tensor dimensionality for splitting");
      }
    }
    return splits;
  }

  // To row major vector (for NCHW it's the same as data)
  std::vector<T> to_vector() const {
    if constexpr (layout == NCHW) {
      return std::vector<T>(data_.get(), data_.get() + data_size_);
    } else {
      throw std::runtime_error("to_vector is only supported for NCHW layout");
    }
  }

  // Load data from row major vector (for NCHW it's the same as data)
  void from_vector(const std::vector<T> &vec) {
    if (vec.size() != data_size_) {
      throw std::invalid_argument("Vector size does not match tensor size");
    }

    if constexpr (layout != NCHW) {
      throw std::runtime_error("from_vector is only supported for NCHW layout");
    }

    std::copy(vec.begin(), vec.end(), data_.get());
  }

  // CNN-specific operations (to be implemented with convolution layers)
  Matrix<T> im2col(size_t kernel_h, size_t kernel_w, size_t stride_h = 1,
                   size_t stride_w = 1, size_t pad_h = 0,
                   size_t pad_w = 0) const {
    static_assert(dims == 4,
                  "im2col is only supported for 4D tensors (NCHW/NHWC)");

    // Apply padding if needed - avoid copy when no padding
    const Tensor<T, layout> *input_ptr = this;
    std::unique_ptr<Tensor<T, layout>> padded_input_storage;

    if (pad_h > 0 || pad_w > 0) {
      padded_input_storage =
          std::make_unique<Tensor<T, layout>>(pad(pad_h, pad_w));
      input_ptr = padded_input_storage.get();
    }
    const Tensor<T, layout> &input_tensor = *input_ptr;
    const T* input_data = input_tensor.data();
    const size_t *strides = input_tensor.strides_ptr();

    const size_t in_h = input_tensor.height();
    const size_t in_w = input_tensor.width();
    const size_t out_h = (in_h - kernel_h) / stride_h + 1;
    const size_t out_w = (in_w - kernel_w) / stride_w + 1;
    const size_t channels = input_tensor.channels();
    const size_t batch_size = input_tensor.batch_size();

    size_t col_height = channels * kernel_h * kernel_w;
    size_t col_width = batch_size * out_h * out_w;
    Matrix<T> col_matrix(col_height, col_width);

#ifdef _OPENMP
#pragma omp parallel for collapse(4)
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t c = 0; c < channels; ++c) {
        for (size_t kh = 0; kh < kernel_h; ++kh) {
          for (size_t kw = 0; kw < kernel_w; ++kw) {
            size_t col_row_idx = (c * kernel_h + kh) * kernel_w + kw;
            for (size_t out_h_idx = 0; out_h_idx < out_h; ++out_h_idx) {
              for (size_t out_w_idx = 0; out_w_idx < out_w; ++out_w_idx) {
                size_t in_h_idx = out_h_idx * stride_h + kh;
                size_t in_w_idx = out_w_idx * stride_w + kw;
                size_t col_col_idx =
                    n * out_h * out_w + out_h_idx * out_w + out_w_idx;

                col_matrix(col_row_idx, col_col_idx) =
                    input_data[n * strides[0] + c * strides[1] +
                               in_h_idx * strides[2] + in_w_idx * strides[3]];
              }
            }
          }
        }
      }
    }
    return col_matrix;
  }

  static Tensor<T, layout> col2im(
      const Matrix<T> &col_matrix, size_t batch_size, size_t channels,
      size_t height, size_t width, size_t kernel_h, size_t kernel_w,
      size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w) {

    // Calculate output dimensions
    size_t padded_h = height + 2 * pad_h;
    size_t padded_w = width + 2 * pad_w;
    size_t output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    size_t output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    Tensor<T, layout> result_padded(batch_size, channels, padded_h, padded_w);

    // The total number of output "patches" is the number of columns in
    // col_matrix
    size_t num_output_patches = output_h * output_w;

    size_t col_rows = channels * kernel_h * kernel_w;
    size_t col_cols = batch_size * num_output_patches;

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t h_out = 0; h_out < output_h; ++h_out) {
        for (size_t w_out = 0; w_out < output_w; ++w_out) {
          // This corresponds to a single output patch position
          size_t col_col_idx =
              n * output_h * output_w + h_out * output_w + w_out;

          for (size_t c = 0; c < channels; ++c) {
            for (size_t kh = 0; kh < kernel_h; ++kh) {
              for (size_t kw = 0; kw < kernel_w; ++kw) {

                size_t col_row_idx = (c * kernel_h + kh) * kernel_w + kw;

                size_t h_dest = h_out * stride_h + kh;
                size_t w_dest = w_out * stride_w + kw;

                result_padded(n, c, h_dest, w_dest) +=
                    col_matrix(col_row_idx, col_col_idx);
              }
            }
          }
        }
      }
    }

    // Now handle the padding removal
    if (pad_h > 0 || pad_w > 0) {
      return result_padded.crop(pad_h, pad_w, padded_h - pad_h - 1,
                                padded_w - pad_w - 1);
    } else {
      return result_padded;
    }
  }


  // Combine multiple tensors into a single tensor with same CHW size and batch size stacked upon each other
  // This is useful for combining outputs from multiple layers or branches in a network
  static Tensor<T> combine(std::vector<Tensor<T>> &tensors) {
    if (tensors.empty()) {
      throw std::invalid_argument("No tensors to combine");
    }

    size_t channels = tensors[0].channels();
    size_t height = tensors[0].height();
    size_t width = tensors[0].width();

    for (const auto &tensor : tensors) {
      if (tensor.channels() != channels ||
          tensor.height() != height || tensor.width() != width) {
        throw std::invalid_argument("All tensors must have the same shape");
      }
    }

    size_t total_batch_size = std::accumulate(
        tensors.begin(), tensors.end(), 0UL,
        [](size_t sum, const Tensor<T> &tensor) { return sum + tensor.batch_size(); });

    Tensor<T> combined(total_batch_size, channels, height, width);

    size_t offset = 0;
    for (const auto &tensor : tensors) {
      size_t batch_size = tensor.batch_size();
      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t c = 0; c < channels; ++c) {
          for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
              combined(offset + n, c, h, w) = tensor(n, c, h, w);
            } 
          }
        }
      }
      offset += batch_size;
    }

    return combined;
  }

  template <Layout new_layout> Tensor<T, new_layout> as_layout() const {
    Tensor<T, new_layout> result(this->shape());

    if constexpr (dims == 4) {
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

  // Print tensor information
  void print_info() const {
    std::cout << "Tensor shape: [";
    for (size_t i = 0; i < dims; ++i) {
      std::cout << shape_[i];
      if (i < dims - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Layout: ";
    switch (layout) {
    case NCHW:
      std::cout << "NCHW";
      break;
    case NHWC:
      std::cout << "NHWC";
      break;
    case NCDHW:
      std::cout << "NCDHW";
      break;
    case NDHWC:
      std::cout << "NDHWC";
      break;
    }
    std::cout << std::endl;
    std::cout << "Dimensions: " << num_dimensions() << "D" << std::endl;
    std::cout << "Total size: " << size() << std::endl;
    std::cout << "Mean: " << mean() << std::endl;
  }

  void print_data() const {
    std::cout << "Tensor data: ";
    if constexpr (dims == 4){
      for(size_t n = 0; n < batch_size(); ++n) {
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

  // Get the index of the maximum value in a specific channel for a given
  // batch, height, and width.
  int argmax_channel(size_t n, size_t h, size_t w) const {
    if (n >= batch_size() || h >= height() || w >= width()) {
      throw std::out_of_range("Index out of range in argmax_channel");
    }

    T max_val = -std::numeric_limits<T>::infinity();
    int max_idx = -1;

    for (size_t c = 0; c < channels(); ++c) {
      T val = operator()(n, c, h, w);
      if (val > max_val) {
        max_val = val;
        max_idx = c;
      }
    }
    return max_idx;
  }

  // Serialization methods
  void save(std::ofstream &out) const {
    if (!out.is_open()) {
      throw std::runtime_error("File is not open for writing");
    }
    // Write shape
    out.write(reinterpret_cast<const char *>(shape_), dims * sizeof(size_t));
    // Write data
    out.write(reinterpret_cast<const char *>(data_.get()),
              data_size_ * sizeof(T));
  }

  static Tensor<T, layout> load(std::ifstream &in) {
    if (!in.is_open()) {
      throw std::runtime_error("File is not open for reading");
    }
    std::vector<size_t> shape(dims);
    in.read(reinterpret_cast<char *>(shape.data()), dims * sizeof(size_t));
    if (in.gcount() != dims * sizeof(size_t)) {
      throw std::runtime_error("Failed to read tensor shape from file");
    }

    Tensor<T, layout> tensor(shape);
    in.read(reinterpret_cast<char *>(tensor.data()), tensor.size() * sizeof(T));
    if (in.gcount() != tensor.size() * sizeof(T)) {
      throw std::runtime_error("Failed to read tensor data from file");
    }
    return tensor;
  }
};
