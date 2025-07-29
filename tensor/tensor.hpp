#pragma once

#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../matrix/matrix.hpp"

enum Layout {
  NCHW,  // 4D: Batch, Channels, Height, Width (most common for CNNs)
  NHWC,  // 4D: Batch, Height, Width, Channels (TensorFlow default)
  NCDHW, // 5D: Batch, Channels, Depth, Height, Width (3D CNNs)
  NDHWC  // 5D: Batch, Depth, Height, Width, Channels (3D TensorFlow default)
};

template <Layout L> struct LayoutTraits;

template <> struct LayoutTraits<NCHW> {
  static constexpr size_t dims = 4;
};

template <> struct LayoutTraits<NHWC> {
  static constexpr size_t dims = 4;
};

template <> struct LayoutTraits<NCDHW> {
  static constexpr size_t dims = 5;
};

template <> struct LayoutTraits<NDHWC> {
  static constexpr size_t dims = 5;
};

// 4D/5D Tensor template class for CNN operations
// Supports NCHW (Batch, Channels, Height, Width) layout primarily for 4D
// Also supports NHWC layout for compatibility
// Supports NCDHW (Batch, Channels, Depth, Height, Width) and NDHWC for 5D
// Template parameter T can be float, double, or other numeric types
template <typename T = float, Layout layout = NCHW> class Tensor {
  static_assert(std::is_arithmetic<T>::value, "Tensor type must be arithmetic");
  static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
                "Tensor type must be floating point or integral");

public:
  // Inferred dimensionality based on layout
  static constexpr size_t dims = LayoutTraits<layout>::dims;

private:
  size_t shape_[dims]; // Raw array: [batch, channels, height, width] for 4D or
                       // [batch, channels, depth, height, width] for 5D
  std::unique_ptr<T[]>
      data_;         // Contiguous raw array storage for better performance
  size_t data_size_; // Total number of elements
  size_t stride_n;   // Strides between batches
  size_t stride_c;   // Strides between channels
  size_t stride_h;   // Strides between height
  size_t stride_w;   // Strides between width
  size_t stride_d;   // Strides between depth (only for 5D tensors)

  inline void compute_strides() { // Should be called after shape is set
    if constexpr (dims == 4) {
      if constexpr (layout == NCHW) {
        stride_n = shape_[1] * shape_[2] * shape_[3];
        stride_c = shape_[2] * shape_[3];
        stride_h = shape_[3];
        stride_w = 1;
      } else if constexpr (layout == NHWC) {
        stride_n = shape_[2] * shape_[3] * shape_[1];
        stride_h = shape_[3] * shape_[1];
        stride_w = shape_[1];
        stride_c = 1;
      } else {
        throw std::runtime_error("Unsupported layout for 4D tensor");
      }
    } else if constexpr (dims == 5) {
      if constexpr (layout == NCDHW) {
        stride_n = shape_[1] * shape_[2] * shape_[3] * shape_[4];
        stride_c = shape_[2] * shape_[3] * shape_[4];
        stride_d = shape_[3] * shape_[4];
        stride_h = shape_[4];
        stride_w = 1;
      } else if constexpr (layout == NDHWC) {
        stride_n = shape_[2] * shape_[3] * shape_[4] * shape_[1];
        stride_d = shape_[3] * shape_[4] * shape_[1];
        stride_h = shape_[4] * shape_[1];
        stride_w = shape_[1];
        stride_c = 1;
      } else {
        throw std::runtime_error("Unsupported layout for 5D tensor");
      }
    }
  }

  inline size_t compute_index(size_t n, size_t c, size_t h, size_t w) const {
    if constexpr (dims != 4) {
      throw std::runtime_error("4D indexing called on non-4D tensor template");
    }

#ifdef NDEBUG
    // Bounds checking
    if (n >= shape_[0] || c >= shape_[1] || h >= shape_[2] || w >= shape_[3]) {
      throw std::out_of_range("Tensor index out of bounds");
    }
#endif
    if constexpr (layout == NCHW) {
      return n * stride_n + c * stride_c + h * stride_h + w;
    } else if constexpr (layout == NHWC) {
      return n * stride_n + h * stride_h + w * stride_w + c;
    } else {
      throw std::runtime_error("Unsupported layout for 4D tensor");
    }
  }

  inline size_t compute_index(size_t n, size_t c, size_t d, size_t h,
                              size_t w) const {
    if constexpr (dims != 5) {
      throw std::runtime_error("5D indexing called on non-5D tensor template");
    }

#ifdef NDEBUG
    if (n >= shape_[0] || c >= shape_[1] || d >= shape_[2] || h >= shape_[3] ||
        w >= shape_[4]) {
      throw std::out_of_range("Tensor index out of bounds");
    }
#endif

    if constexpr (layout == NCDHW) {
      return n * stride_n + c * stride_c + d * stride_d + h * stride_h + w;
    } else if constexpr (layout == NDHWC) {
      return n * stride_n + d * stride_d + h * stride_h + w * stride_w + c;
    } else {
      throw std::runtime_error("Unsupported layout for 5D tensor");
    }
  }

  inline size_t compute_index(const std::vector<size_t> &indices) const {
    if (indices.size() != dims) {
      throw std::runtime_error(
          "Index vector size doesn't match tensor dimensionality");
    }

    if constexpr (dims == 4) {
      return compute_index(indices[0], indices[1], indices[2], indices[3]);
    } else if constexpr (dims == 5) {
      return compute_index(indices[0], indices[1], indices[2], indices[3],
                           indices[4]);
    } else {
      throw std::runtime_error("Unsupported tensor dimensionality");
    }
  }

public:
  // Constructors
  Tensor() : data_size_(0) {
    if constexpr (dims == 4) {
      shape_[0] = shape_[1] = shape_[2] = shape_[3] = 0;
    } else if constexpr (dims == 5) {
      shape_[0] = shape_[1] = shape_[2] = shape_[3] = shape_[4] = 0;
    }
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
    std::fill(data_.get(), data_.get() + data_size_, T(0));
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
    std::fill(data_.get(), data_.get() + data_size_, T(0));
  }

  Tensor(const std::vector<size_t> &shape) {
    if (shape.size() != dims) {
      throw std::invalid_argument(
          "Shape dimensions don't match template parameter");
    }
    if constexpr (dims == 4) {
      if (layout == NCDHW || layout == NDHWC) {
        throw std::invalid_argument("5D layout specified for 4D tensor");
      }
    } else if constexpr (dims == 5) {
      if (layout == NCHW || layout == NHWC) {
        throw std::invalid_argument("4D layout specified for 5D tensor");
      }
    }
    std::copy(shape.begin(), shape.end(), shape_);
    compute_strides();
    data_size_ =
        std::accumulate(shape_, shape_ + dims, 1UL, std::multiplies<size_t>());
    data_ = std::make_unique<T[]>(data_size_);
    std::fill(data_.get(), data_.get() + data_size_, T(0));
  }

  // Constructor with initial data
  Tensor(const std::vector<size_t> &shape, const std::vector<T> &data) {
    if (shape.size() != dims) {
      throw std::invalid_argument(
          "Shape dimensions don't match template parameter");
    }
    if constexpr (dims == 4) {
      if (layout == NCDHW || layout == NDHWC) {
        throw std::invalid_argument("5D layout specified for 4D tensor");
      }
    } else if constexpr (dims == 5) {
      if (layout == NCHW || layout == NHWC) {
        throw std::invalid_argument("4D layout specified for 5D tensor");
      }
    }
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
      std::copy(other.data_.get(), other.data_.get() + data_size_, data_.get());
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
      if (data_size_ > 0) {
        data_ = std::make_unique<T[]>(data_size_);
        std::copy(other.data_.get(), other.data_.get() + data_size_,
                  data_.get());
      } else {
        data_.reset();
      }
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

  // Accessors for 5D tensors
  T &operator()(size_t n, size_t c, size_t d, size_t h, size_t w) {
    return data_[compute_index(n, c, d, h, w)];
  }

  const T &operator()(size_t n, size_t c, size_t d, size_t h, size_t w) const {
    return data_[compute_index(n, c, d, h, w)];
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

  const size_t *shape_ptr() const { return shape_; }

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
    if (index >= dims) {
      throw std::out_of_range("Dimension index out of range");
    }
    return shape_[index];
  }

  size_t num_dimensions() const { return dims; }

  static constexpr size_t expected_dimensions() { return dims; }

  constexpr bool is_4d() const { return dims == 4; }

  constexpr bool is_5d() const { return dims == 5; }

  static constexpr bool is_expected_4d() { return dims == 4; }

  static constexpr bool is_expected_5d() { return dims == 5; }

  size_t size() const { return data_size_; }

  // Data access
  T *data() { return data_.get(); }

  const T *data() const { return data_.get(); }

  // Fill operations
  void fill(T value) {
    std::fill(data_.get(), data_.get() + data_size_, value);
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

  void fill_random_normal(T stddev) {
    static_assert(std::is_floating_point<T>::value,
                  "Normal distribution requires floating point type");
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<T> dis(T(0), stddev);
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
    if constexpr (dims != 4) {
      throw std::runtime_error("2D padding only supported for 4D tensors");
    }

    Tensor<T, layout> result(batch_size(), channels(), height() + 2 * pad_h,
                             width() + 2 * pad_w);
    result.fill(value);

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
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

  // 3D padding for 5D tensors
  Tensor<T, layout> pad_3d(size_t pad_d, size_t pad_h, size_t pad_w,
                           T value = T(0)) const {
    if constexpr (dims != 5) {
      throw std::runtime_error("3D padding only supported for 5D tensors");
    }

    Tensor<T, layout> result(batch_size(), channels(), depth() + 2 * pad_d,
                             height() + 2 * pad_h, width() + 2 * pad_w);
    result.fill(value);

    for (size_t n = 0; n < batch_size(); ++n) {
      for (size_t c = 0; c < channels(); ++c) {
        for (size_t d = 0; d < depth(); ++d) {
          for (size_t h = 0; h < height(); ++h) {
            for (size_t w = 0; w < width(); ++w) {
              result(n, c, d + pad_d, h + pad_h, w + pad_w) =
                  (*this)(n, c, d, h, w);
            }
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
#pragma omp parallel for collapse(3)
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

  // 3D cropping for 5D tensors
  Tensor<T, layout> crop_3d(size_t start_d, size_t start_h, size_t start_w,
                            size_t end_d, size_t end_h, size_t end_w) const {
    if constexpr (dims != 5) {
      throw std::runtime_error("3D cropping only supported for 5D tensors");
    }

    if (end_d >= depth() || end_h >= height() || end_w >= width() ||
        start_d > end_d || start_h > end_h || start_w > end_w) {
      throw std::invalid_argument("Invalid 3D crop dimensions");
    }

    size_t new_depth = end_d - start_d + 1;
    size_t new_height = end_h - start_h + 1;
    size_t new_width = end_w - start_w + 1;

    Tensor<T, layout> result(batch_size(), channels(), new_depth, new_height,
                             new_width);

    for (size_t n = 0; n < batch_size(); ++n) {
      for (size_t c = 0; c < channels(); ++c) {
        for (size_t d = 0; d < new_depth; ++d) {
          for (size_t h = 0; h < new_height; ++h) {
            for (size_t w = 0; w < new_width; ++w) {
              result(n, c, d, h, w) =
                  (*this)(n, c, start_d + d, start_h + h, start_w + w);
            }
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
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
      for (size_t n = 0; n < new_batch_size; ++n) {
        for (size_t c = 0; c < channels(); ++c) {
          for (size_t h = 0; h < height(); ++h) {
            for (size_t w = 0; w < width(); ++w) {
              result(n, c, h, w) = (*this)(start_batch + n, c, h, w);
            }
          }
        }
      }
      return result;
    } else if constexpr (dims == 5) {
      Tensor<T, layout> result(new_batch_size, channels(), depth(), height(),
                               width());

      for (size_t n = 0; n < new_batch_size; ++n) {
        for (size_t c = 0; c < channels(); ++c) {
          for (size_t d = 0; d < depth(); ++d) {
            for (size_t h = 0; h < height(); ++h) {
              for (size_t w = 0; w < width(); ++w) {
                result(n, c, d, h, w) = (*this)(start_batch + n, c, d, h, w);
              }
            }
          }
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
#pragma omp parallel for collapse(3)
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
    } else if constexpr (dims == 5) {
      Tensor<T, layout> result(batch_size(), new_channels, depth(), height(),
                               width());

      for (size_t n = 0; n < batch_size(); ++n) {
        for (size_t c = 0; c < new_channels; ++c) {
          for (size_t d = 0; d < depth(); ++d) {
            for (size_t h = 0; h < height(); ++h) {
              for (size_t w = 0; w < width(); ++w) {
                result(n, c, d, h, w) = (*this)(n, start_ch + c, d, h, w);
              }
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

  // Depth slicing for 5D tensors
  Tensor<T, layout> slice_depth(size_t start_d, size_t end_d) const {
    if (!is_5d()) {
      throw std::runtime_error("Depth slicing only supported for 5D tensors");
    }

    if (end_d >= depth() || start_d > end_d) {
      throw std::invalid_argument("Invalid depth slice range");
    }

    size_t new_depth = end_d - start_d + 1;
    Tensor<T, layout> result(batch_size(), channels(), new_depth, height(),
                             width());

    for (size_t n = 0; n < batch_size(); ++n) {
      for (size_t c = 0; c < channels(); ++c) {
        for (size_t d = 0; d < new_depth; ++d) {
          for (size_t h = 0; h < height(); ++h) {
            for (size_t w = 0; w < width(); ++w) {
              result(n, c, d, h, w) = (*this)(n, c, start_d + d, h, w);
            }
          }
        }
      }
    }

    return result;
  }

  // Arithmetic operations
  Tensor<T, layout> operator+(const Tensor<T, layout> &other) const {
    // Compare shapes element by element
    for (size_t i = 0; i < dims; ++i) {
      if (shape_[i] != other.shape_[i]) {
        throw std::invalid_argument("Tensor shapes must match for addition");
      }
    }

    std::vector<size_t> shape_vec(shape_, shape_ + dims);
    Tensor<T, layout> result(shape_vec);

    if constexpr (dims == 4) {
#ifdef _OPENMP
#pragma omp parallel for collapse(4)
#endif
      for (size_t n = 0; n < batch_size(); ++n) {
        for (size_t c = 0; c < channels(); ++c) {
          for (size_t h = 0; h < height(); ++h) {
            for (size_t w = 0; w < width(); ++w) {
              result(n, c, h, w) = (*this)(n, c, h, w) + other(n, c, h, w);
            }
          }
        }
      }
    } else if constexpr (dims == 5) {
      for (size_t n = 0; n < batch_size(); ++n) {
        for (size_t c = 0; c < channels(); ++c) {
          for (size_t d = 0; d < depth(); ++d) {
            for (size_t h = 0; h < height(); ++h) {
              for (size_t w = 0; w < width(); ++w) {
                result(n, c, d, h, w) =
                    (*this)(n, c, d, h, w) + other(n, c, d, h, w);
              }
            }
          }
        }
      }
    } else {
      throw std::runtime_error(
          "Unsupported tensor dimensionality for addition");
    }
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

    if constexpr (dims == 4) {
#ifdef _OPENMP
#pragma omp parallel for collapse(4)
#endif
      for (size_t n = 0; n < batch_size(); ++n) {
        for (size_t c = 0; c < channels(); ++c) {
          for (size_t h = 0; h < height(); ++h) {
            for (size_t w = 0; w < width(); ++w) {
              result(n, c, h, w) = (*this)(n, c, h, w) - other(n, c, h, w);
            }
          }
        }
      }
    } else if constexpr (dims == 5) {
      for (size_t n = 0; n < batch_size(); ++n) {
        for (size_t c = 0; c < channels(); ++c) {
          for (size_t d = 0; d < depth(); ++d) {
            for (size_t h = 0; h < height(); ++h) {
              for (size_t w = 0; w < width(); ++w) {
                result(n, c, d, h, w) =
                    (*this)(n, c, d, h, w) - other(n, c, d, h, w);
              }
            }
          }
        }
      }
    } else {
      throw std::runtime_error(
          "Unsupported tensor dimensionality for subtraction");
    }
    return result;
  }

  Tensor<T, layout> operator*(T scalar) const {
    std::vector<size_t> shape_vec(shape_, shape_ + dims);
    Tensor<T, layout> result(shape_vec);
#ifdef _OPENMP
#pragma omp parallel for
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
#pragma omp parallel for
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
        throw std::invalid_argument("Tensor shapes must match for addition");
      }
    }

    if constexpr (dims == 4) {
#ifdef _OPENMP
#pragma omp parallel for collapse(4)
#endif
      // Element-wise addition using logical indexing to handle different
      // layouts
      for (size_t n = 0; n < batch_size(); ++n) {
        for (size_t c = 0; c < channels(); ++c) {
          for (size_t h = 0; h < height(); ++h) {
            for (size_t w = 0; w < width(); ++w) {
              (*this)(n, c, h, w) += other(n, c, h, w);
            }
          }
        }
      }
    } else if constexpr (dims == 5) {
      for (size_t n = 0; n < batch_size(); ++n) {
        for (size_t c = 0; c < channels(); ++c) {
          for (size_t d = 0; d < depth(); ++d) {
            for (size_t h = 0; h < height(); ++h) {
              for (size_t w = 0; w < width(); ++w) {
                (*this)(n, c, d, h, w) += other(n, c, d, h, w);
              }
            }
          }
        }
      }
    } else {
      throw std::runtime_error(
          "Unsupported tensor dimensionality for addition");
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

    if constexpr (dims == 4) {
#ifdef _OPENMP
#pragma omp parallel for collapse(4)
#endif
      for (size_t n = 0; n < batch_size(); ++n) {
        for (size_t c = 0; c < channels(); ++c) {
          for (size_t h = 0; h < height(); ++h) {
            for (size_t w = 0; w < width(); ++w) {
              (*this)(n, c, h, w) -= other(n, c, h, w);
            }
          }
        }
      }
    } else if constexpr (dims == 5) {
      for (size_t n = 0; n < batch_size(); ++n) {
        for (size_t c = 0; c < channels(); ++c) {
          for (size_t d = 0; d < depth(); ++d) {
            for (size_t h = 0; h < height(); ++h) {
              for (size_t w = 0; w < width(); ++w) {
                (*this)(n, c, d, h, w) -= other(n, c, d, h, w);
              }
            }
          }
        }
      }
    } else {
      throw std::runtime_error(
          "Unsupported tensor dimensionality for subtraction");
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

    if constexpr (dims == 4) {
#ifdef _OPENMP
#pragma omp parallel for collapse(4)
#endif
      for (size_t n = 0; n < batch_size(); ++n) {
        for (size_t c = 0; c < channels(); ++c) {
          for (size_t h = 0; h < height(); ++h) {
            for (size_t w = 0; w < width(); ++w) {
              (*this)(n, c, h, w) *= other(n, c, h, w);
            }
          }
        }
      }
    } else if constexpr (dims == 5) {
      for (size_t n = 0; n < batch_size(); ++n) {
        for (size_t c = 0; c < channels(); ++c) {
          for (size_t d = 0; d < depth(); ++d) {
            for (size_t h = 0; h < height(); ++h) {
              for (size_t w = 0; w < width(); ++w) {
                (*this)(n, c, d, h, w) *= other(n, c, d, h, w);
              }
            }
          }
        }
      }
    } else {
      throw std::runtime_error(
          "Unsupported tensor dimensionality for element-wise multiplication");
    }
    return *this;
  }

  Tensor<T, layout> &operator*=(T scalar) {
#ifdef _OPENMP
#pragma omp parallel for
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
#pragma omp parallel for
#endif
    for (size_t i = 0; i < data_size_; ++i) {
      data_[i] /= scalar;
    }
    return *this;
  }

  // Statistical operations
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
    } else if constexpr (dims == 5) {
      size_t channel_size = batch_size() * depth() * height() * width();

      for (size_t c = 0; c < channels(); ++c) {
        T sum = T(0);
        for (size_t n = 0; n < batch_size(); ++n) {
          for (size_t d = 0; d < depth(); ++d) {
            for (size_t h = 0; h < height(); ++h) {
              for (size_t w = 0; w < width(); ++w) {
                sum += (*this)(n, c, d, h, w);
              }
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

  // To row major vector (for NCHW it's the same as data)
  std::vector<T> to_rm_vector() const {
    if constexpr (layout == NCHW) {
      return std::vector<T>(data_.get(), data_.get() + data_size_);
    } else {
      // For NHWC, we need to flatten the tensor to a row-major vector
      std::vector<T> result(data_size_);
      size_t idx = 0;

      if constexpr (dims == 4) {
        for (size_t n = 0; n < batch_size(); ++n) {
          for (size_t h = 0; h < height(); ++h) {
            for (size_t w = 0; w < width(); ++w) {
              for (size_t c = 0; c < channels(); ++c) {
                result[idx++] = (*this)(n, c, h, w);
              }
            }
          }
        }
      } else if constexpr (dims == 5) {
        for (size_t n = 0; n < batch_size(); ++n) {
          for (size_t d = 0; d < depth(); ++d) {
            for (size_t h = 0; h < height(); ++h) {
              for (size_t w = 0; w < width(); ++w) {
                for (size_t c = 0; c < channels(); ++c) {
                  result[idx++] = (*this)(n, c, d, h, w);
                }
              }
            }
          }
        }
      } else {
        throw std::runtime_error("Unsupported tensor dimensionality for "
                                 "row-major vector conversion");
      }

      return result;
    }
  }

  void from_rm_vector(const std::vector<T> &vec) {
    if (vec.size() != data_size_) {
      throw std::invalid_argument("Vector size does not match tensor size");
    }

    std::copy(vec.begin(), vec.end(), data_.get());
  }

  // Conversion to/from Matrix (for compatibility with existing layers)
  Matrix<T> to_matrix() const {
    size_t batch = batch_size();
    size_t features;

    if constexpr (dims == 4) {
      features = channels() * height() * width();
    } else if constexpr (dims == 5) {
      features = channels() * depth() * height() * width();
    } else {
      throw std::runtime_error(
          "Unsupported tensor dimensionality for matrix conversion");
    }

    Matrix<T> result(batch, features, 1);

    for (size_t n = 0; n < batch; ++n) {
      size_t feature_idx = 0;

      if constexpr (dims == 4) {
        // Layout-aware flattening for 4D
        if (layout == NCHW) {
          // NCHW: iterate channels first, then height, then width (C H W order)
          for (size_t c = 0; c < channels(); ++c) {
            for (size_t h = 0; h < height(); ++h) {
              for (size_t w = 0; w < width(); ++w) {
                result(n, feature_idx++, 0) =
                    static_cast<double>((*this)(n, c, h, w));
              }
            }
          }
        } else { // NHWC
          // NHWC: iterate height first, then width, then channels (H W C order)
          for (size_t h = 0; h < height(); ++h) {
            for (size_t w = 0; w < width(); ++w) {
              for (size_t c = 0; c < channels(); ++c) {
                result(n, feature_idx++, 0) =
                    static_cast<double>((*this)(n, c, h, w));
              }
            }
          }
        }
      } else if constexpr (dims == 5) {
        // Layout-aware flattening for 5D
        if (layout == NCDHW) {
          // NCDHW: iterate channels first, then depth, height, width (C D H W
          // order)
          for (size_t c = 0; c < channels(); ++c) {
            for (size_t d = 0; d < depth(); ++d) {
              for (size_t h = 0; h < height(); ++h) {
                for (size_t w = 0; w < width(); ++w) {
                  result(n, feature_idx++, 0) =
                      static_cast<double>((*this)(n, c, d, h, w));
                }
              }
            }
          }
        } else { // NDHWC
          // NDHWC: iterate depth, height, width, then channels (D H W C order)
          for (size_t d = 0; d < depth(); ++d) {
            for (size_t h = 0; h < height(); ++h) {
              for (size_t w = 0; w < width(); ++w) {
                for (size_t c = 0; c < channels(); ++c) {
                  result(n, feature_idx++, 0) =
                      static_cast<double>((*this)(n, c, d, h, w));
                }
              }
            }
          }
        }
      }
    }

    return result;
  }

  static Tensor<T, layout> from_matrix(const Matrix<T> &matrix, size_t channels,
                                       size_t height, size_t width) {
    size_t batch = matrix.rows;
    size_t expected_features = channels * height * width;

    if ((size_t)matrix.cols != expected_features || matrix.channels != 1) {
      throw std::invalid_argument(
          "Matrix dimensions incompatible with 4D tensor shape");
    }

    if (layout == NCDHW || layout == NDHWC) {
      throw std::invalid_argument(
          "5D layout specified for 4D tensor reconstruction");
    }

    Tensor<T, layout> result(batch, channels, height, width);

    for (size_t n = 0; n < batch; ++n) {
      size_t feature_idx = 0;

      // Layout-aware reconstruction
      if (layout == NCHW) {
        // NCHW: iterate channels first, then height, then width (C H W order)
        for (size_t c = 0; c < channels; ++c) {
          for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
              result(n, c, h, w) = matrix(n, feature_idx++, 0);
            }
          }
        }
      } else if (layout == NHWC) { // NHWC
        // NHWC: iterate height first, then width, then channels (H W C order)
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            for (size_t c = 0; c < channels; ++c) {
              result(n, c, h, w) = matrix(n, feature_idx++, 0);
            }
          }
        }
      } else {
        throw std::runtime_error(
            "Unsupported layout for conversion from Matrix");
      }
    }

    return result;
  }

  // 5D tensor reconstruction from matrix
  static Tensor<T, layout> from_matrix_5d(const Matrix<T> &matrix,
                                          size_t channels, size_t depth,
                                          size_t height, size_t width) {
    size_t batch = matrix.rows;
    size_t expected_features = channels * depth * height * width;

    if ((size_t)matrix.cols != expected_features || matrix.channels != 1) {
      throw std::invalid_argument(
          "Matrix dimensions incompatible with 5D tensor shape");
    }

    if (layout == NCHW || layout == NHWC) {
      throw std::invalid_argument(
          "4D layout specified for 5D tensor reconstruction");
    }

    Tensor<T, layout> result(batch, channels, depth, height, width);

    for (size_t n = 0; n < batch; ++n) {
      size_t feature_idx = 0;

      // Layout-aware reconstruction
      if (layout == NCDHW) {
        // NCDHW: iterate channels, depth, height, width (C D H W order)
        for (size_t c = 0; c < channels; ++c) {
          for (size_t d = 0; d < depth; ++d) {
            for (size_t h = 0; h < height; ++h) {
              for (size_t w = 0; w < width; ++w) {
                result(n, c, d, h, w) =
                    static_cast<T>(matrix(n, feature_idx++, 0));
              }
            }
          }
        }
      } else if (layout == NDHWC) { // NDHWC
        // NDHWC: iterate depth, height, width, channels (D H W C order)
        for (size_t d = 0; d < depth; ++d) {
          for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
              for (size_t c = 0; c < channels; ++c) {
                result(n, c, d, h, w) =
                    static_cast<T>(matrix(n, feature_idx++, 0));
              }
            }
          }
        }
      } else {
        throw std::runtime_error(
            "Unsupported layout for 5D conversion from Matrix");
      }
    }

    return result;
  }

  // CNN-specific operations (to be implemented with convolution layers)

  // Optimized im2col operation for efficient convolution
  Matrix<T> im2col(size_t kernel_h, size_t kernel_w, size_t stride_h = 1,
                   size_t stride_w = 1, size_t pad_h = 0,
                   size_t pad_w = 0) const {
    size_t output_h = (height() + 2 * pad_h - kernel_h) / stride_h + 1;
    size_t output_w = (width() + 2 * pad_w - kernel_w) / stride_w + 1;

    size_t col_height = channels() * kernel_h * kernel_w;
    size_t col_width = batch_size() * output_h * output_w;

    Matrix<T> col_matrix(col_height, col_width, 1);
    col_matrix.fill(0.0);

    // Apply padding if needed - avoid copy when no padding
    const Tensor<T, layout> *input_ptr = this;
    std::unique_ptr<Tensor<T, layout>> padded_input_storage;

    if (pad_h > 0 || pad_w > 0) {
      padded_input_storage =
          std::make_unique<Tensor<T, layout>>(pad(pad_h, pad_w));
      input_ptr = padded_input_storage.get();
    }

    const Tensor<T, layout> &input_tensor = *input_ptr;

    // Optimize memory access patterns - vectorize inner loops when possible
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t n = 0; n < batch_size(); ++n) {
      size_t batch_offset = n * output_h * output_w;

      for (size_t out_h = 0; out_h < output_h; ++out_h) {
        for (size_t out_w = 0; out_w < output_w; ++out_w) {
          size_t col_idx = batch_offset + out_h * output_w + out_w;
          size_t row_idx = 0;

          // Compute starting positions
          size_t h_start = out_h * stride_h;
          size_t w_start = out_w * stride_w;

          for (size_t c = 0; c < channels(); ++c) {
            // Vectorizable inner loops for small kernels
            for (size_t kh = 0; kh < kernel_h; ++kh) {
              size_t h_idx = h_start + kh;

              if (h_idx < input_tensor.height()) {
                for (size_t kw = 0; kw < kernel_w; ++kw) {
                  size_t w_idx = w_start + kw;

                  if (w_idx < input_tensor.width()) {
                    col_matrix(row_idx, col_idx, 0) =
                        input_tensor(n, c, h_idx, w_idx);
                  }
                  ++row_idx;
                }
              } else {
                // Skip entire row if height is out of bounds
                row_idx += kernel_w;
              }
            }
          }
        }
      }
    }

    return col_matrix;
  }

  /**
   * col2im operation (inverse of im2col) for gradient computation
   * Reconstructs the original tensor from the column matrix.
   */
  static Tensor<T, layout> col2im(const Matrix<T> &col_matrix,
                                  size_t batch_size, size_t channels,
                                  size_t height, size_t width, size_t kernel_h,
                                  size_t kernel_w, size_t stride_h = 1,
                                  size_t stride_w = 1, size_t pad_h = 0,
                                  size_t pad_w = 0) {
    size_t output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    size_t output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    // Create padded result tensor
    size_t padded_h = height + 2 * pad_h;
    size_t padded_w = width + 2 * pad_w;
    Tensor<T, layout> padded_result(batch_size, channels, padded_h, padded_w);
    padded_result.fill(T(0));

    // Reconstruct from column matrix
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      size_t batch_offset = n * output_h * output_w;

      for (size_t out_h = 0; out_h < output_h; ++out_h) {
        for (size_t out_w = 0; out_w < output_w; ++out_w) {
          size_t col_idx = batch_offset + out_h * output_w + out_w;
          size_t row_idx = 0;

          size_t h_start = out_h * stride_h;
          size_t w_start = out_w * stride_w;

          for (size_t c = 0; c < channels; ++c) {
            for (size_t kh = 0; kh < kernel_h; ++kh) {
              size_t h_idx = h_start + kh;

              for (size_t kw = 0; kw < kernel_w; ++kw) {
                size_t w_idx = w_start + kw;

                if (h_idx < padded_h && w_idx < padded_w) {
#ifdef _OPENMP
#pragma omp atomic
#endif
                  padded_result(n, c, h_idx, w_idx) +=
                      col_matrix(row_idx, col_idx, 0);
                }
                ++row_idx;
              }
            }
          }
        }
      }
    }

    // Remove padding if it was applied
    if (pad_h > 0 || pad_w > 0) {
      return padded_result.crop(pad_h, pad_w, padded_h - pad_h - 1,
                                padded_w - pad_w - 1);
    } else {
      return padded_result;
    }
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
    } else if constexpr (dims == 5) {
      for (size_t n = 0; n < shape_[0]; ++n) {
        for (size_t c = 0; c < shape_[1]; ++c) {
          for (size_t d = 0; d < shape_[2]; ++d) {
            for (size_t h = 0; h < shape_[3]; ++h) {
              for (size_t w = 0; w < shape_[4]; ++w) {
                result(n, c, d, h, w) = (*this)(n, c, d, h, w);
              }
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

  // Print tensor values (for debugging small tensors)
  void print(size_t max_elements = 100) const {
    std::cout << "Tensor data:" << std::endl;

    if (size() <= max_elements) {
      if constexpr (dims == 4) {
        for (size_t n = 0; n < batch_size(); ++n) {
          std::cout << "Batch " << n << ":" << std::endl;
          for (size_t c = 0; c < channels(); ++c) {
            std::cout << "  Channel " << c << ":" << std::endl;
            for (size_t h = 0; h < height(); ++h) {
              std::cout << "    ";
              for (size_t w = 0; w < width(); ++w) {
                std::cout << std::fixed << std::setprecision(4)
                          << (*this)(n, c, h, w) << " ";
              }
              std::cout << std::endl;
            }
            if (c < channels() - 1)
              std::cout << std::endl;
          }
          if (n < batch_size() - 1)
            std::cout << std::endl;
        }
      } else if constexpr (dims == 5) {
        for (size_t n = 0; n < batch_size(); ++n) {
          std::cout << "Batch " << n << ":" << std::endl;
          for (size_t c = 0; c < channels(); ++c) {
            std::cout << "  Channel " << c << ":" << std::endl;
            for (size_t d = 0; d < depth(); ++d) {
              std::cout << "    Depth " << d << ":" << std::endl;
              for (size_t h = 0; h < height(); ++h) {
                std::cout << "      ";
                for (size_t w = 0; w < width(); ++w) {
                  std::cout << std::fixed << std::setprecision(4)
                            << (*this)(n, c, d, h, w) << " ";
                }
                std::cout << std::endl;
              }
              if (d < depth() - 1)
                std::cout << std::endl;
            }
            if (c < channels() - 1)
              std::cout << std::endl;
          }
          if (n < batch_size() - 1)
            std::cout << std::endl;
        }
      }
    } else {
      std::cout << "Tensor too large to print (size: " << size()
                << "), showing first 10 elements:" << std::endl;
      for (size_t i = 0; i < std::min(static_cast<size_t>(10), data_size_);
           ++i) {
        std::cout << data_[i] << " ";
      }
      std::cout << "..." << std::endl;
    }
  }

    // Serialization
  void save(std::ofstream &file) const {
    if (!file.is_open()) {
      throw std::runtime_error("File is not open for writing");
    }
    // Write shape
    file.write(reinterpret_cast<const char *>(shape_), dims * sizeof(size_t));
    // Write data
    file.write(reinterpret_cast<const char *>(data_.get()),
               data_size_ * sizeof(T));
  }

  static Tensor<T, layout> load(std::ifstream &file) {
    if (!file.is_open()) {
      throw std::runtime_error("File is not open for reading");
    }
    std::vector<size_t> shape(dims);
    file.read(reinterpret_cast<char *>(shape.data()), dims * sizeof(size_t));
    if (file.gcount() != dims * sizeof(size_t)) {
      throw std::runtime_error("Failed to read tensor shape from file");
    }

    Tensor<T, layout> tensor(shape);
    file.read(reinterpret_cast<char *>(tensor.data()),
              tensor.size() * sizeof(T));
    if (file.gcount() != tensor.size() * sizeof(T)) {
      throw std::runtime_error("Failed to read tensor data from file");
    }
    return tensor;
  }
};

// Convenience functions for creating tensors
namespace Tensors {

// Create 4D tensors with specific initialization (float version)
template <typename T = float, Layout layout = NCHW>
inline Tensor<T, layout> zeros(size_t batch, size_t channels, size_t height,
                               size_t width) {
  return Tensor<T, layout>(batch, channels, height, width);
}

template <typename T = float, Layout layout = NCHW>
inline Tensor<T, layout> ones(size_t batch, size_t channels, size_t height,
                              size_t width) {
  Tensor<T, layout> t(batch, channels, height, width);
  t.fill(T(1));
  return t;
}

template <typename T = float, Layout layout = NCHW>
inline Tensor<T, layout> random_normal(size_t batch, size_t channels,
                                       size_t height, size_t width,
                                       T stddev = T(1)) {
  static_assert(std::is_floating_point<T>::value,
                "Normal distribution requires floating point type");
  Tensor<T, layout> t(batch, channels, height, width);
  t.fill_random_normal(stddev);
  return t;
}

template <typename T = float, Layout layout = NCHW>
inline Tensor<T, layout> random_uniform(size_t batch, size_t channels,
                                        size_t height, size_t width,
                                        T range = T(1)) {
  Tensor<T, layout> t(batch, channels, height, width);
  t.fill_random_uniform(range);
  return t;
}

// Create 5D tensors with specific initialization
template <typename T = float, Layout layout = NCHW>
inline Tensor<T, layout> zeros_5d(size_t batch, size_t channels, size_t depth,
                                  size_t height, size_t width) {
  return Tensor<T, layout>(batch, channels, depth, height, width);
}

template <typename T = float, Layout layout = NCHW>
inline Tensor<T, layout> ones_5d(size_t batch, size_t channels, size_t depth,
                                 size_t height, size_t width) {
  Tensor<T, layout> t(batch, channels, depth, height, width);
  t.fill(T(1));
  return t;
}

template <typename T = float, Layout layout = NCHW>
inline Tensor<T, layout> random_normal_5d(size_t batch, size_t channels,
                                          size_t depth, size_t height,
                                          size_t width, T stddev = T(1)) {
  static_assert(std::is_floating_point<T>::value,
                "Normal distribution requires floating point type");
  Tensor<T, layout> t(batch, channels, depth, height, width);
  t.fill_random_normal(stddev);
  return t;
}

template <typename T = float, Layout layout = NCHW>
inline Tensor<T, layout> random_uniform_5d(size_t batch, size_t channels,
                                           size_t depth, size_t height,
                                           size_t width, T range = T(1)) {
  Tensor<T, layout> t(batch, channels, depth, height, width);
  t.fill_random_uniform(range);
  return t;
}
} // namespace Tensors
