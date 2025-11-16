#pragma once

#include "dkernels.hpp"
#include "skernels.hpp"
#include "threading/thread_handler.hpp"
#include <cmath>
#include <cstddef>
#include <random>
#include <type_traits>

namespace tnn {
namespace ops {
namespace cpu {

// These declarations provide a clean interface T, where T is numeric type (e.g float, double).

template <typename T> void add(const T *a, const T *b, T *c, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    fp::add(a, b, c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::add(a, b, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = a[i] + b[i];
    }
  }
}

template <typename T> void sub(const T *a, const T *b, T *c, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    fp::sub(a, b, c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::sub(a, b, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = a[i] - b[i];
    }
  }
}

template <typename T> void mul(const T *a, const T *b, T *c, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    fp::mul(a, b, c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::mul(a, b, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = a[i] * b[i];
    }
  }
}

template <typename T> void div(const T *a, const T *b, T *c, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    fp::div(a, b, c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::div(a, b, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = a[i] / b[i];
    }
  }
}

template <typename T> void fmadd(const T *a, const T *b, T *c, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    fp::fmadd(a, b, c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::fmadd(a, b, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = std::fma(a[i], b[i], c[i]);
    }
  }
}

template <typename T> void fmsub(const T *a, const T *b, T *c, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    fp::fmsub(a, b, c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::fmsub(a, b, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = (a[i] * b[i]) - c[i];
    }
  }
}

template <typename T> void fnmadd(const T *a, const T *b, T *c, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    fp::fnmadd(a, b, c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::fnmadd(a, b, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = -(a[i] * b[i]) + c[i];
    }
  }
}

// Scalar Operations
template <typename T> void add_scalar(const T *a, T scalar, T *c, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    fp::add_scalar(a, scalar, c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::add_scalar(a, scalar, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = a[i] + scalar;
    }
  }
}

template <typename T> void mul_scalar(const T *a, T scalar, T *c, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    fp::mul_scalar(a, scalar, c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::mul_scalar(a, scalar, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = a[i] * scalar;
    }
  }
}

template <typename T> void div_scalar(const T *a, T scalar, T *c, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    fp::div_scalar(a, scalar, c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::div_scalar(a, scalar, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = a[i] / scalar;
    }
  }
}

template <typename T> void set_scalar(T *c, T scalar, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    fp::set_scalar(c, scalar, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::set_scalar(c, scalar, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = scalar;
    }
  }
}

// Element-wise Functions
template <typename T> void sqrt(const T *a, T *c, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    fp::sqrt(a, c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::sqrt(a, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = std::sqrt(a[i]);
    }
  }
}

template <typename T> void abs(const T *a, T *c, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    fp::abs(a, c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::abs(a, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = std::abs(a[i]);
    }
  }
}

template <typename T> void rsqrt(const T *a, T *c, size_t size) {
  // c[i] = 1.0 / sqrt(a[i])
  if constexpr (std::is_same_v<T, float>) {
    fp::rsqrt(a, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = 1.0 / std::sqrt(a[i]);
    }
  }
}

template <typename T> void rcp(const T *a, T *c, size_t size) {
  // c[i] = 1.0 / a[i]
  if constexpr (std::is_same_v<T, float>) {
    fp::rcp(a, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = 1.0 / a[i];
    }
  }
}

// Comparison and Clamping
template <typename T> void min(const T *a, const T *b, T *c, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    fp::min(a, b, c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::min(a, b, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = a[i] < b[i] ? a[i] : b[i]; // equivalent to std::min
    }
  }
}

template <typename T> void max(const T *a, const T *b, T *c, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    fp::max(a, b, c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::max(a, b, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = a[i] > b[i] ? a[i] : b[i]; // equivalent to std::max
    }
  }
}

template <typename T> void scalar_max(const T *a, T scalar, T *c, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    fp::scalar_max(a, scalar, c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::scalar_max(a, scalar, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = a[i] > scalar ? a[i] : scalar;
    }
  }
}

template <typename T> void clamp(const T *a, T min_val, T max_val, T *c, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    fp::clamp(a, min_val, max_val, c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::clamp(a, min_val, max_val, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = a[i] < min_val ? min_val : (a[i] > max_val ? max_val : a[i]);
    }
  }
}

template <typename T> void equal(const T *a, const T *b, T *c, size_t size) {
  // c[i] = (a[i] == b[i]) ? 1 : 0
  if constexpr (std::is_same_v<T, float>) {
    fp::equal(a, b, c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::equal(a, b, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = (a[i] == b[i]) ? static_cast<T>(1.0) : static_cast<T>(0.0);
    }
  }
}

template <typename T> void greater(const T *a, const T *b, T *c, size_t size) {
  // c[i] = (a[i] > b[i]) ? 1 : 0
  if constexpr (std::is_same_v<T, float>) {
    fp::greater(a, b, c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::greater(a, b, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = (a[i] > b[i]) ? static_cast<T>(1.0) : static_cast<T>(0.0);
    }
  }
}

// Memory Operations
template <typename T> void copy(const T *a, T *c, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    fp::copy(a, c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::copy(a, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = a[i];
    }
  }
}

template <typename T> void zero(T *c, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    fp::zero(c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::zero(c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = static_cast<T>(0.0);
    }
  }
}

// Specialized BatchNorm Operations
template <typename T>
void sub_mul_scalar(const T *a, T sub_scalar, T mul_scalar, T *c, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    fp::sub_mul_scalar(a, sub_scalar, mul_scalar, c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::sub_mul_scalar(a, sub_scalar, mul_scalar, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = (a[i] - sub_scalar) * mul_scalar;
    }
  }
}

template <typename T>
void mul_add_scalar(const T *a, T mul_scalar, T add_scalar, T *c, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    fp::mul_add_scalar(a, mul_scalar, add_scalar, c, size);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::mul_add_scalar(a, mul_scalar, add_scalar, c, size);
  } else {
    for (size_t i = 0; i < size; ++i) {
      c[i] = (a[i] * mul_scalar) + add_scalar;
    }
  }
}

// Reduction Functions
template <typename T> T sum(const T *a, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    return fp::sum(a, size);
  } else if constexpr (std::is_same_v<T, double>) {
    return dp::sum(a, size);
  } else {
    T result = static_cast<T>(0.0);
    for (size_t i = 0; i < size; ++i) {
      result += a[i];
    }
    return result;
  }
}

template <typename T> T dot_product(const T *a, const T *b, size_t size) {
  if constexpr (std::is_same_v<T, float>) {
    return fp::dot_product(a, b, size);
  } else if constexpr (std::is_same_v<T, double>) {
    return dp::dot_product(a, b, size);
  } else {
    T result = static_cast<T>(0.0);
    for (size_t i = 0; i < size; ++i) {
      result += a[i] * b[i];
    }
    return result;
  }
}

template <typename T> T sum_squared_diff(const T *a, T mean, size_t size) {
  // sum((a[i] - mean)^2)
  if constexpr (std::is_same_v<T, float>) {
    return fp::sum_squared_diff(a, mean, size);
  } else if constexpr (std::is_same_v<T, double>) {
    return dp::sum_squared_diff(a, mean, size);
  } else {
    T result = static_cast<T>(0.0);
    for (size_t i = 0; i < size; ++i) {
      T diff = a[i] - mean;
      result += diff * diff;
    }
    return result;
  }
}

template <typename T> T norm_squared(const T *a, size_t size) {
  // sum(a[i]^2)
  T result = static_cast<T>(0.0);
  for (size_t i = 0; i < size; ++i) {
    result += a[i] * a[i];
  }
  return result;
}

template <typename T>
void fill_random_uniform(T *data, size_t size, T min_val, T max_val, unsigned long long seed) {
  if constexpr (std::is_same_v<T, float>) {
    fp::fill_random_uniform(data, size, min_val, max_val, seed);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::fill_random_uniform(data, size, min_val, max_val, seed);
  } else {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> dist(min_val, max_val);
    for (size_t i = 0; i < size; ++i) {
      data[i] = dist(rng);
    }
  }
}

template <typename T>
void fill_random_normal(T *data, size_t size, T mean, T stddev, unsigned long long seed) {
  if constexpr (std::is_same_v<T, float>) {
    fp::fill_random_normal(data, size, mean, stddev, seed);
  } else if constexpr (std::is_same_v<T, double>) {
    dp::fill_random_normal(data, size, mean, stddev, seed);
  } else {
    std::mt19937_64 rng(seed);
    std::normal_distribution<T> dist(mean, stddev);
    for (size_t i = 0; i < size; ++i) {
      data[i] = dist(rng);
    }
  }
}

template <typename T>
void transpose_2d(const T *src, T *dst, const size_t rows, const size_t cols,
                  bool multi_threaded = true) {
  if (multi_threaded) {
    const size_t block_size = 64;
    parallel_for_2d((rows + block_size - 1) / block_size, (cols + block_size - 1) / block_size,
                    [&](size_t i_block, size_t j_block) {
                      const size_t start_row = i_block * block_size;
                      const size_t start_col = j_block * block_size;
                      const size_t end_row = std::min(start_row + block_size, rows);
                      const size_t end_col = std::min(start_col + block_size, cols);
                      for (size_t i = start_row; i < end_row; ++i) {
                        for (size_t j = start_col; j < end_col; ++j) {
                          dst[j * rows + i] = src[i * cols + j];
                        }
                      }
                    });
    return;
  } else {
    const size_t block_size = 64;
    for (size_t i_block = 0; i_block < (rows + block_size - 1) / block_size; ++i_block) {
      for (size_t j_block = 0; j_block < (cols + block_size - 1) / block_size; ++j_block) {
        const size_t start_row = i_block * block_size;
        const size_t start_col = j_block * block_size;
        const size_t end_row = std::min(start_row + block_size, rows);
        const size_t end_col = std::min(start_col + block_size, cols);
        for (size_t i = start_row; i < end_row; ++i) {
          for (size_t j = start_col; j < end_col; ++j) {
            dst[j * rows + i] = src[i * cols + j];
          }
        }
      }
    }
  }
}

template <typename T>
void nchw_to_cnhw(const T *src, T *dst, size_t batch_size, size_t channels, size_t height,
                  size_t width) {
  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    std::copy(&src[n * channels * height * width + c * height * width],
              &src[n * channels * height * width + c * height * width + height * width],
              &dst[c * batch_size * height * width + n * height * width]);
  });
}

template <typename T>
void cnhw_to_nchw(const T *src, T *dst, size_t batch_size, size_t channels, size_t height,
                  size_t width) {
  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    std::copy(&src[c * batch_size * height * width + n * height * width],
              &src[c * batch_size * height * width + n * height * width + height * width],
              &dst[n * channels * height * width + c * height * width]);
  });
}

} // namespace cpu
} // namespace ops
} // namespace tnn