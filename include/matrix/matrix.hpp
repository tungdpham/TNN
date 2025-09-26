/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#pragma once

#include <cstring>
#include <iostream>
#include <random>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <algorithm>
#include <cstdlib>
#include <new>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#endif

template <typename T = float> struct Matrix {
private:
  size_t rows_, cols_;
  T *data_;

  static constexpr size_t AVX2_ALIGNMENT = 32;

  T *allocate_aligned(size_t count) {
    if (count == 0)
      return nullptr;

    size_t bytes = count * sizeof(T);

#ifdef _WIN32
    void *ptr = _aligned_malloc(bytes, AVX2_ALIGNMENT);
#else
    void *ptr = std::aligned_alloc(
        AVX2_ALIGNMENT,
        ((bytes + AVX2_ALIGNMENT - 1) / AVX2_ALIGNMENT) * AVX2_ALIGNMENT);
#endif
    if (!ptr) {
      throw std::bad_alloc();
    }

    return static_cast<T *>(ptr);
  }

  void deallocate_aligned(T *ptr) {
    if (ptr) {
#ifdef _WIN32
      _aligned_free(ptr);
#else
      free(ptr);
#endif
    }
  }

  inline void avx2_add(const T *a, const T *b, T *result, size_t size) const {
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
      const size_t avx_size = 8;

      size_t i = 0;

      for (; i + avx_size <= size; i += avx_size) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vb = _mm256_load_ps(b + i);
        __m256 vresult = _mm256_add_ps(va, vb);
        _mm256_store_ps(result + i, vresult);
      }

      for (; i < size; ++i) {
        result[i] = a[i] + b[i];
      }
    } else {

      for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
      }
    }
#else
    std::copy(a, a + size, result);
    for (size_t i = 0; i < size; ++i) {
      result[i] += b[i];
    }
#endif
  }

  inline void avx2_sub(const T *a, const T *b, T *result, size_t size) const {
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
      const size_t avx_size = 8;

      size_t i = 0;

      for (; i + avx_size <= size; i += avx_size) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vb = _mm256_load_ps(b + i);
        __m256 vresult = _mm256_sub_ps(va, vb);
        _mm256_store_ps(result + i, vresult);
      }

      for (; i < size; ++i) {
        result[i] = a[i] - b[i];
      }
    } else {

      for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] - b[i];
      }
    }
#else
    std::copy(a, a + size, result);
    for (size_t i = 0; i < size; ++i) {
      result[i] -= b[i];
    }
#endif
  }

  inline void avx2_mul_scalar(const T *a, T scalar, T *result,
                              size_t size) const {
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
      const size_t avx_size = 8;
      const __m256 avx_scalar = _mm256_set1_ps(scalar);

      size_t i = 0;

      for (; i + avx_size <= size; i += avx_size) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vresult = _mm256_mul_ps(va, avx_scalar);
        _mm256_store_ps(result + i, vresult);
      }

      for (; i < size; ++i) {
        result[i] = a[i] * scalar;
      }
    } else {

      for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] * scalar;
      }
    }
#else
    std::copy(a, a + size, result);
    for (size_t i = 0; i < size; ++i) {
      result[i] *= scalar;
    }
#endif
  }

  inline void avx2_div_scalar(const T *a, T scalar, T *result,
                              size_t size) const {
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
      const size_t avx_size = 8;
      const __m256 avx_scalar = _mm256_set1_ps(scalar);

      size_t i = 0;

      for (; i + avx_size <= size; i += avx_size) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vresult = _mm256_div_ps(va, avx_scalar);
        _mm256_store_ps(result + i, vresult);
      }

      for (; i < size; ++i) {
        result[i] = a[i] / scalar;
      }
    } else {

      for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] / scalar;
      }
    }
#else
    std::copy(a, a + size, result);
    for (size_t i = 0; i < size; ++i) {
      result[i] /= scalar;
    }
#endif
  }

  inline void avx2_matmul(const Matrix &a, const Matrix &b,
                          Matrix &result) const {
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
      const size_t avx_size = 8;

      const size_t tile_size = 64;

      for (size_t ii = 0; ii < a.rows_; ii += tile_size) {
        for (size_t jj = 0; jj < b.cols_; jj += tile_size) {
          for (size_t kk = 0; kk < a.cols_; kk += tile_size) {

            size_t i_end = std::min(ii + tile_size, a.rows_);
            size_t j_end = std::min(jj + tile_size, b.cols_);
            size_t k_end = std::min(kk + tile_size, a.cols_);

            for (size_t i = ii; i < i_end; ++i) {
              for (size_t j = jj; j + avx_size <= j_end; j += avx_size) {
                __m256 sum = _mm256_load_ps(&result.data_[i * b.cols_ + j]);

                for (size_t k = kk; k < k_end; ++k) {
                  __m256 a_val = _mm256_set1_ps(a.data_[i * a.cols_ + k]);
                  __m256 b_val = _mm256_load_ps(&b.data_[k * b.cols_ + j]);
                  sum = _mm256_fmadd_ps(a_val, b_val, sum);
                }

                _mm256_store_ps(&result.data_[i * b.cols_ + j], sum);
              }

              for (size_t j = (j_end / avx_size) * avx_size; j < j_end; ++j) {
                T sum = result.data_[i * b.cols_ + j];
                for (size_t k = kk; k < k_end; ++k) {
                  sum += a.data_[i * a.cols_ + k] * b.data_[k * b.cols_ + j];
                }
                result.data_[i * b.cols_ + j] = sum;
              }
            }
          }
        }
      }
    } else {

      for (size_t i = 0; i < a.rows_; ++i) {
        for (size_t j = 0; j < b.cols_; ++j) {
          T sum = 0;
          for (size_t k = 0; k < a.cols_; ++k) {
            sum += a.data_[i * a.cols_ + k] * b.data_[k * b.cols_ + j];
          }
          result.data_[i * b.cols_ + j] = sum;
        }
      }
    }
#else

    for (size_t i = 0; i < a.rows_; ++i) {
      for (size_t j = 0; j < b.cols_; ++j) {
        T sum = 0;
        for (size_t k = 0; k < a.cols_; ++k) {
          sum += a.data_[i * a.cols_ + k] * b.data_[k * b.cols_ + j];
        }
        result.data_[i * b.cols_ + j] = sum;
      }
    }
#endif
  }

public:
  Matrix() : rows_(0), cols_(0), data_(nullptr) {}

  Matrix(size_t rows_, size_t cols_, T *initialdata_ = nullptr)
      : rows_(rows_), cols_(cols_) {
    data_ = allocate_aligned(rows_ * cols_);
    if (initialdata_ != nullptr) {
      memcpy(data_, initialdata_, rows_ * cols_ * sizeof(T));
    } else {
    }
  }

  Matrix(const Matrix &other) {
    if (this->size() != other.size()) {
      deallocate_aligned(this->data_);
    }
    this->rows_ = other.rows_;
    this->cols_ = other.cols_;
    memcpy(data_, other.data_, rows_ * cols_ * sizeof(T));
  }

  Matrix(Matrix &&other) noexcept
      : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {

    other.rows_ = 0;
    other.cols_ = 0;
    other.data_ = nullptr;
  }

  Matrix &operator=(Matrix &&other) noexcept {
    if (this != &other) {
      deallocate_aligned(data_);

      rows_ = other.rows_;
      cols_ = other.cols_;
      data_ = other.data_;

      other.rows_ = 0;
      other.cols_ = 0;
      other.data_ = nullptr;
    }
    return *this;
  }

  ~Matrix() { deallocate_aligned(data_); }

  inline T &operator()(size_t row, size_t col) {
    return data_[row * cols_ + col];
  }

  inline const T &operator()(size_t row, size_t col) const {
    return data_[row * cols_ + col];
  }

  T *data() { return data_; }

  const T *data() const { return data_; }

  void fill(T value) {
    std::fill(data_, data_ + rows_ * cols_, value);
  }

  void print() const {
    for (size_t r = 0; r < rows_; ++r) {
      for (size_t c = 0; c < cols_; ++c) {
        std::cout << (*this)(r, c) << " ";
      }
      std::cout << std::endl;
    }
  }

  inline Matrix operator+(const Matrix &other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("Matrix dimensions must match for addition.");
    }
    Matrix result(rows_, cols_);
    size_t size = rows_ * cols_;

    if (size > 32) {
      avx2_add(data_, other.data_, result.data_, size);
    } else {

      for (size_t i = 0; i < size; ++i) {
        result.data_[i] = data_[i] + other.data_[i];
      }
    }
    return result;
  }

  inline Matrix operator+=(const Matrix &other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("Matrix dimensions must match for addition.");
    }
    size_t size = rows_ * cols_;

    if (size > 32) {
      avx2_add(data_, other.data_, data_, size);
    } else {

      for (size_t i = 0; i < size; ++i) {
        data_[i] += other.data_[i];
      }
    }
    return *this;
  }

  inline Matrix operator-(const Matrix &other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument(
          "Matrix dimensions must match for subtraction.");
    }
    Matrix result(rows_, cols_);
    size_t size = rows_ * cols_;

    if (size > 32) {
      avx2_sub(data_, other.data_, result.data_, size);
    } else {

      for (size_t i = 0; i < size; ++i) {
        result.data_[i] = data_[i] - other.data_[i];
      }
    }
    return result;
  }

  inline Matrix operator-=(const Matrix &other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument(
          "Matrix dimensions must match for subtraction.");
    }
    size_t size = rows_ * cols_;

    if (size > 32) {
      avx2_sub(data_, other.data_, data_, size);
    } else {

      for (size_t i = 0; i < size; ++i) {
        data_[i] -= other.data_[i];
      }
    }
    return *this;
  }

  inline Matrix operator*(T scalar) const {
    Matrix result(rows_, cols_);
    size_t size = rows_ * cols_;

    if (size > 32) {
      avx2_mul_scalar(data_, scalar, result.data_, size);
    } else {
      for (size_t i = 0; i < size; ++i) {
        result.data_[i] = data_[i] * scalar;
      }
    }
    return result;
  }

  inline Matrix operator*=(T scalar) {
    size_t size = rows_ * cols_;

    if (size > 32) {
      avx2_mul_scalar(data_, scalar, data_, size);
    } else {

      for (size_t i = 0; i < size; ++i) {
        data_[i] *= scalar;
      }
    }
    return *this;
  }

  inline Matrix operator/(T scalar) const {
    if (scalar == 0) {
      throw std::invalid_argument("Division by zero.");
    }
    Matrix result(rows_, cols_);
    size_t size = rows_ * cols_;

    if (size > 32) {
      avx2_div_scalar(data_, scalar, result.data_, size);
    } else {

      for (size_t i = 0; i < size; ++i) {
        result.data_[i] = data_[i] / scalar;
      }
    }
    return result;
  }

  inline Matrix operator/=(T scalar) {
    if (scalar == 0) {
      throw std::invalid_argument("Division by zero.");
    }
    size_t size = rows_ * cols_;

    if (size > 32) {
      avx2_div_scalar(data_, scalar, data_, size);
    } else {

      for (size_t i = 0; i < size; ++i) {
        data_[i] /= scalar;
      }
    }
    return *this;
  }

  inline Matrix operator*(const Matrix &other) const {
    if (cols_ != other.rows_) {
      throw std::invalid_argument(
          "Matrix dimensions must match for multiplication.");
    }
    Matrix result(rows_, other.cols_);
    result.fill(0.0);

    if constexpr (std::is_same_v<T, float>) {
      if (rows_ * other.cols_ > 1024) {
        avx2_matmul(*this, other, result);
        return result;
      }
    }

    size_t other_cols = other.cols();

#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (size_t r = 0; r < rows_; ++r) {
      for (size_t c = 0; c < other_cols; ++c) {
        T sum = 0.0;
        for (size_t k = 0; k < cols_; ++k) {
          sum += (*this)(r, k) * other(k, c);
        }
        result(r, c) = sum;
      }
    }
    return result;
  }

  inline Matrix &operator=(const Matrix &other) = delete;

  Matrix<T> clone() const { return Matrix(rows_, cols_, data_); }

  Matrix transpose() const {
    Matrix result(cols_, rows_);

    T *result_data = result.data();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t r = 0; r < rows_; ++r) {
      for (size_t c = 0; c < cols_; ++c) {
        result_data[c * rows_ + r] = this->data_[r * cols_ + c];
      }
    }
    return result;
  }

  Matrix reshape(size_t newrows_, size_t newcols_) const {
    if (rows_ * cols_ != newrows_ * newcols_) {
      throw std::invalid_argument(
          "Total number of elements must remain the same for reshape.");
    }
    return Matrix(newrows_, newcols_, data_);
  }

  Matrix pad(size_t padrows_, size_t padcols_, T value = 0.0) const {
    Matrix result(rows_ + 2 * padrows_, cols_ + 2 * padcols_);
    result.fill(value);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t r = 0; r < rows_; ++r) {
      for (size_t c = 0; c < cols_; ++c) {
        result(r + padrows_, c + padcols_) = (*this)(r, c);
      }
    }
    return result;
  }

  Matrix crop(size_t startRow, size_t startCol, size_t endRow,
              size_t endCol) const {
    if (startRow < 0 || startCol < 0 || endRow >= rows_ || endCol >= cols_ ||
        startRow >= endRow || startCol >= endCol) {
      throw std::invalid_argument("Invalid crop dimensions.");
    }
    Matrix result(endRow - startRow + 1, endCol - startCol + 1);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t r = startRow; r <= endRow; ++r) {
      for (size_t c = startCol; c <= endCol; ++c) {
        result(r - startRow, c - startCol) = (*this)(r, c);
      }
    }
    return result;
  }

  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }

  T mean() const {
    size_t size = rows_ * cols_;

#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
      if (size > 32) {
        const size_t avx_size = 8;
        __m256 sum_vec = _mm256_setzero_ps();

        size_t i = 0;

        for (; i + avx_size <= size; i += avx_size) {
          __m256 data_vec = _mm256_load_ps(data_ + i);
          sum_vec = _mm256_add_ps(sum_vec, data_vec);
        }

        float sum_array[8];
        _mm256_storeu_ps(sum_array, sum_vec);
        T sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];

        for (; i < size; ++i) {
          sum += data_[i];
        }

        return sum / static_cast<T>(1.0 * size);
      }
    }
#endif

    T sum = T(0);
    for (size_t i = 0; i < size; ++i) {
      sum += data_[i];
    }
    return (sum / (T(1) * size));
  }

  size_t size() const { return rows_ * cols_; }

  void resize(size_t newrows_, size_t newcols_) {
    if (newrows_ == rows_ && newcols_ == cols_) {
      return;
    }

    deallocate_aligned(data_);
    rows_ = newrows_;
    cols_ = newcols_;

    size_t size = rows_ * cols_;
    data_ = size > 0 ? allocate_aligned(size) : nullptr;
  }

  static void dot(const Matrix &a, const Matrix &b, Matrix &result) {
    if (a.cols_ != b.rows_ || result.rows_ != a.rows_ ||
        result.cols_ != b.cols_) {
      throw std::invalid_argument(
          "Matrix dimensions must match for dot product.");
    }

    result.fill(0.0);

    if constexpr (std::is_same_v<T, float>) {
      if (a.rows_ * b.cols_ > 1024) {
        a.avx2_matmul(a, b, result);
        return;
      }
    }

    const T *a_data = a.data_;
    const T *b_data = b.data_;
    T *result_data = result.data_;
    for (size_t r = 0; r < a.rows_; ++r) {
      for (size_t c = 0; c < b.cols_; ++c) {
        T sum = T(0);
        for (size_t k = 0; k < a.cols_; ++k) {
          sum += a_data[r * a.cols_ + k] * b_data[k * b.cols_ + c];
        }
        result_data[r * b.cols_ + c] = sum;
      }
    }
  }

  void fill_random_uniform(T range) {
    std::mt19937 gen(0);
    std::uniform_real_distribution<T> dis(-range, range);
    for (size_t i = 0; i < rows_ * cols_; ++i) {
      data_[i] = dis(gen);
    }
  }

  void fill_random_normal(T mean, T stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dis(mean, stddev);
    for (size_t i = 0; i < rows_ * cols_; ++i) {
      data_[i] = dis(gen);
    }
  }
};
