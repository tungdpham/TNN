/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#pragma once

#include <cstring>
#include <random>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include "ops/ops.hpp"
#include <cstdlib>
#ifdef _WIN32
#include <malloc.h>
#endif

namespace tnn {
template <typename T = float> struct Matrix {
private:
  size_t rows_, cols_;
  const Device *device_;
  device_ptr<T[]> data_;

  static constexpr size_t MKL_ALIGNMENT = 64;
  static constexpr size_t AVX2_ALIGNMENT = 32;

  void allocate_aligned(size_t count) {
    if (count == 0)
      return;

    size_t bytes = count * sizeof(T);
    size_t aligned_size = ((bytes + MKL_ALIGNMENT - 1) / MKL_ALIGNMENT) * MKL_ALIGNMENT;

    data_ = make_array_ptr<T[]>(device_, aligned_size / sizeof(T));
  }

public:
  Matrix(const Device *device) : rows_(0), cols_(0), device_(device), data_(nullptr) {}

  Matrix(size_t rows, size_t cols, const Device *device = &getCPU())
      : rows_(rows), cols_(cols), device_(device) {
    allocate_aligned(rows_ * cols_);
  }

  Matrix(size_t rows, size_t cols, const device_ptr<T[]> &data, const Device *device = &getCPU())
      : rows_(rows), cols_(cols), device_(device) {
    allocate_aligned(rows_ * cols_);
    if (data.get() != nullptr) {
      ops::copy(data, data_, rows_ * cols_)->sync();
    }
  }

  Matrix(const Matrix<T> &other) {
    this->rows_ = other.rows_;
    this->cols_ = other.cols_;
    this->device_ = other.device_;
    allocate_aligned(rows_ * cols_);
    ops::copy(other.data_, data_, rows_ * cols_)->sync();
  }

  Matrix(Matrix<T> &&other) noexcept
      : rows_(other.rows_), cols_(other.cols_), data_(std::move(other.data_)) {
    other.rows_ = 0;
    other.cols_ = 0;
    other.data_ = nullptr;
  }

  Matrix &operator=(const Matrix<T> &other) = delete;

  Matrix &operator=(Matrix<T> &&other) noexcept {
    if (this != &other) {
      rows_ = other.rows_;
      cols_ = other.cols_;
      data_ = std::move(other.data_);

      other.rows_ = 0;
      other.cols_ = 0;
      other.data_ = nullptr;
    }
    return *this;
  }

  ~Matrix() {}

  const T *data() const { return data_.get(); }
  T *data() { return data_.get(); }

  void fill(T value) { ops::set_scalar(data_, value, rows_ * cols_)->sync(); }

  inline Matrix<T> operator+(const Matrix<T> &other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("Matrix<T> dimensions must match for addition.");
    }
    Matrix<T> result(rows_, cols_);
    size_t size = rows_ * cols_;

    ops::add(data_, other.data_, result.data_, size)->sync();
    return result;
  }

  inline Matrix<T> &operator+=(const Matrix<T> &other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("Matrix<T> dimensions must match for addition.");
    }
    size_t size = rows_ * cols_;

    ops::add(data_, other.data_, data_, size)->sync();
    return *this;
  }

  inline Matrix<T> operator-(const Matrix<T> &other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("Matrix<T> dimensions must match for subtraction.");
    }
    Matrix<T> result(rows_, cols_);
    size_t size = rows_ * cols_;

    ops::sub(data_, other.data_, result.data_, size)->sync();
    return result;
  }

  inline Matrix<T> &operator-=(const Matrix<T> &other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("Matrix<T> dimensions must match for subtraction.");
    }
    size_t size = rows_ * cols_;

    ops::sub(data_, other.data_, data_, size)->sync();

    return *this;
  }

  inline Matrix<T> operator*(T scalar) const {
    Matrix<T> result(rows_, cols_);
    size_t size = rows_ * cols_;

    ops::mul_scalar(data_, scalar, result.data_, size)->sync();
    return result;
  }

  inline Matrix<T> &operator*=(T scalar) {
    size_t size = rows_ * cols_;

    ops::mul_scalar(data_, scalar, data_, size)->sync();
    return *this;
  }

  inline Matrix<T> operator/(T scalar) const {
    if (scalar == 0) {
      throw std::invalid_argument("Division by zero.");
    }
    Matrix<T> result(rows_, cols_);
    size_t size = rows_ * cols_;

    ops::div_scalar(data_, scalar, result.data_, size)->sync();
    return result;
  }

  inline Matrix<T> &operator/=(T scalar) {
    if (scalar == 0) {
      throw std::invalid_argument("Division by zero.");
    }
    size_t size = rows_ * cols_;

    ops::div_scalar(data_, scalar, data_, size)->sync();

    return *this;
  }

  inline Matrix<T> operator*(const Matrix<T> &other) const {
    if (cols_ != other.rows_) {
      throw std::invalid_argument("Matrix<T> dimensions must match for multiplication.");
    }
    Matrix<T> result(rows_, other.cols_);
    result.fill(0.0);

    ops::mul(data_, other.data_, result.data_, size())->sync();
    return result;
  }

  Matrix<T> clone() const { return Matrix(rows_, cols_, data_); }

  Matrix<T> reshape(size_t newrows_, size_t newcols_) const {
    if (rows_ * cols_ != newrows_ * newcols_) {
      throw std::invalid_argument("Total number of elements must remain the same for reshape.");
    }
    return Matrix(newrows_, newcols_, data_);
  }

  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  size_t size() const { return rows_ * cols_; }

  void resize(size_t newrows_, size_t newcols_) {
    if (newrows_ == rows_ && newcols_ == cols_) {
      return;
    }

    rows_ = newrows_;
    cols_ = newcols_;

    size_t size = rows_ * cols_;
    allocate_aligned(size);
  }

  void fill_random_uniform(T range) {
    ops::fill_random_uniform(data_, rows_ * cols_, -range, range,
                             static_cast<unsigned long long>(std::random_device{}()))
        ->sync();
  }

  void fill_random_normal(T mean, T stddev) {
    ops::fill_random_normal(data_, rows_ * cols_, mean, stddev,
                            static_cast<unsigned long long>(std::random_device{}()))
        ->sync();
  }
};
} // namespace tnn