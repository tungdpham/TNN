#pragma once

#include <cstring>
#include <iostream>
#include <random>

// Matrix class without channels
template <typename T = float> struct Matrix {
private:
  T *data_; 
  int rows_, cols_;

public:
  // Default constructor
  Matrix() : rows_(0), cols_(0), data_(nullptr) {}

  Matrix(int rows_, int cols_, T *initialdata_ = nullptr)
      : rows_(rows_), cols_(cols_) {
    data_ = new T[rows_ * cols_]();
    if (initialdata_ != nullptr) {
      memcpy(data_, initialdata_, rows_ * cols_ * sizeof(T));
    } else {
      // Initialize with zeros if no initial data_ provided
      (*this).fill(0.0);
    }
  }

  Matrix(const Matrix &other)
      : rows_(other.rows_), cols_(other.cols_) {
    data_ = new T[rows_ * cols_];
    memcpy(data_, other.data_, rows_ * cols_ * sizeof(T));
  }

  // Move Constructor
  Matrix(Matrix &&other) noexcept
      : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {
    // Steal the data_ pointer from the temporary object
    other.rows_ = 0;
    other.cols_ = 0;
    other.data_ = nullptr;
  }

  // Move Assignment Operator
  Matrix &operator=(Matrix &&other) noexcept {
    if (this != &other) {
      delete[] data_; // Free existing memory

      // Steal data_ and dimensions from the other object
      rows_ = other.rows_;
      cols_ = other.cols_;
      data_ = other.data_;

      // Leave the other object in a valid but empty state
      other.rows_ = 0;
      other.cols_ = 0;
      other.data_ = nullptr;
    }
    return *this;
  }

  ~Matrix() {
    delete[] data_; // delete[] nullptr is safe in C++
  }

  T &operator()(int row, int col) {
    return data_[row * cols_ + col];
  }

  const T &operator()(int row, int col) const {
    return data_[row * cols_ + col];
  }

  inline T* data(){
    return data_;
  }

  void fill(T value) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows_ * cols_; ++i) {
      data_[i] = value;
    }
  }

  void print() const {
    for (int r = 0; r < rows_; ++r) {
      for (int c = 0; c < cols_; ++c) {
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
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows_ * cols_; ++i) {
      result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
  }

  inline Matrix operator+=(const Matrix &other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("Matrix dimensions must match for addition.");
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows_ * cols_; ++i) {
      data_[i] += other.data_[i];
    }
    return *this;
  }

  inline Matrix operator-(const Matrix &other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument(
          "Matrix dimensions must match for subtraction.");
    }
    Matrix result(rows_, cols_);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows_ * cols_; ++i) {
      result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
  }

  inline Matrix operator-=(const Matrix &other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument(
          "Matrix dimensions must match for subtraction.");
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows_ * cols_; ++i) {
      data_[i] -= other.data_[i];
    }
    return *this;
  }

  inline Matrix operator*(T scalar) const {
    Matrix result(rows_, cols_);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows_ * cols_; ++i) {
      result.data_[i] = data_[i] * scalar;
    }
    return result;
  }

  inline Matrix operator*=(T scalar) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows_ * cols_; ++i) {
      data_[i] *= scalar;
    }
    return *this;
  }

  inline Matrix operator/(T scalar) const {
    if (scalar == 0) {
      throw std::invalid_argument("Division by zero.");
    }
    Matrix result(rows_, cols_);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows_ * cols_; ++i) {
      result.data_[i] = data_[i] / scalar;
    }
    return result;
  }

  inline Matrix operator/=(T scalar) {
    if (scalar == 0) {
      throw std::invalid_argument("Division by zero.");
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows_ * cols_; ++i) {
      data_[i] /= scalar;
    }
    return *this;
  }

  inline Matrix operator*(const Matrix &other) const {
    if (cols_ != other.rows_) {
      throw std::invalid_argument(
          "Matrix dimensions must match for multiplication.");
    }
    Matrix result(rows_, other.cols_);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1)
#endif
    for (int r = 0; r < rows_; ++r) {
      for (int c = 0; c < other.cols_; ++c) {
        T sum = 0.0;
        for (int k = 0; k < cols_; ++k) {
          sum += (*this)(r, k) * other(k, c);
        }
        result(r, c) = sum;
      }
    }
    return result;
  }

  inline Matrix &operator=(const Matrix &other) {
    if (this != &other) {
      if (rows_ * cols_ != other.rows_ * other.cols_) {
        delete[] data_;
        data_ = new T[other.rows_ * other.cols_];
      }
      rows_ = other.rows_;
      cols_ = other.cols_;

      memcpy(data_, other.data_, rows_ * cols_ * sizeof(T));
    }
    return *this;
  }

  Matrix transpose() const {
    Matrix result(cols_, rows_);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1)
#endif
    for (int r = 0; r < rows_; ++r) {
      for (int c = 0; c < cols_; ++c) {
        result(c, r) = (*this)(r, c);
      }
    }
    return result;
  }

  Matrix reshape(int newrows_, int newcols_) const {
    if (rows_ * cols_ != newrows_ * newcols_) {
      throw std::invalid_argument(
          "Total number of elements must remain the same for reshape.");
    }
    return Matrix(newrows_, newcols_, data_);
  }

  Matrix pad(int padrows_, int padcols_, T value = 0.0) const {
    Matrix result(rows_ + 2 * padrows_, cols_ + 2 * padcols_);
    result.fill(value);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1)
#endif
    for (int r = 0; r < rows_; ++r) {
      for (int c = 0; c < cols_; ++c) {
        result(r + padrows_, c + padcols_) = (*this)(r, c);
      }
    }
    return result;
  }

  Matrix crop(int startRow, int startCol, int endRow, int endCol) const {
    if (startRow < 0 || startCol < 0 || endRow >= rows_ || endCol >= cols_ ||
        startRow >= endRow || startCol >= endCol) {
      throw std::invalid_argument("Invalid crop dimensions.");
    }
    Matrix result(endRow - startRow + 1, endCol - startCol + 1);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int r = startRow; r <= endRow; ++r) {
      for (int c = startCol; c <= endCol; ++c) {
        result(r - startRow, c - startCol) = (*this)(r, c);
      }
    }
    return result;
  }

  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }

  T mean() const {
    T sum = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum)
#endif
    for (int i = 0; i < rows_ * cols_; ++i) {
      sum += data_[i];
    }
    return (sum / (1.0 * rows_ * cols_));
  }

  int size() const { return rows_ * cols_; }

  void resize(int newrows_, int newcols_) {
    if (newrows_ == rows_ && newcols_ == cols_) {
      return; // Nothing to do
    }

    delete[] data_;
    rows_ = newrows_;
    cols_ = newcols_;

    int size = rows_ * cols_;
    data_ = size > 0 ? new T[size]() : nullptr;
  }

  static void dot(const Matrix &a, const Matrix &b, Matrix &result) {
    if (a.cols_ != b.rows_ || result.rows_ != a.rows_ || result.cols_ != b.cols_) {
      throw std::invalid_argument(
          "Matrix dimensions must match for dot product.");
    }

    const T* a_data = a.data_;
    const T* b_data = b.data_;
    T* result_data = result.data_;
    
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1)
#endif
    for (int r = 0; r < a.rows_; ++r) {
      for (int c = 0; c < b.cols_; ++c) {
        T sum = 0.0;
        for (int k = 0; k < a.cols_; ++k) {
          sum += a_data[r * a.cols_ + k] * b_data[k * b.cols_ + c]; 
        }
        result_data[r * b.cols_ + c] = sum;
      }
    }
  }

  std::vector<T> to_vector() const {
    return std::vector<T>(data_, data_ + rows_ * cols_);
  }

    void fill_random_uniform(T range) {
    std::mt19937 gen(0);
    std::uniform_real_distribution<T> dis(-range, range);
    for (int i = 0; i < rows_ * cols_; ++i) {
      data_[i] = dis(gen);
    }
  }

  void fill_random_normal(T stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dis(0, stddev);
    for (int i = 0; i < rows_ * cols_; ++i) {
      data_[i] = dis(gen);
    }
  }
};
