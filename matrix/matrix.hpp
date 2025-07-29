#pragma once

#include <cstring>
#include <iostream>
#include <random>

// Matrix class without channels
template <typename T = float> struct Matrix {
  int rows, cols;
  T *data;

  // Default constructor
  Matrix() : rows(0), cols(0), data(nullptr) {}

  Matrix(int rows, int cols, T *initialData = nullptr)
      : rows(rows), cols(cols) {
    data = new T[rows * cols]();
    if (initialData != nullptr) {
      memcpy(data, initialData, rows * cols * sizeof(T));
    } else {
      // Initialize with zeros if no initial data provided
      (*this).fill(0.0);
    }
  }

  Matrix(const Matrix &other)
      : rows(other.rows), cols(other.cols) {
    data = new T[rows * cols];
    memcpy(data, other.data, rows * cols * sizeof(T));
  }

  // Move Constructor
  Matrix(Matrix &&other) noexcept
      : rows(other.rows), cols(other.cols), data(other.data) {
    // Steal the data pointer from the temporary object
    other.rows = 0;
    other.cols = 0;
    other.data = nullptr;
  }

  // Move Assignment Operator
  Matrix &operator=(Matrix &&other) noexcept {
    if (this != &other) {
      delete[] data; // Free existing memory

      // Steal data and dimensions from the other object
      rows = other.rows;
      cols = other.cols;
      data = other.data;

      // Leave the other object in a valid but empty state
      other.rows = 0;
      other.cols = 0;
      other.data = nullptr;
    }
    return *this;
  }

  ~Matrix() {
    delete[] data; // delete[] nullptr is safe in C++
  }

  T &operator()(int row, int col) {
    return data[row * cols + col];
  }

  const T &operator()(int row, int col) const {
    return data[row * cols + col];
  }

  void fill(T value) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols; ++i) {
      data[i] = value;
    }
  }

  void print() const {
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        std::cout << (*this)(r, c) << " ";
      }
      std::cout << std::endl;
    }
  }

  inline Matrix operator+(const Matrix &other) const {
    if (rows != other.rows || cols != other.cols) {
      throw std::invalid_argument("Matrix dimensions must match for addition.");
    }
    Matrix result(rows, cols);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols; ++i) {
      result.data[i] = data[i] + other.data[i];
    }
    return result;
  }

  inline Matrix operator+=(const Matrix &other) {
    if (rows != other.rows || cols != other.cols) {
      throw std::invalid_argument("Matrix dimensions must match for addition.");
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols; ++i) {
      data[i] += other.data[i];
    }
    return *this;
  }

  inline Matrix operator-(const Matrix &other) const {
    if (rows != other.rows || cols != other.cols) {
      throw std::invalid_argument(
          "Matrix dimensions must match for subtraction.");
    }
    Matrix result(rows, cols);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols; ++i) {
      result.data[i] = data[i] - other.data[i];
    }
    return result;
  }

  inline Matrix operator-=(const Matrix &other) {
    if (rows != other.rows || cols != other.cols) {
      throw std::invalid_argument(
          "Matrix dimensions must match for subtraction.");
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols; ++i) {
      data[i] -= other.data[i];
    }
    return *this;
  }

  inline Matrix operator*(T scalar) const {
    Matrix result(rows, cols);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols; ++i) {
      result.data[i] = data[i] * scalar;
    }
    return result;
  }

  inline Matrix operator*=(T scalar) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols; ++i) {
      data[i] *= scalar;
    }
    return *this;
  }

  inline Matrix operator/(T scalar) const {
    if (scalar == 0) {
      throw std::invalid_argument("Division by zero.");
    }
    Matrix result(rows, cols);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols; ++i) {
      result.data[i] = data[i] / scalar;
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
    for (int i = 0; i < rows * cols; ++i) {
      data[i] /= scalar;
    }
    return *this;
  }

  inline Matrix operator*(const Matrix &other) const {
    if (cols != other.rows) {
      throw std::invalid_argument(
          "Matrix dimensions must match for multiplication.");
    }
    Matrix result(rows, other.cols);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1)
#endif
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < other.cols; ++c) {
        T sum = 0.0;
        for (int k = 0; k < cols; ++k) {
          sum += (*this)(r, k) * other(k, c);
        }
        result(r, c) = sum;
      }
    }
    return result;
  }

  inline Matrix &operator=(const Matrix &other) {
    if (this != &other) {
      if (rows * cols != other.rows * other.cols) {
        delete[] data;
        data = new T[other.rows * other.cols];
      }
      rows = other.rows;
      cols = other.cols;

      memcpy(data, other.data, rows * cols * sizeof(T));
    }
    return *this;
  }

  Matrix transpose() const {
    Matrix result(cols, rows);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1)
#endif
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        result(c, r) = (*this)(r, c);
      }
    }
    return result;
  }

  Matrix reshape(int newRows, int newCols) const {
    if (rows * cols != newRows * newCols) {
      throw std::invalid_argument(
          "Total number of elements must remain the same for reshape.");
    }
    return Matrix(newRows, newCols, data);
  }

  Matrix pad(int padRows, int padCols, T value = 0.0) const {
    Matrix result(rows + 2 * padRows, cols + 2 * padCols);
    result.fill(value);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1)
#endif
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        result(r + padRows, c + padCols) = (*this)(r, c);
      }
    }
    return result;
  }

  Matrix crop(int startRow, int startCol, int endRow, int endCol) const {
    if (startRow < 0 || startCol < 0 || endRow >= rows || endCol >= cols ||
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

  T mean() const {
    T sum = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum)
#endif
    for (int i = 0; i < rows * cols; ++i) {
      sum += data[i];
    }
    return (sum / (1.0 * rows * cols));
  }

  int size() const { return rows * cols; }

  void resize(int newRows, int newCols) {
    if (newRows == rows && newCols == cols) {
      return; // Nothing to do
    }

    delete[] data;
    rows = newRows;
    cols = newCols;

    int size = rows * cols;
    data = size > 0 ? new T[size]() : nullptr;
  }

  void dot(const Matrix &other, Matrix &result) const {
    if (cols != other.rows) {
      throw std::invalid_argument(
          "Matrix dimensions must match for dot product.");
    }
    result = Matrix(rows, other.cols);
    result = (*this) * other;
  }

  std::vector<T> to_vector() const {
    return std::vector<T>(data, data + rows * cols);
  }

    void fill_random_uniform(T range) {
    std::mt19937 gen(0);
    std::uniform_real_distribution<T> dis(-range, range);
    for (int i = 0; i < rows * cols; ++i) {
      data[i] = dis(gen);
    }
  }

  void fill_random_normal(T stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dis(0, stddev);
    for (int i = 0; i < rows * cols; ++i) {
      data[i] = dis(gen);
    }
  }
};
