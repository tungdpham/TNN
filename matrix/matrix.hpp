#pragma once

#include <cstring>
#include <iostream>
#include <random>

// Channel-Major layout Matrix class
// C-H-W Layout
template <typename T = float> struct Matrix {
  int rows, cols, channels;
  T *data;

  // Default constructor
  Matrix() : rows(0), cols(0), channels(0), data(nullptr) {}

  Matrix(int rows, int cols, int channels = 1, T *initialData = nullptr)
      : rows(rows), cols(cols), channels(channels) {
    data = new T[rows * cols * channels]();
    if (initialData != nullptr) {
      memcpy(data, initialData, rows * cols * channels * sizeof(T));
    } else {
      // Initialize with zeros if no initial data provided
      (*this).fill(0.0);
    }
  }

  Matrix(const Matrix &other)
      : rows(other.rows), cols(other.cols), channels(other.channels) {
    data = new T[rows * cols * channels];
    memcpy(data, other.data, rows * cols * channels * sizeof(T));
  }

  // Move Constructor
  Matrix(Matrix &&other) noexcept
      : rows(other.rows), cols(other.cols), channels(other.channels),
        data(other.data) {
    // Steal the data pointer from the temporary object
    other.rows = 0;
    other.cols = 0;
    other.channels = 0;
    other.data = nullptr;
  }

  // Move Assignment Operator
  Matrix &operator=(Matrix &&other) noexcept {
    if (this != &other) {
      delete[] data; // Free existing memory

      // Steal data and dimensions from the other object
      rows = other.rows;
      cols = other.cols;
      channels = other.channels;
      data = other.data;

      // Leave the other object in a valid but empty state
      other.rows = 0;
      other.cols = 0;
      other.channels = 0;
      other.data = nullptr;
    }
    return *this;
  }

  ~Matrix() {
    delete[] data; // delete[] nullptr is safe in C++
  }

  T &operator()(int row, int col, int channel = 0) {
    return data[channel * (rows * cols) + row * cols + col];
  }

  const T &operator()(int row, int col, int channel = 0) const {
    return data[channel * (rows * cols) + row * cols + col];
  }

  void fill(T value) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols * channels; ++i) {
      data[i] = value;
    }
  }

  void print() const {
    for (int ch = 0; ch < channels; ++ch) {
      std::cout << "Channel " << ch << ":" << std::endl;
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          std::cout << (*this)(r, c, ch) << " ";
        }
        std::cout << std::endl;
      }
      if (ch < channels - 1) {
        std::cout << std::endl;
      }
    }
  }

  inline Matrix operator+(const Matrix &other) const {
    if (rows != other.rows || cols != other.cols ||
        channels != other.channels) {
      throw std::invalid_argument("Matrix dimensions must match for addition.");
    }
    Matrix result(rows, cols, channels);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols * channels; ++i) {
      result.data[i] = data[i] + other.data[i];
    }
    return result;
  }

  inline Matrix operator+=(const Matrix &other) {
    if (rows != other.rows || cols != other.cols ||
        channels != other.channels) {
      throw std::invalid_argument("Matrix dimensions must match for addition.");
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols * channels; ++i) {
      data[i] += other.data[i];
    }
    return *this;
  }

  inline Matrix operator-(const Matrix &other) const {
    if (rows != other.rows || cols != other.cols ||
        channels != other.channels) {
      throw std::invalid_argument(
          "Matrix dimensions must match for subtraction.");
    }
    Matrix result(rows, cols, channels);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols * channels; ++i) {
      result.data[i] = data[i] - other.data[i];
    }
    return result;
  }

  inline Matrix operator-=(const Matrix &other) {
    if (rows != other.rows || cols != other.cols ||
        channels != other.channels) {
      throw std::invalid_argument(
          "Matrix dimensions must match for subtraction.");
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols * channels; ++i) {
      data[i] -= other.data[i];
    }
    return *this;
  }

  inline Matrix operator*(T scalar) const {
    Matrix result(rows, cols, channels);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols * channels; ++i) {
      result.data[i] = data[i] * scalar;
    }
    return result;
  }

  inline Matrix operator*=(T scalar) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols * channels; ++i) {
      data[i] *= scalar;
    }
    return *this;
  }

  inline Matrix operator/(T scalar) const {
    if (scalar == 0) {
      throw std::invalid_argument("Division by zero.");
    }
    Matrix result(rows, cols, channels);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols * channels; ++i) {
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
    for (int i = 0; i < rows * cols * channels; ++i) {
      data[i] /= scalar;
    }
    return *this;
  }

  inline Matrix operator*(const Matrix &other) const {
    if (cols != other.rows || channels != other.channels) {
      throw std::invalid_argument(
          "Matrix dimensions must match for multiplication.");
    }
    Matrix result(rows, other.cols, channels);
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < other.cols; ++c) {
        for (int ch = 0; ch < channels; ++ch) {
          T sum = 0.0;
          for (int k = 0; k < cols; ++k) {
            sum += (*this)(r, k, ch) * other(k, c, ch);
          }
          result(r, c, ch) = sum;
        }
      }
    }
    return result;
  }

  inline Matrix &operator=(const Matrix &other) {
    if (this != &other) {
      if (rows * cols * channels != other.rows * other.cols * other.channels) {
        delete[] data;
        data = new T[other.rows * other.cols * other.channels];
      }
      rows = other.rows;
      cols = other.cols;
      channels = other.channels;

      memcpy(data, other.data, rows * cols * channels * sizeof(T));
    }
    return *this;
  }

  Matrix transpose() const {
    Matrix result(cols, rows, channels);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        for (int ch = 0; ch < channels; ++ch) {
          result(c, r, ch) = (*this)(r, c, ch);
        }
      }
    }
    return result;
  }

  Matrix reshape(int newRows, int newCols, int newChannels = 1) const {
    if (rows * cols * channels != newRows * newCols * newChannels) {
      throw std::invalid_argument(
          "Total number of elements must remain the same for reshape.");
    }
    return Matrix(newRows, newCols, newChannels, data);
  }

  Matrix pad(int padRows, int padCols, T value = 0.0) const {
    Matrix result(rows + 2 * padRows, cols + 2 * padCols, channels);
    result.fill(value);
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        for (int ch = 0; ch < channels; ++ch) {
          result(r + padRows, c + padCols, ch) = (*this)(r, c, ch);
        }
      }
    }
    return result;
  }

  Matrix crop(int startRow, int startCol, int endRow, int endCol) const {
    if (startRow < 0 || startCol < 0 || endRow >= rows || endCol >= cols ||
        startRow >= endRow || startCol >= endCol) {
      throw std::invalid_argument("Invalid crop dimensions.");
    }
    Matrix result(endRow - startRow + 1, endCol - startCol + 1, channels);
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (int r = startRow; r <= endRow; ++r) {
      for (int c = startCol; c <= endCol; ++c) {
        for (int ch = 0; ch < channels; ++ch) {
          result(r - startRow, c - startCol, ch) = (*this)(r, c, ch);
        }
      }
    }
    return result;
  }

  Matrix get_chans(int startChan, int endChan) const {
    if (startChan < 0 || endChan >= channels || startChan > endChan) {
      throw std::invalid_argument("Invalid channel range.");
    }
    return Matrix(rows, cols, endChan - startChan + 1,
                  data + startChan * rows * cols);
  }

  T mean() const {
    T sum = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum)
#endif
    for (int i = 0; i < rows * cols * channels; ++i) {
      sum += data[i];
    }
    return (sum / (1.0 * rows * cols * channels));
  }

  T remove_mean(int channel) {
    int s = rows * cols;
    T mean = 0.0;
    int offset = channel * s;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : mean)
#endif
    for (int i = 0; i < s; ++i) {
      mean += data[offset + i];
    }
    mean /= 1.0 * s;
    for (int i = 0; i < s; ++i) {
      data[offset + i] -= mean;
    }
    return mean;
  }

  void fill_random_uniform(T range) {
    std::mt19937 gen(0);
    std::uniform_real_distribution<T> dis(-range, range);
    for (int i = 0; i < rows * cols * channels; ++i) {
      data[i] = dis(gen);
    }
  }

  void fill_random_normal(T stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dis(0, stddev);
    for (int i = 0; i < rows * cols * channels; ++i) {
      data[i] = dis(gen);
    }
  }

  int size() const { return rows * cols * channels; }

  void resize(int newRows, int newCols, int newChannels = 1) {
    if (newRows == rows && newCols == cols && newChannels == channels) {
      return; // Nothing to do
    }

    delete[] data;
    rows = newRows;
    cols = newCols;
    channels = newChannels;

    int size = rows * cols * channels;
    data = size > 0 ? new T[size]() : nullptr;
  }

  void dot(const Matrix &other, Matrix &result) const {
    if (cols != other.rows || channels != other.channels) {
      throw std::invalid_argument(
          "Matrix dimensions must match for dot product.");
    }
    result = Matrix(rows, other.cols, channels);
    result = (*this) * other;
  }

  std::vector<T> to_vector() const {
    return std::vector<T>(data, data + rows * cols * channels);
  }
};
