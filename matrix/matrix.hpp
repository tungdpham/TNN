#pragma once

#include <cstring>
#include <iostream>
#include <random>
#ifdef __AVX2__
#include <immintrin.h> // For AVX2 intrinsics
#endif
#include <algorithm>
#include <vector>
#include <cstdlib>     // For aligned_alloc and free
#include <new>         // For std::bad_alloc

// Matrix class without channels
template <typename T = float> struct Matrix {
private:
  T *data_; 
  int rows_, cols_;

  // Memory alignment helpers for AVX2 optimization
  static constexpr size_t AVX2_ALIGNMENT = 32; // 32-byte alignment for AVX2
  
  // Aligned memory allocation
  T* allocate_aligned(size_t count) {
    if (count == 0) return nullptr;
    
    // Calculate total bytes needed including padding for alignment
    size_t bytes = count * sizeof(T);
    
    // Use aligned_alloc for C++17 compliance
    void* ptr = std::aligned_alloc(AVX2_ALIGNMENT, 
                                   ((bytes + AVX2_ALIGNMENT - 1) / AVX2_ALIGNMENT) * AVX2_ALIGNMENT);
    if (!ptr) {
      throw std::bad_alloc();
    }
    
    // Initialize to zero
    std::memset(ptr, 0, bytes);
    return static_cast<T*>(ptr);
  }
  
  // Aligned memory deallocation
  void deallocate_aligned(T* ptr) {
    if (ptr) {
      std::free(ptr);
    }
  }

  // AVX2 helper functions for vectorized operations
  inline void avx2_fill(T value, int size) {
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
      const int avx_size = 8; // 8 floats per AVX2 register
      const __m256 avx_value = _mm256_set1_ps(value);
      
      int i = 0;
      // Process 8 elements at a time with aligned stores
      for (; i + avx_size <= size; i += avx_size) {
        _mm256_store_ps(data_ + i, avx_value); // Use aligned store
      }
      // Handle remaining elements
      for (; i < size; ++i) {
        data_[i] = value;
      }
    } else {
      // Fallback for non-float types
      for (int i = 0; i < size; ++i) {
        data_[i] = value;
      }
    }
#else
    // Fallback for non-AVX2 systems
    std::fill(data_, data_ + size, value);
#endif
  }

  inline void avx2_add(const T* a, const T* b, T* result, int size) {
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
      const int avx_size = 8; // 8 floats per AVX2 register
      
      int i = 0;
      // Process 8 elements at a time with aligned loads/stores
      for (; i + avx_size <= size; i += avx_size) {
        __m256 va = _mm256_load_ps(a + i);     // Use aligned load
        __m256 vb = _mm256_load_ps(b + i);     // Use aligned load
        __m256 vresult = _mm256_add_ps(va, vb);
        _mm256_store_ps(result + i, vresult);  // Use aligned store
      }
      // Handle remaining elements
      for (; i < size; ++i) {
        result[i] = a[i] + b[i];
      }
    } else {
      // Fallback for non-float types
      for (int i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
      }
    }
#else
    std::copy(a, a + size, result);
    for (int i = 0; i < size; ++i) {
      result[i] += b[i];  
    }
#endif
  }

  inline void avx2_sub(const T* a, const T* b, T* result, int size) {
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
      const int avx_size = 8; // 8 floats per AVX2 register
      
      int i = 0;
      // Process 8 elements at a time with aligned loads/stores
      for (; i + avx_size <= size; i += avx_size) {
        __m256 va = _mm256_load_ps(a + i);     // Use aligned load
        __m256 vb = _mm256_load_ps(b + i);     // Use aligned load
        __m256 vresult = _mm256_sub_ps(va, vb);
        _mm256_store_ps(result + i, vresult);  // Use aligned store
      }
      // Handle remaining elements
      for (; i < size; ++i) {
        result[i] = a[i] - b[i];
      }
    } else {
      // Fallback for non-float types
      for (int i = 0; i < size; ++i) {
        result[i] = a[i] - b[i];
      }
    }
#else
    std::copy(a, a + size, result);
    for (int i = 0; i < size; ++i) {
      result[i] -= b[i];  
    }
#endif
  }

  inline void avx2_mul_scalar(const T* a, T scalar, T* result, int size) {
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
      const int avx_size = 8; // 8 floats per AVX2 register
      const __m256 avx_scalar = _mm256_set1_ps(scalar);
      
      int i = 0;
      // Process 8 elements at a time with aligned loads/stores
      for (; i + avx_size <= size; i += avx_size) {
        __m256 va = _mm256_load_ps(a + i);     // Use aligned load
        __m256 vresult = _mm256_mul_ps(va, avx_scalar);
        _mm256_store_ps(result + i, vresult);  // Use aligned store
      }
      // Handle remaining elements
      for (; i < size; ++i) {
        result[i] = a[i] * scalar;
      }
    } else {
      // Fallback for non-float types
      for (int i = 0; i < size; ++i) {
        result[i] = a[i] * scalar;
      }
    }
#else
    std::copy(a, a + size, result);
    for (int i = 0; i < size; ++i) {
      result[i] *= scalar;
    }
#endif
  }

  inline void avx2_div_scalar(const T* a, T scalar, T* result, int size) {
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
      const int avx_size = 8; // 8 floats per AVX2 register
      const __m256 avx_scalar = _mm256_set1_ps(scalar);
      
      int i = 0;
      // Process 8 elements at a time with aligned loads/stores
      for (; i + avx_size <= size; i += avx_size) {
        __m256 va = _mm256_load_ps(a + i);     // Use aligned load
        __m256 vresult = _mm256_div_ps(va, avx_scalar);
        _mm256_store_ps(result + i, vresult);  // Use aligned store
      }
      // Handle remaining elements
      for (; i < size; ++i) {
        result[i] = a[i] / scalar;
      }
    } else {
      // Fallback for non-float types
      for (int i = 0; i < size; ++i) {
        result[i] = a[i] / scalar;
      }
    }
#else
    std::copy(a, a + size, result);
    for (int i = 0; i < size; ++i) {
      result[i] /= scalar;  
    }
#endif
  }

  // AVX2-optimized matrix multiplication for float matrices
  inline void avx2_matmul(const Matrix &a, const Matrix &b, Matrix &result) {
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
      const int avx_size = 8; // 8 floats per AVX2 register
      
      // Use tiled multiplication for better cache performance
      const int tile_size = 64;
      
      for (int ii = 0; ii < a.rows_; ii += tile_size) {
        for (int jj = 0; jj < b.cols_; jj += tile_size) {
          for (int kk = 0; kk < a.cols_; kk += tile_size) {
            
            int i_end = std::min(ii + tile_size, a.rows_);
            int j_end = std::min(jj + tile_size, b.cols_);
            int k_end = std::min(kk + tile_size, a.cols_);
            
            for (int i = ii; i < i_end; ++i) {
              for (int j = jj; j + avx_size <= j_end; j += avx_size) {
                __m256 sum = _mm256_load_ps(&result.data_[i * b.cols_ + j]); // Use aligned load
                
                for (int k = kk; k < k_end; ++k) {
                  __m256 a_val = _mm256_set1_ps(a.data_[i * a.cols_ + k]);
                  __m256 b_val = _mm256_load_ps(&b.data_[k * b.cols_ + j]); // Use aligned load
                  sum = _mm256_fmadd_ps(a_val, b_val, sum);
                }
                
                _mm256_store_ps(&result.data_[i * b.cols_ + j], sum); // Use aligned store
              }
              
              // Handle remaining columns
              for (int j = (j_end / avx_size) * avx_size; j < j_end; ++j) {
                T sum = result.data_[i * b.cols_ + j];
                for (int k = kk; k < k_end; ++k) {
                  sum += a.data_[i * a.cols_ + k] * b.data_[k * b.cols_ + j];
                }
                result.data_[i * b.cols_ + j] = sum;
              }
            }
          }
        }
      }
    } else {
      // Fallback for non-float types - use standard algorithm
      for (int i = 0; i < a.rows_; ++i) {
        for (int j = 0; j < b.cols_; ++j) {
          T sum = 0;
          for (int k = 0; k < a.cols_; ++k) {
            sum += a.data_[i * a.cols_ + k] * b.data_[k * b.cols_ + j];
          }
          result.data_[i * b.cols_ + j] = sum;
        }
      }
    }
#else
    // Fallback for non-AVX2 systems - use standard algorithm
    for (int i = 0; i < a.rows_; ++i) {
      for (int j = 0; j < b.cols_; ++j) {
        T sum = 0;
        for (int k = 0; k < a.cols_; ++k) {
          sum += a.data_[i * a.cols_ + k] * b.data_[k * b.cols_ + j];
        }
        result.data_[i * b.cols_ + j] = sum;  
      }
    }
#endif
  }

public:
  // Default constructor
  Matrix() : rows_(0), cols_(0), data_(nullptr) {}

  Matrix(int rows_, int cols_, T *initialdata_ = nullptr)
      : rows_(rows_), cols_(cols_) {
    data_ = allocate_aligned(rows_ * cols_);
    if (initialdata_ != nullptr) {
      memcpy(data_, initialdata_, rows_ * cols_ * sizeof(T));
    } else {
      // Initialize with zeros if no initial data_ provided (already done in allocate_aligned)
      // (*this).fill(0.0);
    }
  }

  Matrix(const Matrix &other)
      : rows_(other.rows_), cols_(other.cols_) {
    data_ = allocate_aligned(rows_ * cols_);
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
      deallocate_aligned(data_); // Free existing memory

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
    deallocate_aligned(data_); // deallocate_aligned handles nullptr safely
  }

  T &operator()(int row, int col) {
    return data_[row * cols_ + col];
  }

  const T &operator()(int row, int col) const {
    return data_[row * cols_ + col];
  }

  T* data(){
    return data_;
  }

  T* data() const {
    return data_;
  }

  void fill(T value) {
    int size = rows_ * cols_;
    if (size > 32) { // Use AVX2 for larger arrays
      avx2_fill(value, size);
    } else {
      // Use simple loop for small arrays
      for (int i = 0; i < size; ++i) {
        data_[i] = value;
      }
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
    int size = rows_ * cols_;
    
    if (size > 32) { // Use AVX2 for larger arrays
      avx2_add(data_, other.data_, result.data_, size);
    } else {
      // Use simple loop for small arrays
      for (int i = 0; i < size; ++i) {
        result.data_[i] = data_[i] + other.data_[i];
      }
    }
    return result;
  }

  inline Matrix operator+=(const Matrix &other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("Matrix dimensions must match for addition.");
    }
    int size = rows_ * cols_;
    
    if (size > 32) { // Use AVX2 for larger arrays
      avx2_add(data_, other.data_, data_, size);
    } else {
      // Use simple loop for small arrays
      for (int i = 0; i < size; ++i) {
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
    int size = rows_ * cols_;
    
    if (size > 32) { // Use AVX2 for larger arrays
      avx2_sub(data_, other.data_, result.data_, size);
    } else {
      // Use simple loop for small arrays
      for (int i = 0; i < size; ++i) {
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
    int size = rows_ * cols_;
    
    if (size > 32) { // Use AVX2 for larger arrays
      avx2_sub(data_, other.data_, data_, size);
    } else {
      // Use simple loop for small arrays
      for (int i = 0; i < size; ++i) {
        data_[i] -= other.data_[i];
      }
    }
    return *this;
  }

  inline Matrix operator*(T scalar) const {
    Matrix result(rows_, cols_);
    int size = rows_ * cols_;
    
    if (size > 32) { // Use AVX2 for larger arrays
      avx2_mul_scalar(data_, scalar, result.data_, size);
    } else {
      // Use simple loop for small arrays
      for (int i = 0; i < size; ++i) {
        result.data_[i] = data_[i] * scalar;
      }
    }
    return result;
  }

  inline Matrix operator*=(T scalar) {
    int size = rows_ * cols_;
    
    if (size > 32) { // Use AVX2 for larger arrays
      avx2_mul_scalar(data_, scalar, data_, size);
    } else {
      // Use simple loop for small arrays
      for (int i = 0; i < size; ++i) {
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
    int size = rows_ * cols_;
    
    if (size > 32) { // Use AVX2 for larger arrays
      avx2_div_scalar(data_, scalar, result.data_, size);
    } else {
      // Use simple loop for small arrays
      for (int i = 0; i < size; ++i) {
        result.data_[i] = data_[i] / scalar;
      }
    }
    return result;
  }

  inline Matrix operator/=(T scalar) {
    if (scalar == 0) {
      throw std::invalid_argument("Division by zero.");
    }
    int size = rows_ * cols_;
    
    if (size > 32) { // Use AVX2 for larger arrays
      avx2_div_scalar(data_, scalar, data_, size);
    } else {
      // Use simple loop for small arrays
      for (int i = 0; i < size; ++i) {
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
    result.fill(0.0); // Initialize to zero for accumulation
    
    // Use AVX2 for float matrices and larger sizes
    if constexpr (std::is_same_v<T, float>) {
      if (rows_ * other.cols_ > 1024) { // Use AVX2 for larger matrices
        avx2_matmul(*this, other, result);
        return result;
      }
    }
    
    // Fallback to standard algorithm for small matrices or non-float types
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
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
        deallocate_aligned(data_);
        data_ = allocate_aligned(other.rows_ * other.cols_);
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
#pragma omp parallel for collapse(2) schedule(static)
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
#pragma omp parallel for schedule(static)
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
    int size = rows_ * cols_;
    
#ifdef __AVX2__
    if constexpr (std::is_same_v<T, float>) {
      if (size > 32) { // Use AVX2 for larger arrays
        const int avx_size = 8; // 8 floats per AVX2 register
        __m256 sum_vec = _mm256_setzero_ps();
        
        int i = 0;
        // Process 8 elements at a time with aligned loads
        for (; i + avx_size <= size; i += avx_size) {
          __m256 data_vec = _mm256_load_ps(data_ + i); // Use aligned load
          sum_vec = _mm256_add_ps(sum_vec, data_vec);
        }
        
        // Horizontal sum of the vector
        float sum_array[8];
        _mm256_storeu_ps(sum_array, sum_vec);
        T sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
        
        // Handle remaining elements
        for (; i < size; ++i) {
          sum += data_[i];
        }
        
        return sum / (1.0 * size);
      }
    }
#endif
    
    // Fallback for small arrays or non-float types
    T sum = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum)
#endif
    for (int i = 0; i < size; ++i) {
      sum += data_[i];
    }
    return (sum / (1.0 * size));
  }

  int size() const { return rows_ * cols_; }

  void resize(int newrows_, int newcols_) {
    if (newrows_ == rows_ && newcols_ == cols_) {
      return; // Nothing to do
    }

    deallocate_aligned(data_);
    rows_ = newrows_;
    cols_ = newcols_;

    int size = rows_ * cols_;
    data_ = size > 0 ? allocate_aligned(size) : nullptr;
  }

  static void dot(const Matrix &a, const Matrix &b, Matrix &result) {
    if (a.cols_ != b.rows_ || result.rows_ != a.rows_ || result.cols_ != b.cols_) {
      throw std::invalid_argument(
          "Matrix dimensions must match for dot product.");
    }

    result.fill(0.0); // Initialize to zero for accumulation
    
    // Use AVX2 for float matrices and larger sizes
    if constexpr (std::is_same_v<T, float>) {
      if (a.rows_ * b.cols_ > 1024) { // Use AVX2 for larger matrices
        a.avx2_matmul(a, b, result);
        return;
      }
    }
    
    // Fallback to standard algorithm
    const T* a_data = a.data_;
    const T* b_data = b.data_;
    T* result_data = result.data_;
    
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
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
