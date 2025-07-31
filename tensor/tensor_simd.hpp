#pragma once

#include "tensor_view.hpp"
#include "../matrix/matrix.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <execution>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef __AVX__
#include <immintrin.h>
#endif

// SIMD alignment constants
static constexpr size_t SIMD_ALIGNMENT = 64; // AVX-512 alignment
static constexpr size_t DEFAULT_ALIGNMENT = 32; // AVX2 alignment

template <typename T = float, Layout layout = NCHW> 
class Tensor {
  static_assert(std::is_arithmetic_v<T>, "Tensor type must be arithmetic");
  static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
                "Tensor type must be floating point or integral");

private:
  using View = TensorView<T, layout>;
  static constexpr size_t dims = View::dims;
  
  alignas(SIMD_ALIGNMENT) size_t shape_[dims];  // Shape of the tensor
  alignas(SIMD_ALIGNMENT) size_t strides_[dims]; // Strides for each dimension
  
  T* data_;
  size_t data_size_;      // Total number of elements in the tensor
  size_t allocated_size_; // Actual allocated size (may be larger for alignment)
  
  // Memory management with alignment
  static constexpr size_t get_alignment() {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
      return SIMD_ALIGNMENT;
    }
    return DEFAULT_ALIGNMENT;
  }

  inline void compute_strides() noexcept { 
    View::compute_strides(strides_, shape_); 
  }

  inline size_t compute_index(std::initializer_list<size_t> indices) const {
    if (indices.size() != dims) [[unlikely]] {
      throw std::out_of_range("Index dimension mismatch");
    }
    
    size_t index = 0;
    auto it = indices.begin();
    for (size_t i = 0; i < dims; ++i, ++it) {
      if (*it >= shape_[i]) [[unlikely]] {
        throw std::out_of_range("Index out of range for tensor dimensions");
      }
      index += (*it) * strides_[i];
    }
    return index;
  }

  template<typename... Indices>
  inline size_t compute_index_variadic(Indices... indices) const noexcept {
    static_assert(sizeof...(indices) == dims, "Number of indices must match tensor dimensions");
    size_t index = 0;
    size_t stride_idx = 0;
    ((index += indices * strides_[stride_idx++]), ...);
    return index;
  }

  inline void allocate_aligned_memory() {
    data_size_ = std::accumulate(shape_, shape_ + dims, 1UL, std::multiplies<size_t>());
    
    // Round up to nearest multiple of SIMD elements for better vectorization
    constexpr size_t simd_elements = get_alignment() / sizeof(T);
    allocated_size_ = ((data_size_ + simd_elements - 1) / simd_elements) * simd_elements;
    
    // Use aligned allocation for SIMD operations
    data_ = static_cast<T*>(std::aligned_alloc(get_alignment(), allocated_size_ * sizeof(T)));
    if (!data_) [[unlikely]] {
      throw std::bad_alloc();
    }
    
    // Initialize with zeros
    std::fill_n(std::execution::par_unseq, data_, data_size_, T(0));
  }

  inline void deallocate_data() noexcept {
    std::free(data_);
    data_ = nullptr;
    data_size_ = 0;
    allocated_size_ = 0;
  }

public:
  // Default constructor
  Tensor() noexcept : data_(nullptr), data_size_(0), allocated_size_(0) {
    std::fill_n(shape_, dims, 0);
    std::fill_n(strides_, dims, 0);
  }

  // 4D constructor (NCHW/NHWC)
  Tensor(size_t batch, size_t channels, size_t height, size_t width) 
    requires(dims == 4) {
    shape_[0] = batch;
    shape_[1] = channels;
    shape_[2] = height;
    shape_[3] = width;
    compute_strides();
    allocate_aligned_memory();
  }

  // 5D constructor (NCDHW/NDHWC)
  Tensor(size_t batch, size_t channels, size_t depth, size_t height, size_t width)
    requires(dims == 5) {
    shape_[0] = batch;
    shape_[1] = channels;
    shape_[2] = depth;
    shape_[3] = height;
    shape_[4] = width;
    compute_strides();
    allocate_aligned_memory();
  }

  // Shape-based constructors
  explicit Tensor(const std::vector<size_t>& shape) {
    if (shape.size() != dims) [[unlikely]] {
      throw std::invalid_argument("Shape must have " + std::to_string(dims) + " dimensions");
    }
    std::copy(shape.begin(), shape.end(), shape_);
    compute_strides();
    allocate_aligned_memory();
  }

  explicit Tensor(std::initializer_list<size_t> shape) {
    if (shape.size() != dims) [[unlikely]] {
      throw std::invalid_argument("Shape must have " + std::to_string(dims) + " dimensions");
    }
    std::copy(shape.begin(), shape.end(), shape_);
    compute_strides();
    allocate_aligned_memory();
  }

  // Constructor with data
  Tensor(const std::vector<size_t>& shape, const std::vector<T>& data) {
    if (shape.size() != dims) [[unlikely]] {
      throw std::invalid_argument("Shape must have " + std::to_string(dims) + " dimensions");
    }
    std::copy(shape.begin(), shape.end(), shape_);
    compute_strides();
    
    const size_t expected_size = std::accumulate(shape_, shape_ + dims, 1UL, std::multiplies<size_t>());
    if (data.size() != expected_size) [[unlikely]] {
      throw std::invalid_argument("Data size doesn't match tensor shape");
    }
    
    allocate_aligned_memory();
    std::copy(std::execution::par_unseq, data.begin(), data.begin() + data_size_, data_);
  }

  // Destructor
  ~Tensor() noexcept {
    deallocate_data();
  }

  // Copy constructor
  Tensor(const Tensor& other) {
    std::copy(other.shape_, other.shape_ + dims, shape_);
    compute_strides();
    data_size_ = other.data_size_;
    allocated_size_ = other.allocated_size_;
    
    if (data_size_ > 0) {
      data_ = static_cast<T*>(std::aligned_alloc(get_alignment(), allocated_size_ * sizeof(T)));
      if (!data_) [[unlikely]] {
        throw std::bad_alloc();
      }
      std::copy(std::execution::par_unseq, other.data_, other.data_ + data_size_, data_);
    } else {
      data_ = nullptr;
    }
  }

  // Move constructor
  Tensor(Tensor&& other) noexcept {
    std::copy(other.shape_, other.shape_ + dims, shape_);
    std::copy(other.strides_, other.strides_ + dims, strides_);
    data_ = std::exchange(other.data_, nullptr);
    data_size_ = std::exchange(other.data_size_, 0);
    allocated_size_ = std::exchange(other.allocated_size_, 0);
  }

  // Copy assignment
  Tensor& operator=(const Tensor& other) {
    if (this != &other) {
      deallocate_data();
      std::copy(other.shape_, other.shape_ + dims, shape_);
      compute_strides();
      data_size_ = other.data_size_;
      allocated_size_ = other.allocated_size_;
      
      if (data_size_ > 0) {
        data_ = static_cast<T*>(std::aligned_alloc(get_alignment(), allocated_size_ * sizeof(T)));
        if (!data_) [[unlikely]] {
          throw std::bad_alloc();
        }
        std::copy(std::execution::par_unseq, other.data_, other.data_ + data_size_, data_);
      } else {
        data_ = nullptr;
      }
    }
    return *this;
  }

  // Move assignment
  Tensor& operator=(Tensor&& other) noexcept {
    if (this != &other) {
      deallocate_data();
      std::copy(other.shape_, other.shape_ + dims, shape_);
      std::copy(other.strides_, other.strides_ + dims, strides_);
      data_ = std::exchange(other.data_, nullptr);
      data_size_ = std::exchange(other.data_size_, 0);
      allocated_size_ = std::exchange(other.allocated_size_, 0);
    }
    return *this;
  }

  // Element access operators
  template<typename... Indices>
  T& operator()(Indices... indices) noexcept {
    return data_[compute_index_variadic(indices...)];
  }

  template<typename... Indices>
  const T& operator()(Indices... indices) const noexcept {
    return data_[compute_index_variadic(indices...)];
  }

  // Shape and size information
  [[nodiscard]] constexpr size_t num_dimensions() const noexcept { return dims; }
  [[nodiscard]] size_t size() const noexcept { return data_size_; }
  [[nodiscard]] size_t allocated_size() const noexcept { return allocated_size_; }
  
  [[nodiscard]] std::vector<size_t> shape() const {
    return std::vector<size_t>(shape_, shape_ + dims);
  }
  
  [[nodiscard]] const size_t* shape_ptr() const noexcept { return shape_; }
  [[nodiscard]] const size_t* strides_ptr() const noexcept { return strides_; }
  
  [[nodiscard]] size_t dimension(size_t index) const {
    if (index >= dims) [[unlikely]] {
      throw std::out_of_range("Dimension index out of range");
    }
    return shape_[index];
  }
  
  [[nodiscard]] size_t stride(size_t index) const {
    if (index >= dims) [[unlikely]] {
      throw std::out_of_range("Stride index out of range");
    }
    return strides_[index];
  }

  // Convenience accessors for common dimensions
  [[nodiscard]] size_t batch_size() const noexcept { return shape_[0]; }

  [[nodiscard]] size_t channels() const noexcept { return shape_[1]; }
  
  [[nodiscard]] size_t height() const noexcept 
    requires(dims == 4 || dims == 5) {
    if constexpr (dims == 4) {
      return shape_[2];
    } else {
      return shape_[3]; // For 5D tensors
    }
  }
  
  [[nodiscard]] size_t width() const noexcept 
    requires(dims == 4 || dims == 5) {
    if constexpr (dims == 4) {
      return shape_[3];
    } else {
      return shape_[4]; // For 5D tensors
    }
  }
  
  [[nodiscard]] size_t depth() const noexcept 
    requires(dims == 5) {
    return shape_[2];
  }

  // Data access
  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  
  // Check if memory is properly aligned for SIMD
  [[nodiscard]] bool is_simd_aligned() const noexcept {
    return (reinterpret_cast<uintptr_t>(data_) % get_alignment()) == 0;
  }

  // Fill operations with SIMD optimization
  void fill(T value) noexcept {
    if constexpr (std::is_same_v<T, float>) {
      fill_simd_float(value);
    } else if constexpr (std::is_same_v<T, double>) {
      fill_simd_double(value);
    } else {
      std::fill(std::execution::par_unseq, data_, data_ + data_size_, value);
    }
  }

private:
  // SIMD-optimized fill operations
  void fill_simd_float(float value) noexcept {
#ifdef __AVX__
    const __m256 vec_value = _mm256_set1_ps(value);
    const size_t simd_size = 8; // 8 floats per AVX register
    const size_t simd_iterations = data_size_ / simd_size;
    
    for (size_t i = 0; i < simd_iterations; ++i) {
      _mm256_store_ps(data_ + i * simd_size, vec_value);
    }
    
    // Handle remaining elements
    const size_t remaining = data_size_ % simd_size;
    if (remaining > 0) {
      std::fill(data_ + simd_iterations * simd_size, data_ + data_size_, value);
    }
#else
    std::fill(std::execution::par_unseq, data_, data_ + data_size_, value);
#endif
  }

  void fill_simd_double(double value) noexcept {
#ifdef __AVX__
    const __m256d vec_value = _mm256_set1_pd(value);
    const size_t simd_size = 4; // 4 doubles per AVX register
    const size_t simd_iterations = data_size_ / simd_size;
    
    for (size_t i = 0; i < simd_iterations; ++i) {
      _mm256_store_pd(data_ + i * simd_size, vec_value);
    }
    
    // Handle remaining elements
    const size_t remaining = data_size_ % simd_size;
    if (remaining > 0) {
      std::fill(data_ + simd_iterations * simd_size, data_ + data_size_, value);
    }
#else
    std::fill(std::execution::par_unseq, data_, data_ + data_size_, value);
#endif
  }

public:
  // Random initialization
  void fill_random_uniform(T min_val = T(-1), T max_val = T(1)) {
    thread_local std::mt19937 gen(std::random_device{}());
    
    if constexpr (std::is_floating_point_v<T>) {
      std::uniform_real_distribution<T> dis(min_val, max_val);
      std::generate(std::execution::par_unseq, data_, data_ + data_size_, 
                   [&] { return dis(gen); });
    } else {
      std::uniform_int_distribution<T> dis(min_val, max_val);
      std::generate(std::execution::par_unseq, data_, data_ + data_size_, 
                   [&] { return dis(gen); });
    }
  }

  void fill_random_normal(T mean = T(0), T stddev = T(1)) 
    requires(std::is_floating_point_v<T>) {
    thread_local std::mt19937 gen(std::random_device{}());
    std::normal_distribution<T> dis(mean, stddev);
    std::generate(std::execution::par_unseq, data_, data_ + data_size_, 
                 [&] { return dis(gen); });
  }

  // Clone operation
  [[nodiscard]] Tensor clone() const {
    return Tensor(*this);
  }

  // Basic arithmetic operations (SIMD-ready)
  Tensor operator+(const Tensor& other) const {
    if (!shapes_match(other)) [[unlikely]] {
      throw std::invalid_argument("Tensor shapes must match for addition");
    }
    
    Tensor result(shape());
    add_simd(data_, other.data_, result.data_, data_size_);
    return result;
  }

  Tensor operator-(const Tensor& other) const {
    if (!shapes_match(other)) [[unlikely]] {
      throw std::invalid_argument("Tensor shapes must match for subtraction");
    }
    
    Tensor result(shape());
    sub_simd(data_, other.data_, result.data_, data_size_);
    return result;
  }

  Tensor operator*(T scalar) const {
    Tensor result(shape());
    mul_scalar_simd(data_, scalar, result.data_, data_size_);
    return result;
  }

  Tensor& operator+=(const Tensor& other) {
    if (!shapes_match(other)) [[unlikely]] {
      throw std::invalid_argument("Tensor shapes must match for addition");
    }
    
    add_inplace_simd(data_, other.data_, data_size_);
    return *this;
  }

  Tensor& operator*=(T scalar) {
    mul_scalar_inplace_simd(data_, scalar, data_size_);
    return *this;
  }

  Tensor& operator-=(const Tensor& other) {
    if (!shapes_match(other)) [[unlikely]] {
      throw std::invalid_argument("Tensor shapes must match for subtraction");
    }
    
    sub_inplace_simd(data_, other.data_, data_size_);
    return *this;
  }

  Tensor operator/(T scalar) const {
    if (scalar == T(0)) [[unlikely]] {
      throw std::invalid_argument("Division by zero");
    }
    
    Tensor result(shape());
    div_scalar_simd(data_, scalar, result.data_, data_size_);
    return result;
  }

  Tensor& operator/=(T scalar) {
    if (scalar == T(0)) [[unlikely]] {
      throw std::invalid_argument("Division by zero");
    }
    
    div_scalar_inplace_simd(data_, scalar, data_size_);
    return *this;
  }

  // Statistical operations with SIMD optimization
  [[nodiscard]] T sum() const noexcept {
    if constexpr (std::is_same_v<T, float>) {
      return sum_simd_float();
    } else if constexpr (std::is_same_v<T, double>) {
      return sum_simd_double();
    } else {
      return std::reduce(std::execution::par_unseq, data_, data_ + data_size_, T(0));
    }
  }

  [[nodiscard]] T mean() const noexcept {
    return sum() / static_cast<T>(data_size_);
  }

  // Reshape operation (preserves total size)
  [[nodiscard]] Tensor reshape(const std::vector<size_t>& new_shape) const {
    if (new_shape.size() != dims) [[unlikely]] {
      throw std::invalid_argument("New shape must have " + std::to_string(dims) + " dimensions");
    }
    
    const size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1UL, std::multiplies<size_t>());
    if (new_size != data_size_) [[unlikely]] {
      throw std::invalid_argument("New shape must preserve total number of elements");
    }
    
    return Tensor(new_shape, std::vector<T>(data_, data_ + data_size_));
  }

  // Slicing operations
  [[nodiscard]] Tensor slice_batch(size_t start_idx, size_t count) const {
    if (start_idx + count > batch_size()) [[unlikely]] {
      throw std::out_of_range("Batch slice out of range");
    }
    
    std::vector<size_t> new_shape(shape_, shape_ + dims);
    new_shape[0] = count;
    
    Tensor result(new_shape);
    
    if constexpr (dims == 4) {
      const size_t batch_stride = channels() * height() * width();
      std::copy(std::execution::par_unseq,
                data_ + start_idx * batch_stride,
                data_ + (start_idx + count) * batch_stride,
                result.data_);
    } else if constexpr (dims == 5) {
      const size_t batch_stride = channels() * depth() * height() * width();
      std::copy(std::execution::par_unseq,
                data_ + start_idx * batch_stride,
                data_ + (start_idx + count) * batch_stride,
                result.data_);
    }
    
    return result;
  }

  // Debugging and utility functions
  void print_info() const {
    std::cout << "Tensor<" << typeid(T).name() << "> Shape: [";
    for (size_t i = 0; i < dims; ++i) {
      std::cout << shape_[i];
      if (i < dims - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Layout: ";
    switch (layout) {
      case NCHW: std::cout << "NCHW"; break;
      case NHWC: std::cout << "NHWC"; break;
      case NCDHW: std::cout << "NCDHW"; break;
      case NDHWC: std::cout << "NDHWC"; break;
    }
    std::cout << ", Dimensions: " << dims << "D" << std::endl;
    std::cout << "Size: " << data_size_ << ", Allocated: " << allocated_size_ << std::endl;
    std::cout << "SIMD Aligned: " << (is_simd_aligned() ? "Yes" : "No") << std::endl;
    if (data_size_ > 0) {
      std::cout << "Mean: " << mean() << std::endl;
    }
  }

private:
  // Helper function to check if shapes match
  [[nodiscard]] bool shapes_match(const Tensor& other) const noexcept {
    return std::equal(shape_, shape_ + dims, other.shape_);
  }

  // SIMD arithmetic operations
  static void add_simd(const T* a, const T* b, T* result, size_t size) noexcept {
    if constexpr (std::is_same_v<T, float>) {
      add_simd_float(a, b, result, size);
    } else if constexpr (std::is_same_v<T, double>) {
      add_simd_double(a, b, result, size);
    } else {
      std::transform(std::execution::par_unseq, a, a + size, b, result, std::plus<T>());
    }
  }

  static void add_simd_float(const float* a, const float* b, float* result, size_t size) noexcept {
#ifdef __AVX__
    const size_t simd_size = 8;
    const size_t simd_iterations = size / simd_size;
    
    for (size_t i = 0; i < simd_iterations; ++i) {
      const size_t offset = i * simd_size;
      __m256 va = _mm256_load_ps(a + offset);
      __m256 vb = _mm256_load_ps(b + offset);
      __m256 vr = _mm256_add_ps(va, vb);
      _mm256_store_ps(result + offset, vr);
    }
    
    // Handle remaining elements
    for (size_t i = simd_iterations * simd_size; i < size; ++i) {
      result[i] = a[i] + b[i];
    }
#else
    std::transform(std::execution::par_unseq, a, a + size, b, result, std::plus<float>());
#endif
  }

  static void add_simd_double(const double* a, const double* b, double* result, size_t size) noexcept {
#ifdef __AVX__
    const size_t simd_size = 4;
    const size_t simd_iterations = size / simd_size;
    
    for (size_t i = 0; i < simd_iterations; ++i) {
      const size_t offset = i * simd_size;
      __m256d va = _mm256_load_pd(a + offset);
      __m256d vb = _mm256_load_pd(b + offset);
      __m256d vr = _mm256_add_pd(va, vb);
      _mm256_store_pd(result + offset, vr);
    }
    
    // Handle remaining elements
    for (size_t i = simd_iterations * simd_size; i < size; ++i) {
      result[i] = a[i] + b[i];
    }
#else
    std::transform(std::execution::par_unseq, a, a + size, b, result, std::plus<double>());
#endif
  }

  static void sub_simd(const T* a, const T* b, T* result, size_t size) noexcept {
    if constexpr (std::is_same_v<T, float>) {
      sub_simd_float(a, b, result, size);
    } else if constexpr (std::is_same_v<T, double>) {
      sub_simd_double(a, b, result, size);
    } else {
      std::transform(std::execution::par_unseq, a, a + size, b, result, std::minus<T>());
    }
  }

  static void sub_simd_float(const float* a, const float* b, float* result, size_t size) noexcept {
#ifdef __AVX__
    const size_t simd_size = 8;
    const size_t simd_iterations = size / simd_size;
    
    for (size_t i = 0; i < simd_iterations; ++i) {
      const size_t offset = i * simd_size;
      __m256 va = _mm256_load_ps(a + offset);
      __m256 vb = _mm256_load_ps(b + offset);
      __m256 vr = _mm256_sub_ps(va, vb);
      _mm256_store_ps(result + offset, vr);
    }
    
    // Handle remaining elements
    for (size_t i = simd_iterations * simd_size; i < size; ++i) {
      result[i] = a[i] - b[i];
    }
#else
    std::transform(std::execution::par_unseq, a, a + size, b, result, std::minus<float>());
#endif
  }

  static void sub_simd_double(const double* a, const double* b, double* result, size_t size) noexcept {
#ifdef __AVX__
    const size_t simd_size = 4;
    const size_t simd_iterations = size / simd_size;
    
    for (size_t i = 0; i < simd_iterations; ++i) {
      const size_t offset = i * simd_size;
      __m256d va = _mm256_load_pd(a + offset);
      __m256d vb = _mm256_load_pd(b + offset);
      __m256d vr = _mm256_sub_pd(va, vb);
      _mm256_store_pd(result + offset, vr);
    }
    
    // Handle remaining elements
    for (size_t i = simd_iterations * simd_size; i < size; ++i) {
      result[i] = a[i] - b[i];
    }
#else
    std::transform(std::execution::par_unseq, a, a + size, b, result, std::minus<double>());
#endif
  }

  static void mul_scalar_simd(const T* a, T scalar, T* result, size_t size) noexcept {
    if constexpr (std::is_same_v<T, float>) {
      mul_scalar_simd_float(a, scalar, result, size);
    } else if constexpr (std::is_same_v<T, double>) {
      mul_scalar_simd_double(a, scalar, result, size);
    } else {
      std::transform(std::execution::par_unseq, a, a + size, result, 
                    [scalar](T val) { return val * scalar; });
    }
  }

  static void mul_scalar_simd_float(const float* a, float scalar, float* result, size_t size) noexcept {
#ifdef __AVX__
    const __m256 vec_scalar = _mm256_set1_ps(scalar);
    const size_t simd_size = 8;
    const size_t simd_iterations = size / simd_size;
    
    for (size_t i = 0; i < simd_iterations; ++i) {
      const size_t offset = i * simd_size;
      __m256 va = _mm256_load_ps(a + offset);
      __m256 vr = _mm256_mul_ps(va, vec_scalar);
      _mm256_store_ps(result + offset, vr);
    }
    
    // Handle remaining elements
    for (size_t i = simd_iterations * simd_size; i < size; ++i) {
      result[i] = a[i] * scalar;
    }
#else
    std::transform(std::execution::par_unseq, a, a + size, result, 
                  [scalar](float val) { return val * scalar; });
#endif
  }

  static void mul_scalar_simd_double(const double* a, double scalar, double* result, size_t size) noexcept {
#ifdef __AVX__
    const __m256d vec_scalar = _mm256_set1_pd(scalar);
    const size_t simd_size = 4;
    const size_t simd_iterations = size / simd_size;
    
    for (size_t i = 0; i < simd_iterations; ++i) {
      const size_t offset = i * simd_size;
      __m256d va = _mm256_load_pd(a + offset);
      __m256d vr = _mm256_mul_pd(va, vec_scalar);
      _mm256_store_pd(result + offset, vr);
    }
    
    // Handle remaining elements
    for (size_t i = simd_iterations * simd_size; i < size; ++i) {
      result[i] = a[i] * scalar;
    }
#else
    std::transform(std::execution::par_unseq, a, a + size, result, 
                  [scalar](double val) { return val * scalar; });
#endif
  }

  static void add_inplace_simd(T* a, const T* b, size_t size) noexcept {
    if constexpr (std::is_same_v<T, float>) {
      add_inplace_simd_float(a, b, size);
    } else if constexpr (std::is_same_v<T, double>) {
      add_inplace_simd_double(a, b, size);
    } else {
      std::transform(std::execution::par_unseq, a, a + size, b, a, std::plus<T>());
    }
  }

  static void add_inplace_simd_float(float* a, const float* b, size_t size) noexcept {
#ifdef __AVX__
    const size_t simd_size = 8;
    const size_t simd_iterations = size / simd_size;
    
    for (size_t i = 0; i < simd_iterations; ++i) {
      const size_t offset = i * simd_size;
      __m256 va = _mm256_load_ps(a + offset);
      __m256 vb = _mm256_load_ps(b + offset);
      __m256 vr = _mm256_add_ps(va, vb);
      _mm256_store_ps(a + offset, vr);
    }
    
    // Handle remaining elements
    for (size_t i = simd_iterations * simd_size; i < size; ++i) {
      a[i] += b[i];
    }
#else
    std::transform(std::execution::par_unseq, a, a + size, b, a, std::plus<float>());
#endif
  }

  static void add_inplace_simd_double(double* a, const double* b, size_t size) noexcept {
#ifdef __AVX__
    const size_t simd_size = 4;
    const size_t simd_iterations = size / simd_size;
    
    for (size_t i = 0; i < simd_iterations; ++i) {
      const size_t offset = i * simd_size;
      __m256d va = _mm256_load_pd(a + offset);
      __m256d vb = _mm256_load_pd(b + offset);
      __m256d vr = _mm256_add_pd(va, vb);
      _mm256_store_pd(a + offset, vr);
    }
    
    // Handle remaining elements
    for (size_t i = simd_iterations * simd_size; i < size; ++i) {
      a[i] += b[i];
    }
#else
    std::transform(std::execution::par_unseq, a, a + size, b, a, std::plus<double>());
#endif
  }

  static void mul_scalar_inplace_simd(T* a, T scalar, size_t size) noexcept {
    if constexpr (std::is_same_v<T, float>) {
      mul_scalar_inplace_simd_float(a, scalar, size);
    } else if constexpr (std::is_same_v<T, double>) {
      mul_scalar_inplace_simd_double(a, scalar, size);
    } else {
      std::transform(std::execution::par_unseq, a, a + size, a, 
                    [scalar](T val) { return val * scalar; });
    }
  }

  static void mul_scalar_inplace_simd_float(float* a, float scalar, size_t size) noexcept {
#ifdef __AVX__
    const __m256 vec_scalar = _mm256_set1_ps(scalar);
    const size_t simd_size = 8;
    const size_t simd_iterations = size / simd_size;
    
    for (size_t i = 0; i < simd_iterations; ++i) {
      const size_t offset = i * simd_size;
      __m256 va = _mm256_load_ps(a + offset);
      __m256 vr = _mm256_mul_ps(va, vec_scalar);
      _mm256_store_ps(a + offset, vr);
    }
    
    // Handle remaining elements
    for (size_t i = simd_iterations * simd_size; i < size; ++i) {
      a[i] *= scalar;
    }
#else
    std::transform(std::execution::par_unseq, a, a + size, a, 
                  [scalar](float val) { return val * scalar; });
#endif
  }

  static void mul_scalar_inplace_simd_double(double* a, double scalar, size_t size) noexcept {
#ifdef __AVX__
    const __m256d vec_scalar = _mm256_set1_pd(scalar);
    const size_t simd_size = 4;
    const size_t simd_iterations = size / simd_size;
    
    for (size_t i = 0; i < simd_iterations; ++i) {
      const size_t offset = i * simd_size;
      __m256d va = _mm256_load_pd(a + offset);
      __m256d vr = _mm256_mul_pd(va, vec_scalar);
      _mm256_store_pd(a + offset, vr);
    }
    
    // Handle remaining elements
    for (size_t i = simd_iterations * simd_size; i < size; ++i) {
      a[i] *= scalar;
    }
#else
    std::transform(std::execution::par_unseq, a, a + size, a, 
                  [scalar](double val) { return val * scalar; });
#endif
  }

  static void sub_inplace_simd(T* a, const T* b, size_t size) noexcept {
    if constexpr (std::is_same_v<T, float>) {
      sub_inplace_simd_float(a, b, size);
    } else if constexpr (std::is_same_v<T, double>) {
      sub_inplace_simd_double(a, b, size);
    } else {
      std::transform(std::execution::par_unseq, a, a + size, b, a, std::minus<T>());
    }
  }

  static void sub_inplace_simd_float(float* a, const float* b, size_t size) noexcept {
#ifdef __AVX__
    const size_t simd_size = 8;
    const size_t simd_iterations = size / simd_size;
    
    for (size_t i = 0; i < simd_iterations; ++i) {
      const size_t offset = i * simd_size;
      __m256 va = _mm256_load_ps(a + offset);
      __m256 vb = _mm256_load_ps(b + offset);
      __m256 vr = _mm256_sub_ps(va, vb);
      _mm256_store_ps(a + offset, vr);
    }
    
    // Handle remaining elements
    for (size_t i = simd_iterations * simd_size; i < size; ++i) {
      a[i] -= b[i];
    }
#else
    std::transform(std::execution::par_unseq, a, a + size, b, a, std::minus<float>());
#endif
  }

  static void sub_inplace_simd_double(double* a, const double* b, size_t size) noexcept {
#ifdef __AVX__
    const size_t simd_size = 4;
    const size_t simd_iterations = size / simd_size;
    
    for (size_t i = 0; i < simd_iterations; ++i) {
      const size_t offset = i * simd_size;
      __m256d va = _mm256_load_pd(a + offset);
      __m256d vb = _mm256_load_pd(b + offset);
      __m256d vr = _mm256_sub_pd(va, vb);
      _mm256_store_pd(a + offset, vr);
    }
    
    // Handle remaining elements
    for (size_t i = simd_iterations * simd_size; i < size; ++i) {
      a[i] -= b[i];
    }
#else
    std::transform(std::execution::par_unseq, a, a + size, b, a, std::minus<double>());
#endif
  }

  static void div_scalar_simd(const T* a, T scalar, T* result, size_t size) noexcept {
    if constexpr (std::is_same_v<T, float>) {
      div_scalar_simd_float(a, scalar, result, size);
    } else if constexpr (std::is_same_v<T, double>) {
      div_scalar_simd_double(a, scalar, result, size);
    } else {
      std::transform(std::execution::par_unseq, a, a + size, result, 
                    [scalar](T val) { return val / scalar; });
    }
  }

  static void div_scalar_simd_float(const float* a, float scalar, float* result, size_t size) noexcept {
#ifdef __AVX__
    const __m256 vec_scalar = _mm256_set1_ps(scalar);
    const size_t simd_size = 8;
    const size_t simd_iterations = size / simd_size;
    
    for (size_t i = 0; i < simd_iterations; ++i) {
      const size_t offset = i * simd_size;
      __m256 va = _mm256_load_ps(a + offset);
      __m256 vr = _mm256_div_ps(va, vec_scalar);
      _mm256_store_ps(result + offset, vr);
    }
    
    // Handle remaining elements
    for (size_t i = simd_iterations * simd_size; i < size; ++i) {
      result[i] = a[i] / scalar;
    }
#else
    std::transform(std::execution::par_unseq, a, a + size, result, 
                  [scalar](float val) { return val / scalar; });
#endif
  }

  static void div_scalar_simd_double(const double* a, double scalar, double* result, size_t size) noexcept {
#ifdef __AVX__
    const __m256d vec_scalar = _mm256_set1_pd(scalar);
    const size_t simd_size = 4;
    const size_t simd_iterations = size / simd_size;
    
    for (size_t i = 0; i < simd_iterations; ++i) {
      const size_t offset = i * simd_size;
      __m256d va = _mm256_load_pd(a + offset);
      __m256d vr = _mm256_div_pd(va, vec_scalar);
      _mm256_store_pd(result + offset, vr);
    }
    
    // Handle remaining elements
    for (size_t i = simd_iterations * simd_size; i < size; ++i) {
      result[i] = a[i] / scalar;
    }
#else
    std::transform(std::execution::par_unseq, a, a + size, result, 
                  [scalar](double val) { return val / scalar; });
#endif
  }

  static void div_scalar_inplace_simd(T* a, T scalar, size_t size) noexcept {
    if constexpr (std::is_same_v<T, float>) {
      div_scalar_inplace_simd_float(a, scalar, size);
    } else if constexpr (std::is_same_v<T, double>) {
      div_scalar_inplace_simd_double(a, scalar, size);
    } else {
      std::transform(std::execution::par_unseq, a, a + size, a, 
                    [scalar](T val) { return val / scalar; });
    }
  }

  static void div_scalar_inplace_simd_float(float* a, float scalar, size_t size) noexcept {
#ifdef __AVX__
    const __m256 vec_scalar = _mm256_set1_ps(scalar);
    const size_t simd_size = 8;
    const size_t simd_iterations = size / simd_size;
    
    for (size_t i = 0; i < simd_iterations; ++i) {
      const size_t offset = i * simd_size;
      __m256 va = _mm256_load_ps(a + offset);
      __m256 vr = _mm256_div_ps(va, vec_scalar);
      _mm256_store_ps(a + offset, vr);
    }
    
    // Handle remaining elements
    for (size_t i = simd_iterations * simd_size; i < size; ++i) {
      a[i] /= scalar;
    }
#else
    std::transform(std::execution::par_unseq, a, a + size, a, 
                  [scalar](float val) { return val / scalar; });
#endif
  }

  static void div_scalar_inplace_simd_double(double* a, double scalar, size_t size) noexcept {
#ifdef __AVX__
    const __m256d vec_scalar = _mm256_set1_pd(scalar);
    const size_t simd_size = 4;
    const size_t simd_iterations = size / simd_size;
    
    for (size_t i = 0; i < simd_iterations; ++i) {
      const size_t offset = i * simd_size;
      __m256d va = _mm256_load_pd(a + offset);
      __m256d vr = _mm256_div_pd(va, vec_scalar);
      _mm256_store_pd(a + offset, vr);
    }
    
    // Handle remaining elements
    for (size_t i = simd_iterations * simd_size; i < size; ++i) {
      a[i] /= scalar;
    }
#else
    std::transform(std::execution::par_unseq, a, a + size, a, 
                  [scalar](double val) { return val / scalar; });
#endif
  }

  // SIMD sum operations
  T sum_simd_float() const noexcept {
#ifdef __AVX__
    __m256 vec_sum = _mm256_setzero_ps();
    const size_t simd_size = 8;
    const size_t simd_iterations = data_size_ / simd_size;
    
    for (size_t i = 0; i < simd_iterations; ++i) {
      __m256 va = _mm256_load_ps(data_ + i * simd_size);
      vec_sum = _mm256_add_ps(vec_sum, va);
    }
    
    // Horizontal sum of the vector
    __m128 hi = _mm256_extractf128_ps(vec_sum, 1);
    __m128 lo = _mm256_castps256_ps128(vec_sum);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    
    T total = _mm_cvtss_f32(sum);
    
    // Handle remaining elements
    for (size_t i = simd_iterations * simd_size; i < data_size_; ++i) {
      total += data_[i];
    }
    
    return total;
#else
    return std::reduce(std::execution::par_unseq, data_, data_ + data_size_, T(0));
#endif
  }

  T sum_simd_double() const noexcept {
#ifdef __AVX__
    __m256d vec_sum = _mm256_setzero_pd();
    const size_t simd_size = 4;
    const size_t simd_iterations = data_size_ / simd_size;
    
    for (size_t i = 0; i < simd_iterations; ++i) {
      __m256d va = _mm256_load_pd(data_ + i * simd_size);
      vec_sum = _mm256_add_pd(vec_sum, va);
    }
    
    // Horizontal sum of the vector
    __m128d hi = _mm256_extractf128_pd(vec_sum, 1);
    __m128d lo = _mm256_castpd256_pd128(vec_sum);
    __m128d sum = _mm_add_pd(hi, lo);
    sum = _mm_hadd_pd(sum, sum);
    
    T total = _mm_cvtsd_f64(sum);
    
    // Handle remaining elements
    for (size_t i = simd_iterations * simd_size; i < data_size_; ++i) {
      total += data_[i];
    }
    
    return total;
#else
    return std::reduce(std::execution::par_unseq, data_, data_ + data_size_, T(0));
#endif
  }

public:
  // Padding operations
  [[nodiscard]] Tensor pad(size_t pad_h, size_t pad_w, T value = T(0)) const 
    requires(dims == 4) {
    if (pad_h == 0 && pad_w == 0) {
      return *this; // No padding needed
    }

    Tensor result(batch_size(), channels(), height() + 2 * pad_h, width() + 2 * pad_w);
    result.fill(value);

    // Copy original data to padded tensor
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
  [[nodiscard]] Tensor pad_3d(size_t pad_d, size_t pad_h, size_t pad_w, T value = T(0)) const 
    requires(dims == 5) {
    if (pad_d == 0 && pad_h == 0 && pad_w == 0) {
      return *this; // No padding needed
    }

    Tensor result(batch_size(), channels(), depth() + 2 * pad_d, 
                  height() + 2 * pad_h, width() + 2 * pad_w);
    result.fill(value);

    // Copy original data to padded tensor
    for (size_t n = 0; n < batch_size(); ++n) {
      for (size_t c = 0; c < channels(); ++c) {
        for (size_t d = 0; d < depth(); ++d) {
          for (size_t h = 0; h < height(); ++h) {
            for (size_t w = 0; w < width(); ++w) {
              result(n, c, d + pad_d, h + pad_h, w + pad_w) = (*this)(n, c, d, h, w);
            }
          }
        }
      }
    }

    return result;
  }

  // Cropping operations for 4D tensors
  [[nodiscard]] Tensor crop(size_t start_h, size_t start_w, size_t end_h, size_t end_w) const 
    requires(dims == 4) {
    if (end_h >= height() || end_w >= width() || start_h > end_h || start_w > end_w) {
      throw std::invalid_argument("Invalid crop dimensions");
    }

    const size_t new_height = end_h - start_h + 1;
    const size_t new_width = end_w - start_w + 1;

    Tensor result(batch_size(), channels(), new_height, new_width);

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
  [[nodiscard]] Tensor crop_3d(size_t start_d, size_t start_h, size_t start_w, 
                                size_t end_d, size_t end_h, size_t end_w) const 
    requires(dims == 5) {
    if (end_d >= depth() || end_h >= height() || end_w >= width() || 
        start_d > end_d || start_h > end_h || start_w > end_w) {
      throw std::invalid_argument("Invalid 3D crop dimensions");
    }

    const size_t new_depth = end_d - start_d + 1;
    const size_t new_height = end_h - start_h + 1;
    const size_t new_width = end_w - start_w + 1;

    Tensor result(batch_size(), channels(), new_depth, new_height, new_width);

    for (size_t n = 0; n < batch_size(); ++n) {
      for (size_t c = 0; c < channels(); ++c) {
        for (size_t d = 0; d < new_depth; ++d) {
          for (size_t h = 0; h < new_height; ++h) {
            for (size_t w = 0; w < new_width; ++w) {
              result(n, c, d, h, w) = (*this)(n, c, start_d + d, start_h + h, start_w + w);
            }
          }
        }
      }
    }

    return result;
  }

  // im2col operation for convolution (optimized for SIMD)
  [[nodiscard]] Matrix<T> im2col(size_t kernel_h, size_t kernel_w,
                                  size_t stride_h = 1, size_t stride_w = 1,
                                  size_t pad_h = 0, size_t pad_w = 0) const 
    requires(dims == 4) {
    
    // Apply padding if needed
    const Tensor* input_ptr = this;
    std::unique_ptr<Tensor> padded_input_storage;

    if (pad_h > 0 || pad_w > 0) {
      padded_input_storage = std::make_unique<Tensor>(pad(pad_h, pad_w));
      input_ptr = padded_input_storage.get();
    }
    const Tensor& input_tensor = *input_ptr;

    const size_t in_h = input_tensor.height();
    const size_t in_w = input_tensor.width();
    const size_t out_h = (in_h - kernel_h) / stride_h + 1;
    const size_t out_w = (in_w - kernel_w) / stride_w + 1;
    const size_t channels = input_tensor.channels();
    const size_t batch_size = input_tensor.batch_size();

    const size_t col_height = channels * kernel_h * kernel_w;
    const size_t col_width = batch_size * out_h * out_w;
    Matrix<T> col_matrix(col_height, col_width);

    // SIMD-optimized im2col operation
    #pragma omp parallel for collapse(4) if(batch_size * channels > 8)
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t c = 0; c < channels; ++c) {
        for (size_t kh = 0; kh < kernel_h; ++kh) {
          for (size_t kw = 0; kw < kernel_w; ++kw) {
            const size_t col_row_idx = (c * kernel_h + kh) * kernel_w + kw;
            for (size_t out_h_idx = 0; out_h_idx < out_h; ++out_h_idx) {
              for (size_t out_w_idx = 0; out_w_idx < out_w; ++out_w_idx) {
                const size_t in_h_idx = out_h_idx * stride_h + kh;
                const size_t in_w_idx = out_w_idx * stride_w + kw;
                const size_t col_col_idx = n * out_h * out_w + out_h_idx * out_w + out_w_idx;

                col_matrix(col_row_idx, col_col_idx) = input_tensor(n, c, in_h_idx, in_w_idx);
              }
            }
          }
        }
      }
    }
    return col_matrix;
  }

  // col2im operation for convolution backward pass
  static Tensor col2im(const Matrix<T>& col_matrix, size_t batch_size,
                       size_t channels, size_t height, size_t width,
                       size_t kernel_h, size_t kernel_w, 
                       size_t stride_h = 1, size_t stride_w = 1,
                       size_t pad_h = 0, size_t pad_w = 0) 
    requires(dims == 4) {

    const size_t padded_h = height + 2 * pad_h;
    const size_t padded_w = width + 2 * pad_w;
    const size_t output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    const size_t output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    Tensor result(batch_size, channels, padded_h, padded_w);
    result.fill(T(0));

    #pragma omp parallel for collapse(4) if(batch_size * channels > 8)
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t c = 0; c < channels; ++c) {
        for (size_t h_idx = 0; h_idx < padded_h; ++h_idx) {
          for (size_t w_idx = 0; w_idx < padded_w; ++w_idx) {
            T sum = 0;
            // Iterate through all possible kernel positions that would cover this pixel
            for (size_t kh = 0; kh < kernel_h; ++kh) {
              for (size_t kw = 0; kw < kernel_w; ++kw) {
                if (h_idx >= kh && w_idx >= kw) {
                  const size_t out_h_idx = (h_idx - kh);
                  const size_t out_w_idx = (w_idx - kw);
                  
                  if (out_h_idx % stride_h == 0 && out_w_idx % stride_w == 0) {
                    const size_t out_h_final = out_h_idx / stride_h;
                    const size_t out_w_final = out_w_idx / stride_w;
                    
                    if (out_h_final < output_h && out_w_final < output_w) {
                      const size_t col_row = (c * kernel_h + kh) * kernel_w + kw;
                      const size_t col_col = n * output_h * output_w + out_h_final * output_w + out_w_final;
                      sum += col_matrix(col_row, col_col);
                    }
                  }
                }
              }
            }
            result(n, c, h_idx, w_idx) = sum;
          }
        }
      }
    }

    // Remove padding if it was applied
    if (pad_h > 0 || pad_w > 0) {
      return result.crop(pad_h, pad_w, padded_h - pad_h - 1, padded_w - pad_w - 1);
    } else {
      return result;
    }
  }

  // Serialization operations
  void save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    save(file);
    file.close();
  }

  void save(std::ofstream& out) const {
    if (!out.is_open()) {
      throw std::runtime_error("File is not open for writing");
    }
    
    // Write tensor metadata
    const uint32_t magic_number = 0x54454E53; // "TENS" in ASCII
    const uint32_t version = 1;
    const uint32_t layout_val = static_cast<uint32_t>(layout);
    const uint32_t dims_val = static_cast<uint32_t>(dims);
    const uint32_t type_size = sizeof(T);
    
    out.write(reinterpret_cast<const char*>(&magic_number), sizeof(magic_number));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&layout_val), sizeof(layout_val));
    out.write(reinterpret_cast<const char*>(&dims_val), sizeof(dims_val));
    out.write(reinterpret_cast<const char*>(&type_size), sizeof(type_size));
    
    // Write shape
    out.write(reinterpret_cast<const char*>(shape_), dims * sizeof(size_t));
    
    // Write data size and actual data
    out.write(reinterpret_cast<const char*>(&data_size_), sizeof(data_size_));
    out.write(reinterpret_cast<const char*>(data_), data_size_ * sizeof(T));
    
    if (!out.good()) {
      throw std::runtime_error("Error writing tensor data to file");
    }
  }

  static Tensor load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    auto result = load(file);
    file.close();
    return result;
  }

  static Tensor load(std::ifstream& in) {
    if (!in.is_open()) {
      throw std::runtime_error("File is not open for reading");
    }
    
    // Read and verify metadata
    uint32_t magic_number, version, layout_val, dims_val, type_size;
    
    in.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&layout_val), sizeof(layout_val));
    in.read(reinterpret_cast<char*>(&dims_val), sizeof(dims_val));
    in.read(reinterpret_cast<char*>(&type_size), sizeof(type_size));
    
    if (magic_number != 0x54454E53) {
      throw std::runtime_error("Invalid tensor file format");
    }
    if (version != 1) {
      throw std::runtime_error("Unsupported tensor file version");
    }
    if (layout_val != static_cast<uint32_t>(layout)) {
      throw std::runtime_error("Layout mismatch in tensor file");
    }
    if (dims_val != dims) {
      throw std::runtime_error("Dimension mismatch in tensor file");
    }
    if (type_size != sizeof(T)) {
      throw std::runtime_error("Type size mismatch in tensor file");
    }
    
    // Read shape
    std::vector<size_t> shape(dims);
    in.read(reinterpret_cast<char*>(shape.data()), dims * sizeof(size_t));
    
    // Read data size and create tensor
    size_t data_size;
    in.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
    
    Tensor tensor(shape);
    if (tensor.size() != data_size) {
      throw std::runtime_error("Data size mismatch in tensor file");
    }
    
    // Read actual data
    in.read(reinterpret_cast<char*>(tensor.data()), data_size * sizeof(T));
    
    if (!in.good()) {
      throw std::runtime_error("Error reading tensor data from file");
    }
    
    return tensor;
  }

  // Layout conversion
  template<Layout new_layout>
  [[nodiscard]] Tensor<T, new_layout> as_layout() const {
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
      throw std::runtime_error("Conversion for this dimensionality not implemented.");
    }
    return result;
  }

  // Utility function to get tensor as a row-major vector
  [[nodiscard]] std::vector<T> to_vector() const {
    if constexpr (layout == NCHW || layout == NCDHW) {
      // Data is already in row-major order for these layouts
      return std::vector<T>(data_, data_ + data_size_);
    } else {
      // For NHWC/NDHWC, need to reorder data
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
      }
      return result;
    }
  }

  // Utility function to set tensor from a row-major vector
  void from_vector(const std::vector<T>& vec) {
    if (vec.size() != data_size_) {
      throw std::invalid_argument("Vector size does not match tensor size");
    }
    std::copy(std::execution::par_unseq, vec.begin(), vec.end(), data_);
  }
};

// Free functions for SIMD operations
template<typename T>
inline bool is_simd_available() {
#ifdef __AVX__
    return std::is_same_v<T, float> || std::is_same_v<T, double>;
#else
    return false;
#endif
}
