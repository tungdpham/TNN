#pragma once

#include <cstdint>
#include <stdexcept>
#ifdef USE_CUDA
#include <cuda_fp16.h> // IWYU pragma: export
#endif

namespace tnn {
#if defined(USE_CUDA)
typedef __half fp16;
typedef __nv_bfloat16 bf16;
#else
struct fp16 {
  uint16_t data;

  fp16() : data(0) {}
  explicit fp16(uint16_t d) : data(d) {}
};
struct bf16 {
  uint16_t data;

  bf16() : data(0) {}
  explicit bf16(uint16_t d) : data(d) {}
};
#endif
typedef float fp32;
typedef double fp64;

template <typename T> struct TypeTraits;

template <> struct TypeTraits<fp16> {
  static constexpr const char *name = "fp16";
  static const float epsilon;
  using ComputePrecision = fp32;
  using HigherPrecision = fp32;
};

template <> struct TypeTraits<bf16> {
  static constexpr const char *name = "bf16";
  static const float epsilon;
  using ComputePrecision = fp32;
  using HigherPrecision = fp32;
};

template <> struct TypeTraits<fp32> {
  static constexpr const char *name = "float32";
  static const float epsilon;
  using ComputePrecision = fp32;
  using HigherPrecision = fp64;
};

template <> struct TypeTraits<fp64> {
  static constexpr const char *name = "float64";
  static const float epsilon;
  using ComputePrecision = fp64;
  using HigherPrecision = fp64;
};

enum class DType_t : uint32_t { BYTE, BF16, FP16, FP32, FP64, UNKNOWN };

template <typename T> constexpr DType_t dtype_of() { return DType_t::UNKNOWN; }
template <> constexpr DType_t dtype_of<fp16>() { return DType_t::FP16; }
template <> constexpr DType_t dtype_of<bf16>() { return DType_t::BF16; }
template <> constexpr DType_t dtype_of<float>() { return DType_t::FP32; }
template <> constexpr DType_t dtype_of<double>() { return DType_t::FP64; }

inline float dtype_eps(DType_t dtype) {
  switch (dtype) {
  case DType_t::BF16:
    return TypeTraits<bf16>::epsilon;
  case DType_t::FP16:
    return TypeTraits<fp16>::epsilon;
  case DType_t::FP32:
    return TypeTraits<fp32>::epsilon;
  case DType_t::FP64:
    return TypeTraits<fp64>::epsilon;
  default:
    throw std::runtime_error("Unknown data type for dtype_eps");
  }
}

inline size_t get_dtype_size(DType_t dtype) {
  switch (dtype) {
  case DType_t::BYTE:
    return sizeof(uint8_t);
  case DType_t::FP16:
    return sizeof(fp16);
  case DType_t::FP32:
    return sizeof(fp32);
  case DType_t::FP64:
    return sizeof(fp64);
  default:
    throw std::runtime_error("Unknown data type for get_dtype_size");
  }
}

} // namespace tnn